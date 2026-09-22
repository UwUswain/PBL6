"""Main training script for ISLES-2022 stroke lesion segmentation.

Orchestrates data loaders, model architecture, compound DiceJaccardLoss,
mixed-precision (AMP FP16) training, sliding-window validation, and checkpointing.
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from src.dataset.dataloader import build_kfold_dataloaders
from src.losses.builder import get_loss_function
from src.metrics.dice import compute_dice
from src.models.builder import build_model

logger = logging.getLogger("train_engine")


def setup_logger(log_file: Path) -> None:
    """Configures multi-handler logging to file and standard stream."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def parse_args() -> argparse.Namespace:
    """Parses training command-line arguments."""
    parser = argparse.ArgumentParser(description="ISLES-2022 3D Segmentation Training Engine")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/ISLES-2022",
        help="Path to ISLES-2022 BIDS dataset root.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="segresnet",
        help="Model architecture ('segresnet' or 'mednext').",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Validation fold index [0, 4] for 5-fold cross-validation.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Total number of patient-level folds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size of 3D patches for training.",
    )
    parser.add_argument(
        "--patch-h",
        type=int,
        default=96,
        help="Patch height.",
    )
    parser.add_argument(
        "--patch-w",
        type=int,
        default=96,
        help="Patch width.",
    )
    parser.add_argument(
        "--patch-d",
        type=int,
        default=32,
        help="Patch depth (slices).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="Epoch interval for validation evaluation.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments/runs",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP FP16).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a minimal smoke test (2 train batches, 1 val case) to verify pipeline.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader background workers.",
    )
    parser.add_argument(
        "--sw-overlap",
        type=float,
        default=0.25,
        help="Sliding window overlap ratio for validation inference [0.0 - 0.99].",
    )
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    max_steps: int = 0,
) -> float:
    """Runs one training epoch over the patch dataloader."""
    model.train()
    epoch_loss = 0.0
    step_count = 0

    total_steps = len(loader) if hasattr(loader, "__len__") else 0
    for step, batch_data in enumerate(loader):
        inputs = batch_data["image"].to(device, non_blocking=True)
        labels = batch_data["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        step_count += 1

        if (step + 1) % 25 == 0 or (step + 1) == total_steps:
            logger.info(
                "    [Train Batch %d/%s] Step Loss: %.4f | Running Avg Loss: %.4f",
                step + 1,
                str(total_steps) if total_steps > 0 else "?",
                loss.item(),
                epoch_loss / step_count,
            )

        if max_steps > 0 and step_count >= max_steps:
            break

    return epoch_loss / max(step_count, 1)


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    use_amp: bool,
    overlap: float = 0.25,
    max_cases: int = 0,
) -> Tuple[float, int]:
    """Runs full-volume evaluation using sliding window inference."""
    model.eval()
    total_dice = 0.0
    case_count = 0
    total_loader_cases = len(loader.dataset) if hasattr(loader, "dataset") else -1

    with torch.no_grad():
        for case_idx, batch_data in enumerate(loader):
            case_start = time.time()
            val_images = batch_data["image"].to(device, non_blocking=True)
            val_labels = batch_data["mask"].to(device, non_blocking=True)
            img_shape = tuple(val_images.shape[2:])

            logger.info(
                "  [Val Case %d/%s] Running sliding-window inference (shape: %s, overlap: %.2f)...",
                case_idx + 1,
                str(total_loader_cases) if total_loader_cases > 0 else "?",
                img_shape,
                overlap,
            )

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                val_outputs = sliding_window_inference(
                    inputs=val_images,
                    roi_size=patch_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )

            # Binarize output mask
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            dice = compute_dice(val_preds[0, 0], val_labels[0, 0])
            elapsed = time.time() - case_start

            total_dice += dice
            case_count += 1

            logger.info(
                "  [Val Case %d/%s] Finished in %.2fs | Dice Score: %.4f",
                case_idx + 1,
                str(total_loader_cases) if total_loader_cases > 0 else "?",
                elapsed,
                dice,
            )

            if max_cases > 0 and case_count >= max_cases:
                break

    mean_dice = total_dice / max(case_count, 1)
    return mean_dice, case_count


def main() -> None:
    """Main execution function for model training."""
    args = parse_args()

    # Determine execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    # Setup experiment directory and logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model_name}_fold{args.fold}_{timestamp}"
    if args.dry_run:
        exp_name = f"dry_run_{exp_name}"
    run_dir = Path(args.save_dir).resolve() / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(run_dir / "train.log")

    patch_size = (args.patch_h, args.patch_w, args.patch_d)

    logger.info("=" * 75)
    logger.info("ISLES-2022 TRAINING ENGINE")
    logger.info("=" * 75)
    logger.info("Experiment run dir: %s", run_dir)
    logger.info("Device: %s | AMP Enabled: %s", device, use_amp)
    logger.info("Model: %s | Fold: %d/%d", args.model_name, args.fold, args.n_splits)
    logger.info("Patch dimensions: %s | Batch size: %d", patch_size, args.batch_size)
    logger.info("Learning rate: %.1e | Dry-run mode: %s", args.lr, args.dry_run)

    # 1. Build DataLoaders
    logger.info("Building Fold %d DataLoaders...", args.fold)
    train_loader, val_loader = build_kfold_dataloaders(
        data_dir=args.data_dir,
        fold=args.fold,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        patch_size=patch_size,
        num_workers=args.num_workers,
        use_cache=False,
    )

    # 2. Build Model
    logger.info("Instantiating %s model (3 channels in, 1 channel out)...", args.model_name)
    model = build_model(
        model_name=args.model_name,
        in_channels=3,
        out_channels=1,
    ).to(device)

    # 3. Setup Loss Function (SOTA DiceJaccardLoss)
    loss_fn = get_loss_function(include_background=False, sigmoid=True, squared_pred=True)

    # 4. Setup Optimizer, Scheduler, and GradScaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_epochs = 1 if args.dry_run else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 5. Handle Dry-Run Mode
    if args.dry_run:
        logger.info("[DRY-RUN] Executing 2 training steps...")
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            max_steps=2,
        )
        logger.info("[DRY-RUN] Training steps completed. Train Loss: %.4f", train_loss)

        logger.info("[DRY-RUN] Executing sliding window validation on 1 volume...")
        val_dice, num_cases = validate(
            model=model,
            loader=val_loader,
            patch_size=patch_size,
            device=device,
            use_amp=use_amp,
            overlap=args.sw_overlap,
            max_cases=1,
        )
        logger.info("[DRY-RUN] Validation completed. Case Dice Score: %.4f", val_dice)

        checkpoint_path = run_dir / "dry_run_checkpoint.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
            },
            checkpoint_path,
        )
        logger.info("[DRY-RUN] Checkpoint saved successfully at %s", checkpoint_path)
        logger.info("=" * 75)
        logger.info("DRY-RUN PASSED 100%! The entire training pipeline is verified and ready.")
        logger.info("=" * 75)
        return

    # 6. Standard Training Loop
    best_dice = -1.0
    val_dice = 0.0
    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        logger.info("-" * 75)
        logger.info(
            "Starting Epoch %d/%d (LR: %.2e)...",
            epoch,
            total_epochs,
            optimizer.param_groups[0]["lr"],
        )
        epoch_start = time.time()
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        scheduler.step()
        epoch_duration = time.time() - epoch_start

        logger.info(
            "Epoch %d/%d (%.1fs) - Train Loss: %.4f | LR: %.2e",
            epoch,
            total_epochs,
            epoch_duration,
            train_loss,
            optimizer.param_groups[0]["lr"],
        )

        # Validation Step
        if epoch % args.val_interval == 0 or epoch == total_epochs:
            val_start = time.time()
            val_dice, num_cases = validate(
                model=model,
                loader=val_loader,
                patch_size=patch_size,
                device=device,
                use_amp=use_amp,
                overlap=args.sw_overlap,
            )
            val_duration = time.time() - val_start

            logger.info(
                "Epoch %d Validation (%.1fs): Mean Dice: %.4f (%d subjects)",
                epoch,
                val_duration,
                val_dice,
                num_cases,
            )

            # Checkpoint: Best model
            if val_dice > best_dice:
                best_dice = val_dice
                best_model_path = run_dir / "best_metric_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_dice": best_dice,
                    },
                    best_model_path,
                )
                logger.info(
                    "--> New best metric model saved (Dice: %.4f) at %s",
                    best_dice,
                    best_model_path,
                )

        # Checkpoint: Latest checkpoint
        latest_path = run_dir / "last_checkpoint.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "current_dice": val_dice,
            },
            latest_path,
        )

    total_time = time.time() - start_time
    logger.info("=" * 75)
    logger.info(
        "TRAINING FINISHED in %.1f minutes. Best Validation Dice: %.4f",
        total_time / 60.0,
        best_dice,
    )
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
