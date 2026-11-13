"""Main training script for ISLES-2022 stroke lesion segmentation.

Orchestrates MLOps YAML configuration loading, data loaders (CacheDataset / Dataset),
model architectures (SegResNet, MedNeXt, SwinUNETR), compound SOTA DiceJaccardLoss,
mixed-precision (AMP FP16) training, sliding-window validation, and checkpointing.
"""

import argparse
from datetime import datetime
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import yaml
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from src.dataset.dataloader import build_kfold_dataloaders
from src.losses.builder import get_loss_function
from src.metrics.dice import compute_dice
from src.models.builder import get_model
from src.utils.manifest import build_run_manifest, write_run_manifest
from src.utils.postprocess import remove_small_lesion_islands

logger = logging.getLogger("train_engine")


def _compute_model_checksum(model: nn.Module) -> Tuple[float, float]:
    """Computes deterministic parameter sum and L2 norm for initialization/reload verification."""
    with torch.no_grad():
        total_sum = sum(float(p.detach().double().sum().cpu()) for p in model.parameters())
        total_sq = sum(float((p.detach().double() ** 2).sum().cpu()) for p in model.parameters())
    return total_sum, float(total_sq ** 0.5)


def setup_logger(log_file: Path) -> None:
    """Configures multi-handler logging to file and standard stream."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    for target_name in ("train_engine", "src"):
        target_logger = logging.getLogger(target_name)
        target_logger.setLevel(logging.INFO)
        target_logger.handlers.clear()
        target_logger.addHandler(file_handler)
        target_logger.addHandler(stream_handler)
        target_logger.propagate = False


def parse_args() -> argparse.Namespace:
    """Parses training command-line arguments, including the MLOps YAML config path."""
    parser = argparse.ArgumentParser(
        description="ISLES-2022 3D Stroke Lesion Segmentation Engine (MLOps Config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/local_4gb.yaml",
        help="Path to YAML configuration file (e.g. configs/local_4gb.yaml, configs/pc_8gb.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/ISLES-2022",
        help="Path to ISLES-2022 BIDS dataset root.",
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
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="Epoch interval for sliding-window validation evaluation.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments/runs",
        help="Directory to save experiment checkpoints and logs.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP FP16).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a full pipeline smoke test (train steps, scheduler, val, save & reload verification).",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=2,
        help="Number of real training batches to execute in --dry-run mode (default: 2).",
    )
    parser.add_argument(
        "--dry-run-val-cases",
        type=int,
        default=2,
        help="Number of validation volumes to evaluate in --dry-run mode (default: 2).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override number of worker processes for DataLoader (default: use config value).",
    )
    parser.add_argument(
        "--sw-overlap",
        type=float,
        default=0.25,
        help="Sliding window overlap ratio for validation inference [0.0 - 0.99].",
    )
    parser.add_argument(
        "--postprocess-min-size",
        type=int,
        default=0,
        help="Minimum 3D connected-component voxel size for validation filtering (default: 0 = disabled).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu'). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for weight init, DataLoader shuffle, and transforms (overrides config).",
    )
    parser.add_argument(
        "--strict-isles250",
        action="store_true",
        help="Enforce strict verification of all 250 ISLES-2022 case IDs and files before training.",
    )
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
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
            if isinstance(outputs, (tuple, list)):
                main_out, aux1, aux2 = outputs
                main_loss = loss_fn(main_out, labels)
                target_aux1 = torch.nn.functional.interpolate(
                    labels, size=aux1.shape[2:], mode='nearest'
                )
                target_aux2 = torch.nn.functional.interpolate(
                    labels, size=aux2.shape[2:], mode='nearest'
                )
                aux1_loss = loss_fn(aux1, target_aux1)
                aux2_loss = loss_fn(aux2, target_aux2)
                # Multi-scale deep supervision formulation (Paper 32 Eq. 15)
                loss = main_loss + 0.5 * aux1_loss + 0.25 * aux2_loss
            else:
                loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
    postprocess_min_size: int = 0,
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
            if postprocess_min_size > 0:
                val_preds = remove_small_lesion_islands(
                    val_preds, min_size_voxels=postprocess_min_size
                )
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
    """Main execution function for model training with MLOps config support."""
    args = parse_args()

    # =========================================================================
    # BƯỚC 1: ĐỌC VÀ PARSE CẤU HÌNH TỪ FILE YAML
    # =========================================================================
    # Kiểm tra đường dẫn file cấu hình YAML truyền từ CLI
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình YAML tại: {config_path}")

    # Đọc nội dung YAML và parse thành dictionary cfg bằng yaml.safe_load()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})

    # 1. Trích xuất các tham số thực nghiệm (experiment) từ cfg:
    # - model_name: Kiến trúc mạng ('segresnet', 'mednext', 'swin_unetr')
    # - max_epochs: Số epoch huấn luyện tối đa
    # - learning_rate: Tốc độ học khởi tạo cho Optimizer AdamW
    model_name = str(exp_cfg["model_name"])
    max_epochs = int(exp_cfg["max_epochs"])
    learning_rate = float(exp_cfg["learning_rate"])

    raw_seed = args.seed if args.seed is not None else exp_cfg.get("seed", None)
    seed_val = int(raw_seed) if raw_seed is not None else None
    if seed_val is not None:
        set_determinism(seed=seed_val)

    strict_isles250 = bool(
        args.strict_isles250
        or exp_cfg.get("strict_isles250", False)
        or data_cfg.get("strict_isles250", False)
    )
    expected_num_cases = exp_cfg.get(
        "expected_num_cases",
        data_cfg.get("expected_num_cases", 250 if strict_isles250 else None),
    )

    # 2. Trích xuất và ép kiểu các tham số dữ liệu (data) từ cfg:
    # - roi_size: Bắt buộc ép kiểu thành tuple (H, W, D) để truyền vào MONAI transforms lúc train
    # - val_roi_size: Sliding window ROI lúc validation (nếu không khai báo, mặc định bằng roi_size)
    # - batch_size: Số lượng volume được đưa vào mỗi bước huấn luyện
    # - num_workers: Số tiến trình CPU đọc và tiền xử lý dữ liệu ngầm
    roi_size = tuple(data_cfg["roi_size"])
    val_roi_size = tuple(data_cfg.get("val_roi_size", roi_size))
    batch_size = int(data_cfg["batch_size"])
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(data_cfg["num_workers"])
    )

    # 3. Logic CacheDataset: Dựa vào cờ cfg['data']['use_cache'], quyết định xem sẽ
    # bọc dữ liệu bằng monai.data.CacheDataset (nếu True) hay monai.data.Dataset thông thường (nếu False).
    # - Khi use_cache = True: Bộ nạp sẽ giữ toàn bộ hoặc một phần (cache_rate) dữ liệu đã tiền xử lý
    #   trực tiếp trên bộ nhớ RAM. Giúp bỏ qua bước I/O từ ổ cứng ở mỗi epoch, tăng tốc độ 5-8 lần.
    # - Khi use_cache = False: Bộ nạp sẽ xử lý on-the-fly trên từng batch, giúp tiết kiệm RAM tối đa.
    use_cache = bool(data_cfg.get("use_cache", False)) and (not args.dry_run)
    cache_rate = float(data_cfg.get("cache_rate", 1.0 if use_cache else 0.5))
    use_edwi = bool(data_cfg.get("use_edwi", False))
    pos_ratio = float(data_cfg.get("pos_ratio", 1.0))
    neg_ratio = float(data_cfg.get("neg_ratio", 1.0))
    num_samples_per_volume = int(data_cfg.get("num_samples_per_volume", 2))

    # Parse input modalities and channel configuration
    modalities = data_cfg.get("modalities", None)
    cfg_channels = exp_cfg.get("in_channels", data_cfg.get("in_channels", None))
    if modalities is not None:
        in_channels = int(cfg_channels) if cfg_channels is not None else len(modalities)
    elif cfg_channels is not None:
        in_channels = int(cfg_channels)
        modalities = ["dwi", "adc"] if in_channels == 2 else ["dwi", "adc", "flair"]
    else:
        in_channels = 3
        modalities = ["dwi", "adc", "flair"]

    # =========================================================================
    # BƯỚC 2: THIẾT LẬP THIẾT BỊ VÀ THƯ MỤC THỰC NGHIỆM
    # =========================================================================
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model_name}_fold{args.fold}_{timestamp}"
    if args.dry_run:
        exp_name = f"dry_run_{exp_name}"
    run_dir = Path(args.save_dir).resolve() / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(run_dir / "train.log")

    modalities_str = " + ".join([m.upper() for m in modalities])
    if use_edwi:
        modalities_str += " + eDWI (CLAHE)"

    logger.info("=" * 75)
    logger.info("ISLES-2022 TRAINING ENGINE (MLOps Config Mode)")
    logger.info("=" * 75)
    logger.info("Config file: %s", config_path)
    logger.info("Experiment run dir: %s", run_dir)
    logger.info("Device: %s | AMP FP16 Enabled: %s", device, use_amp)
    logger.info("Model: %s | Fold: %d/%d", model_name, args.fold, args.n_splits)
    logger.info("Input modalities: [%s] (%d channels)", modalities_str, in_channels)
    logger.info("Patch dimensions (roi_size): %s | Val ROI size: %s | Batch size: %d", roi_size, val_roi_size, batch_size)
    logger.info("Num workers: %d | Use CacheDataset: %s (cache_rate: %.2f)", num_workers, use_cache, cache_rate)
    logger.info(
        "Max epochs: %d | Learning rate: %.1e | Seed: %s | Strict ISLES-250: %s | Dry-run: %s",
        max_epochs,
        learning_rate,
        str(seed_val),
        strict_isles250,
        args.dry_run,
    )

    # =========================================================================
    # BƯỚC 3: XÂY DỰNG DATALOADERS (BỌC CACHEDATASET HOẶC DATASET)
    # =========================================================================
    if use_cache:
        logger.info(
            "-> RAM Cache enabled: Wrapping data with monai.data.CacheDataset (cache_rate: %.2f)",
            cache_rate,
        )
    else:
        logger.info(
            "-> RAM Cache disabled: Wrapping data with standard monai.data.Dataset for memory efficiency."
        )

    sampling_mode = data_cfg.get("sampling_mode", exp_cfg.get("sampling_mode", "standard"))
    if sampling_mode == "component_aware":
        logger.info("-> Active Sampling: Component-Aware Cropping (balanced micro-lesion oversampling enabled).")
    else:
        logger.info("-> Active Sampling: Standard Voxel-Uniform Pos/Neg Cropping.")

    split_seed = int(exp_cfg.get("split_seed", 42))
    train_seed = int(seed_val if seed_val is not None else 42)
    split_manifest: Dict[str, Any] = {}
    train_loader, val_loader = build_kfold_dataloaders(
        data_dir=args.data_dir,
        fold=args.fold,
        n_splits=args.n_splits,
        batch_size=batch_size,
        patch_size=roi_size,
        num_workers=num_workers,
        use_cache=use_cache,
        cache_rate=cache_rate,
        seed=split_seed,
        use_edwi=use_edwi,
        modalities=modalities,
        in_channels=in_channels,
        sampling_mode=sampling_mode,
        pos_ratio=pos_ratio,
        neg_ratio=neg_ratio,
        num_samples_per_volume=num_samples_per_volume,
        strict_isles250=strict_isles250,
        expected_num_cases=expected_num_cases,
        train_seed=train_seed,
        split_info_out=split_manifest,
    )

    # =========================================================================
    # BƯỚC 4: KHỞI TẠO MÔ HÌNH (MODEL ARCHITECTURE)
    # =========================================================================
    init_filters = int(exp_cfg.get("init_filters", 32 if "dual_segresnet" in model_name else 16))
    fusion_mode = str(exp_cfg.get("fusion_mode", "bidirectional"))
    deep_supervision = bool(exp_cfg.get("deep_supervision", True))

    if seed_val is not None:
        set_determinism(seed=seed_val)

    logger.info(
        "Instantiating %s (channels: %d, init_filters: %d, fusion: %s, deep_supervision: %s)...",
        model_name,
        in_channels,
        init_filters,
        fusion_mode,
        deep_supervision,
    )
    model = get_model(
        model_name=model_name,
        in_channels=in_channels,
        out_channels=1,
        init_filters=init_filters,
        fusion_mode=fusion_mode,
        deep_supervision=deep_supervision,
    ).to(device)

    init_sum, init_l2 = _compute_model_checksum(model)
    logger.info("Initial model param checksum | sum=%.6f | l2_norm=%.6f", init_sum, init_l2)

    # =========================================================================
    # BƯỚC 5: HÀM MẤT MÁT (SOTA DICEJACCARDLOSS / DICEFOCALLOSS), OPTIMIZER VÀ SCHEDULER
    # =========================================================================
    loss_type = str(exp_cfg.get("loss", "dice_ce"))
    loss_gamma = float(exp_cfg.get("loss_gamma", 2.0))
    loss_alpha = exp_cfg.get("loss_alpha", None)
    if loss_alpha is not None:
        loss_alpha = float(loss_alpha)

    loss_fn = get_loss_function(
        loss_type=loss_type,
        include_background=False,
        sigmoid=True,
        squared_pred=False,
        gamma=loss_gamma,
        alpha=loss_alpha,
    )

    weight_decay = float(exp_cfg.get("weight_decay", args.weight_decay))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_epochs = 1 if args.dry_run else max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    val_interval = args.val_interval
    if val_interval == 1 and "val_interval" in exp_cfg:
        val_interval = int(exp_cfg["val_interval"])

    postprocess_min_size = args.postprocess_min_size
    if postprocess_min_size == 0:
        postprocess_min_size = int(exp_cfg.get("postprocess_min_size", 0))
    if postprocess_min_size > 0:
        logger.info(
            "3D Connected-Component Postprocessing enabled (min_size: %d voxels) | Val interval: %d",
            postprocess_min_size,
            val_interval,
        )
    else:
        logger.info(
            "Checkpoint selection uses raw validation Dice (postprocess_min_size: 0) | Val interval: %d",
            val_interval,
        )

    resolved_manifest_cfg = {
        "model_name": model_name,
        "in_channels": in_channels,
        "modalities": list(modalities),
        "use_edwi": use_edwi,
        "init_filters": init_filters,
        "fusion_mode": fusion_mode,
        "deep_supervision": deep_supervision,
        "sampling_mode": sampling_mode,
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
        "num_samples_per_volume": num_samples_per_volume,
        "roi_size": list(roi_size),
        "val_roi_size": list(val_roi_size),
        "batch_size": batch_size,
        "use_cache": use_cache,
        "cache_rate": cache_rate,
        "loss_type": loss_type,
        "loss_gamma": loss_gamma,
        "loss_alpha": loss_alpha,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "total_epochs": total_epochs,
        "val_interval": val_interval,
        "sw_overlap": float(args.sw_overlap),
        "postprocess_min_size": postprocess_min_size,
        "amp": use_amp,
        "seed": seed_val,
        "is_dry_run": bool(args.dry_run),
    }
    run_manifest = build_run_manifest(
        repo_root=PROJECT_ROOT,
        config_path=config_path,
        raw_cfg=cfg,
        resolved_cfg=resolved_manifest_cfg,
        split_manifest=split_manifest,
        data_dir=args.data_dir,
        strict_isles250=strict_isles250,
        expected_num_cases=expected_num_cases,
        init_sum=init_sum,
        init_l2=init_l2,
    )
    manifest_file = write_run_manifest(run_manifest, run_dir / "run_manifest.json")
    logger.info(
        "Saved run manifest: %s | git_commit=%s (dirty=%s) | config_sha256=%s",
        manifest_file,
        run_manifest["code_version"]["git_commit"][:12],
        run_manifest["code_version"]["git_dirty"],
        run_manifest["config"]["config_sha256"][:16],
    )

    # =========================================================================
    # BƯỚC 6: XỬ LÝ CHẾ ĐỘ THỬ NGHIỆM NHANH (DRY-RUN SMOKE TEST)
    # =========================================================================
    if args.dry_run:
        logger.info(
            "[DRY-RUN] Executing %d real training batches (forward + backward + optimizer/AMP)...",
            args.dry_run_steps,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
        t0_train = time.time()
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            max_steps=args.dry_run_steps,
        )
        t_train = time.time() - t0_train
        step_sec = t_train / max(args.dry_run_steps, 1)
        train_peak_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        if not math.isfinite(train_loss):
            raise RuntimeError(f"[DRY-RUN] Non-finite training loss observed: {train_loss}")
        scheduler.step()
        post_sum, post_l2 = _compute_model_checksum(model)
        if abs(post_sum - init_sum) < 1e-9 and abs(post_l2 - init_l2) < 1e-9:
            raise RuntimeError("[DRY-RUN] Model weights did not change after optimizer step.")
        logger.info(
            "[DRY-RUN] Train steps + optimizer + scheduler OK | Loss: %.4f | Step: %.2fs/batch | Train Peak VRAM: %.1f MB (%.2f GB)",
            train_loss,
            step_sec,
            train_peak_mb,
            train_peak_mb / 1024.0,
        )

        logger.info(
            "[DRY-RUN] Executing sliding-window validation on %d real volumes (val_roi_size: %s)...",
            args.dry_run_val_cases,
            val_roi_size,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0_val = time.time()
        val_dice, num_cases = validate(
            model=model,
            loader=val_loader,
            patch_size=val_roi_size,
            device=device,
            use_amp=use_amp,
            overlap=args.sw_overlap,
            max_cases=args.dry_run_val_cases,
            postprocess_min_size=postprocess_min_size,
        )
        t_val = time.time() - t0_val
        val_case_sec = t_val / max(num_cases, 1)
        val_peak_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        if num_cases < 1 or not math.isfinite(val_dice):
            raise RuntimeError(f"[DRY-RUN] Invalid validation result: cases={num_cases}, dice={val_dice}")
        logger.info(
            "[DRY-RUN] Validation completed (%d cases in %.1fs, %.2fs/case). Mean Dice: %.4f | Val Peak VRAM: %.1f MB (%.2f GB)",
            num_cases,
            t_val,
            val_case_sec,
            val_dice,
            val_peak_mb,
            val_peak_mb / 1024.0,
        )

        total_gpu_mb = (
            torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        overall_peak_mb = max(train_peak_mb, val_peak_mb)
        headroom_mb = total_gpu_mb - overall_peak_mb
        batches_per_ep = max(len(train_loader), 1)
        total_val_cases = len(val_loader.dataset) if hasattr(val_loader, "dataset") else 50
        num_val_runs = max(100 // max(val_interval, 1), 1)
        est_100ep_sec = (100 * batches_per_ep * step_sec) + (num_val_runs * total_val_cases * val_case_sec)
        est_100ep_hours = est_100ep_sec / 3600.0

        logger.info("=" * 75)
        logger.info("[BENCHMARK SUMMARY] Resource Profile & Scaling Estimate:")
        logger.info("  Train Patch Size (roi_size): %s | Val Sliding Window: %s", roi_size, val_roi_size)
        logger.info("  Train Speed: %.2fs / batch (est. train-only 100ep: %.2f h)", step_sec, (100 * batches_per_ep * step_sec) / 3600.0)
        logger.info("  Val Speed:   %.2fs / case  (est. val-only 100ep [%d runs]: %.2f h)", val_case_sec, num_val_runs, (num_val_runs * total_val_cases * val_case_sec) / 3600.0)
        logger.info("  Peak VRAM:   Train = %.1f MB (%.2f GB) | Val = %.1f MB (%.2f GB)", train_peak_mb, train_peak_mb / 1024.0, val_peak_mb, val_peak_mb / 1024.0)
        logger.info("  Headroom:    %.1f MB (%.2f GB free out of %.1f MB total on %s)", headroom_mb, headroom_mb / 1024.0, total_gpu_mb, device)
        logger.info("  Projected 100-Epoch Duration: %.2f hours", est_100ep_hours)
        logger.info("=" * 75)

        checkpoint_path = run_dir / "dry_run_checkpoint.pth"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_dice": val_dice,
                "best_dice": val_dice,
                "run_manifest": run_manifest,
            },
            checkpoint_path,
        )
        logger.info("[DRY-RUN] Checkpoint saved at %s. Verifying reload + inference parity...", checkpoint_path)

        reloaded_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if reloaded_ckpt.get("run_manifest", {}).get("split", {}).get("val_sha256") != split_manifest["val_sha256"]:
            raise RuntimeError("[DRY-RUN] Reloaded checkpoint manifest mismatch.")

        verify_model = get_model(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=1,
            init_filters=init_filters,
            fusion_mode=fusion_mode,
            deep_supervision=deep_supervision,
        ).to(device)
        verify_model.load_state_dict(reloaded_ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(reloaded_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(reloaded_ckpt["scheduler_state_dict"])
        scaler.load_state_dict(reloaded_ckpt["scaler_state_dict"])

        reload_sum, reload_l2 = _compute_model_checksum(verify_model)
        if abs(reload_sum - post_sum) > 1e-3 or abs(reload_l2 - post_l2) > 1e-4:
            raise RuntimeError(
                f"[DRY-RUN] Weight mismatch after reload: saved=({post_sum:.6f}, {post_l2:.6f}) "
                f"vs reloaded=({reload_sum:.6f}, {reload_l2:.6f})."
            )

        verify_model.eval()
        with torch.no_grad():
            smoke_batch = next(iter(train_loader))
            smoke_inputs = smoke_batch["image"][:1].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                smoke_out = verify_model(smoke_inputs)
                main_out = smoke_out[0] if isinstance(smoke_out, (tuple, list)) else smoke_out
                if not torch.isfinite(main_out).all():
                    raise RuntimeError("[DRY-RUN] Non-finite output tensor during post-reload forward pass.")
        del verify_model

        logger.info(
            "[DRY-RUN] Save/Load + Post-reload forward verification PASSED (sum=%.6f, out_shape=%s).",
            reload_sum,
            tuple(main_out.shape),
        )
        logger.info("=" * 75)
        logger.info("DRY-RUN PASSED 100%%! Pipeline configured via %s is verified and ready.", config_path.name)
        logger.info("=" * 75)
        return

    # =========================================================================
    # BƯỚC 7: VÒNG LẶP HUẤN LUYỆN CHUẨN (STANDARD TRAINING & VALIDATION LOOP)
    # =========================================================================
    if seed_val is not None:
        set_determinism(seed=seed_val)
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

        # Đánh giá Validation định kỳ
        if epoch % val_interval == 0 or epoch == total_epochs:
            val_start = time.time()
            val_dice, num_cases = validate(
                model=model,
                loader=val_loader,
                patch_size=val_roi_size,
                device=device,
                use_amp=use_amp,
                overlap=args.sw_overlap,
                postprocess_min_size=postprocess_min_size,
            )
            val_duration = time.time() - val_start

            logger.info(
                "Epoch %d Validation (%.1fs): Mean Dice: %.4f (%d subjects)",
                epoch,
                val_duration,
                val_dice,
                num_cases,
            )

            # Lưu checkpoint tốt nhất (Best Metric)
            if val_dice > best_dice:
                best_dice = val_dice
                best_model_path = run_dir / "best_metric_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_dice": best_dice,
                        "run_manifest": run_manifest,
                    },
                    best_model_path,
                )
                logger.info(
                    "--> New best metric model saved (Dice: %.4f) at %s",
                    best_dice,
                    best_model_path,
                )

        # Ensure logs are flushed to disk immediately on every epoch
        for handler in logger.handlers:
            handler.flush()
        sys.stdout.flush()

        # Lưu checkpoint gần nhất ở mỗi epoch (Last Checkpoint)
        latest_path = run_dir / "last_checkpoint.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "current_dice": val_dice,
                "run_manifest": run_manifest,
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
