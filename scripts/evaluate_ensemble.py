"""ISLES-2022 Multi-Model & Cross-Validation Ensemble Evaluation Engine.

Provides unified inference and benchmarking for:
1. Single Fold evaluation.
2. 5-Fold Out-Of-Fold (OOF) full-cohort benchmark.
3. K-Fold Soft Probability Voting Ensemble.
4. Cross-Architecture Hybrid Blending (e.g. SwinUNETR + SegResNet).

Computes the official ISLES-2022 4-metrics suite:
- Dice Similarity Coefficient (DSC)
- Absolute Volume Difference (AVD in mL)
- Lesion Count Difference (LCD)
- Lesion-wise Detection F1 Score
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monai.inferers import sliding_window_inference
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.dataset.dataloader import build_kfold_dataloaders
from src.metrics.evaluator import ISLESBatchEvaluator, evaluate_isles_case
from src.models.builder import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_engine")


DEFAULT_MODEL_CONFIGS = {
    "swin_unetr": "configs/swin_unetr_30ep.yaml",
    "segresnet": "configs/train_30ep.yaml",
    "mednext": "configs/mednext_30ep.yaml",
}


def find_checkpoints_for_model(model_name: str, root_dir: Path) -> List[Path]:
    """Discovers best_metric_model.pth checkpoints for a given model architecture."""
    model_name_norm = model_name.lower().strip()
    checkpoints = []

    search_dirs = [
        root_dir / "reserve" / model_name_norm,
        root_dir / "reserve" / f"{model_name_norm}_fold",
        root_dir / "experiments" / "runs",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for fold_idx in range(5):
            fold_pattern = f"*{model_name_norm}*fold{fold_idx}*"
            matches = list(search_dir.glob(fold_pattern))
            for match in matches:
                ckpt_path = match / "best_metric_model.pth"
                if ckpt_path.is_file() and ckpt_path not in checkpoints:
                    checkpoints.append(ckpt_path)
                    break

    # Sort checkpoints by fold number
    sorted_checkpoints = []
    for f in range(5):
        for cp in checkpoints:
            if f"fold{f}" in str(cp).lower():
                sorted_checkpoints.append(cp)
                break

    return sorted_checkpoints


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Instantiates model architecture and loads weights from checkpoint."""
    logger.info("Loading %s weights from: %s", model_name, checkpoint_path.name)
    model = get_model(
        model_name=model_name,
        in_channels=3,
        out_channels=1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_sliding_window_prob(
    model: nn.Module,
    images: torch.Tensor,
    roi_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    """Computes soft probability map using MONAI sliding window inference."""
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )
            probs = torch.sigmoid(logits)
    return probs


def evaluate_cohort(
    predictions_and_targets: List[Tuple[str, np.ndarray, np.ndarray, Sequence[float]]],
    output_dir: Path,
    run_name: str,
) -> pd.DataFrame:
    """Computes ISLES-2022 metrics for each case and produces summary reports."""
    batch_evaluator = ISLESBatchEvaluator()
    case_rows = []

    for case_id, pred_mask, gt_mask, voxel_spacing in predictions_and_targets:
        batch_evaluator.add_case(
            case_id=case_id,
            pred=pred_mask,
            gt=gt_mask,
            voxel_spacing=voxel_spacing,
        )

    df_cases = batch_evaluator.to_dataframe()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed CSV
    csv_path = output_dir / f"{run_name}_cases.csv"
    df_cases.to_csv(csv_path, index=False)
    logger.info("Saved case-level metrics to: %s", csv_path)

    # Generate Markdown summary
    summary_path = output_dir / f"{run_name}_summary.md"
    summary_md = generate_markdown_summary(df_cases, run_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    logger.info("Saved summary report to: %s", summary_path)

    return df_cases


def generate_markdown_summary(df: pd.DataFrame, title: str) -> str:
    """Builds a formatted GitHub Markdown report from evaluation results."""
    dsc_mean, dsc_std = df["dsc"].mean(), df["dsc"].std()
    avd_mean, avd_std = df["avd_ml"].mean(), df["avd_ml"].std()
    lcd_mean, lcd_std = df["lcd"].mean(), df["lcd"].std()
    f1_mean, f1_std = df["lesion_f1"].mean(), df["lesion_f1"].std()

    md = f"""# ISLES-2022 EVALUATION REPORT: {title}
**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Evaluated Cases:** {len(df)}

---

## 1. OFFICIAL ISLES-2022 4-METRICS SUMMARY

| Metric | Mean ± Std | Median | Min | Max |
| :--- | :---: | :---: | :---: | :---: |
| **Dice Similarity Coefficient (DSC)** | **{dsc_mean:.4f} ± {dsc_std:.4f}** | {df['dsc'].median():.4f} | {df['dsc'].min():.4f} | {df['dsc'].max():.4f} |
| **Absolute Volume Difference (AVD)** | **{avd_mean:.2f} ± {avd_std:.2f} mL** | {df['avd_ml'].median():.2f} mL | {df['avd_ml'].min():.2f} mL | {df['avd_ml'].max():.2f} mL |
| **Lesion Count Difference (LCD)** | **{lcd_mean:.2f} ± {lcd_std:.2f}** | {df['lcd'].median():.1f} | {df['lcd'].min():.0f} | {df['lcd'].max():.0f} |
| **Lesion Detection F1 Score** | **{f1_mean:.4f} ± {f1_std:.4f}** | {df['lesion_f1'].median():.4f} | {df['lesion_f1'].min():.4f} | {df['lesion_f1'].max():.4f} |

---

## 2. SECONDARY LESION DETECTION METRICS

- **Lesion Precision:** {df['lesion_precision'].mean():.4f} ± {df['lesion_precision'].std():.4f}
- **Lesion Recall:** {df['lesion_recall'].mean():.4f} ± {df['lesion_recall'].std():.4f}
- **Mean Ground Truth Volume:** {df['gt_volume_ml'].mean():.2f} mL
- **Mean Predicted Volume:** {df['pred_volume_ml'].mean():.2f} mL
"""
    return md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISLES-2022 Cross-Validation & Multi-Model Ensemble Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="swin_unetr",
        choices=["swin_unetr", "segresnet", "mednext"],
        help="Primary model architecture to evaluate.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="oof",
        choices=["oof", "ensemble", "single"],
        help="Evaluation mode: 'oof' (Out-Of-Fold on full 200 cases), 'ensemble' (Soft voting on val-fold), or 'single'.",
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        default=0,
        help="Validation fold index for 'ensemble' or 'single' mode (0 to 4).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary probability decision threshold (default: 0.5).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Sliding window overlap ratio (default: 0.25).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/ISLES-2022",
        help="Path to ISLES-2022 BIDS dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory to save evaluation reports and CSVs.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit number of evaluated cases (0 = all cases, useful for dry-runs).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP FP16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    roi_size = (96, 96, 32)
    output_dir = Path(args.output_dir)

    logger.info("===========================================================================")
    logger.info("ISLES-2022 ENSEMBLE EVALUATION ENGINE")
    logger.info("===========================================================================")
    logger.info("Model: %s | Mode: %s | Device: %s (AMP: %s)", args.model, args.mode, device, use_amp)

    # 1. Discover checkpoints
    checkpoints = find_checkpoints_for_model(args.model, PROJECT_ROOT)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found for model '{args.model}' in reserve/ or experiments/runs/."
        )
    logger.info("Found %d checkpoints for %s:", len(checkpoints), args.model)
    for idx, cp in enumerate(checkpoints):
        logger.info("  Fold %d Checkpoint: %s", idx, cp)

    results_buffer = []

    # -------------------------------------------------------------------------
    # MODE 1: OUT-OF-FOLD (OOF) FULL-COHORT BENCHMARK
    # -------------------------------------------------------------------------
    if args.mode == "oof":
        logger.info("\n--- Starting 5-Fold Out-Of-Fold (OOF) Evaluation ---")
        total_cases_evaluated = 0

        for fold_idx in range(len(checkpoints)):
            ckpt_path = checkpoints[fold_idx]
            logger.info("\n>>> Evaluating Fold %d (Unseen Validation Set)...", fold_idx)

            # Load model
            model = load_model_from_checkpoint(args.model, ckpt_path, device)

            # Load fold-specific validation loader
            _, val_loader = build_kfold_dataloaders(
                data_dir=args.data_dir,
                fold=fold_idx,
                n_splits=5,
                patch_size=roi_size,
                num_workers=2,
                use_cache=False,
            )

            for case_idx, batch_data in enumerate(val_loader):
                case_id = f"fold{fold_idx}_case{case_idx:02d}"
                images = batch_data["image"].to(device, non_blocking=True)
                labels = batch_data["mask"]

                prob = predict_sliding_window_prob(
                    model=model,
                    images=images,
                    roi_size=roi_size,
                    overlap=args.overlap,
                    device=device,
                    use_amp=use_amp,
                )

                pred_mask = (prob[0, 0] >= args.threshold).cpu().numpy().astype(np.uint8)
                gt_mask = (labels[0, 0] > 0).cpu().numpy().astype(np.uint8)
                voxel_spacing = (1.0, 1.0, 1.0)

                results_buffer.append((case_id, pred_mask, gt_mask, voxel_spacing))
                total_cases_evaluated += 1

                if args.max_cases > 0 and total_cases_evaluated >= args.max_cases:
                    break

            # Free VRAM
            del model
            torch.cuda.empty_cache()

            if args.max_cases > 0 and total_cases_evaluated >= args.max_cases:
                break

        run_name = f"{args.model}_5fold_oof"

    # -------------------------------------------------------------------------
    # MODE 2: K-FOLD SOFT ENSEMBLE ON A SPECIFIC VALIDATION FOLD
    # -------------------------------------------------------------------------
    elif args.mode == "ensemble":
        logger.info("\n--- Starting K-Fold Soft Voting Ensemble on Fold %d ---", args.val_fold)

        # Pre-load all available models into memory/device
        models = [
            load_model_from_checkpoint(args.model, cp, device)
            for cp in checkpoints
        ]

        _, val_loader = build_kfold_dataloaders(
            data_dir=args.data_dir,
            fold=args.val_fold,
            n_splits=5,
            patch_size=roi_size,
            num_workers=2,
            use_cache=False,
        )

        for case_idx, batch_data in enumerate(val_loader):
            case_id = f"ens_val_fold{args.val_fold}_case{case_idx:02d}"
            images = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["mask"]

            # Accumulate probabilities across all K models
            prob_sum = torch.zeros(images.shape[2:], dtype=torch.float32, device=device)
            for m in models:
                p = predict_sliding_window_prob(
                    model=m,
                    images=images,
                    roi_size=roi_size,
                    overlap=args.overlap,
                    device=device,
                    use_amp=use_amp,
                )
                prob_sum += p[0, 0]

            ensemble_prob = prob_sum / len(models)
            pred_mask = (ensemble_prob >= args.threshold).cpu().numpy().astype(np.uint8)
            gt_mask = (labels[0, 0] > 0).cpu().numpy().astype(np.uint8)
            voxel_spacing = (1.0, 1.0, 1.0)

            results_buffer.append((case_id, pred_mask, gt_mask, voxel_spacing))

            if args.max_cases > 0 and (case_idx + 1) >= args.max_cases:
                break

        run_name = f"{args.model}_kfold_ensemble_valfold{args.val_fold}"

    # -------------------------------------------------------------------------
    # MODE 3: SINGLE MODEL EVALUATION
    # -------------------------------------------------------------------------
    else:
        logger.info("\n--- Starting Single Model Evaluation (Fold %d) ---", args.val_fold)
        ckpt_path = checkpoints[args.val_fold]
        model = load_model_from_checkpoint(args.model, ckpt_path, device)

        _, val_loader = build_kfold_dataloaders(
            data_dir=args.data_dir,
            fold=args.val_fold,
            n_splits=5,
            patch_size=roi_size,
            num_workers=2,
            use_cache=False,
        )

        for case_idx, batch_data in enumerate(val_loader):
            case_id = f"single_fold{args.val_fold}_case{case_idx:02d}"
            images = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["mask"]

            prob = predict_sliding_window_prob(
                model=model,
                images=images,
                roi_size=roi_size,
                overlap=args.overlap,
                device=device,
                use_amp=use_amp,
            )

            pred_mask = (prob[0, 0] >= args.threshold).cpu().numpy().astype(np.uint8)
            gt_mask = (labels[0, 0] > 0).cpu().numpy().astype(np.uint8)
            voxel_spacing = (1.0, 1.0, 1.0)

            results_buffer.append((case_id, pred_mask, gt_mask, voxel_spacing))

            if args.max_cases > 0 and (case_idx + 1) >= args.max_cases:
                break

        run_name = f"{args.model}_single_fold{args.val_fold}"

    # Generate results
    logger.info("\nCalculating official ISLES-2022 metrics for %d cases...", len(results_buffer))
    df = evaluate_cohort(results_buffer, output_dir, run_name)

    logger.info("===========================================================================")
    logger.info("FINAL BENCHMARK RESULTS [%s]:", run_name)
    logger.info("Mean Dice (DSC): %.4f ± %.4f", df["dsc"].mean(), df["dsc"].std())
    logger.info("Mean AVD:        %.2f ± %.2f mL", df["avd_ml"].mean(), df["avd_ml"].std())
    logger.info("Mean LCD:        %.2f ± %.2f", df["lcd"].mean(), df["lcd"].std())
    logger.info("Mean Lesion F1:  %.4f ± %.4f", df["lesion_f1"].mean(), df["lesion_f1"].std())
    logger.info("===========================================================================")


if __name__ == "__main__":
    main()
