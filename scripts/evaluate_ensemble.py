"""ISLES-2022 Multi-Model & Cross-Validation Ensemble Evaluation Engine.

Provides unified inference, benchmarking, and subgroup analysis for:
1. Single Fold evaluation (with optional direct checkpoint path).
2. 5-Fold Out-Of-Fold (OOF) full-cohort benchmark.
3. K-Fold Soft Probability Voting Ensemble.
4. Cross-Architecture Hybrid Blending (e.g., SwinUNETR + SegResNet / DualEncoderSegResNet).

Computes the official ISLES-2022 4-metrics suite:
- Dice Similarity Coefficient (DSC)
- Absolute Volume Difference (AVD in mL)
- Lesion Count Difference (LCD)
- Lesion-wise Detection F1 Score (Precision, Recall)

Performs clinical volume stratification:
- Small Lesions (< 5 mL)
- Medium Lesions (5 - 30 mL)
- Large Lesions (> 30 mL)
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
from src.metrics.evaluator import ISLESBatchEvaluator
from src.models.builder import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_engine")

DEFAULT_MODEL_CONFIGS: Dict[str, str] = {
    "dual_segresnet": "configs/dual_segresnet_2ch_30ep.yaml",
    "dual_encoder_segresnet": "configs/dual_segresnet_2ch_30ep.yaml",
    "segresnet": "configs/train_30ep.yaml",
    "swin_unetr": "configs/swin_unetr_30ep.yaml",
    "mednext": "configs/mednext_30ep.yaml",
}


def get_model_modality_info(model_name: str) -> Tuple[int, List[str]]:
    """Determines channel count and modality list based on architecture name."""
    norm_name = model_name.lower().strip()
    if norm_name in ("dual_segresnet", "dual_encoder_segresnet"):
        return 2, ["dwi", "adc"]
    return 3, ["dwi", "adc", "flair"]


def find_checkpoints_for_model(model_name: str, root_dir: Path) -> List[Path]:
    """Discovers best_metric_model.pth checkpoints for a given model architecture."""
    model_name_norm = model_name.lower().strip()
    checkpoints: List[Path] = []

    search_dirs = [
        root_dir / "reserve" / model_name_norm,
        root_dir / "reserve" / f"{model_name_norm}_fold",
        root_dir / "experiments" / "runs",
        root_dir / "experiments" / "Kaggle",
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

    sorted_checkpoints: List[Path] = []
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
    in_channels: int = 3,
) -> nn.Module:
    """Instantiates model architecture and loads weights from checkpoint."""
    logger.info("Loading %s (in_channels=%d) weights from: %s", model_name, in_channels, checkpoint_path.name)
    model = get_model(
        model_name=model_name,
        in_channels=in_channels,
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


def compute_subgroup_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Union[int, float]]]:
    """Stratifies cases into Small (<5mL), Medium (5-30mL), and Large (>30mL) subgroups."""
    subgroups = {
        "Small (< 5 mL)": df[df["gt_volume_ml"] < 5.0],
        "Medium (5 - 30 mL)": df[(df["gt_volume_ml"] >= 5.0) & (df["gt_volume_ml"] <= 30.0)],
        "Large (> 30 mL)": df[df["gt_volume_ml"] > 30.0],
    }

    subgroup_summary: Dict[str, Dict[str, Union[int, float]]] = {}
    for name, sub_df in subgroups.items():
        count = len(sub_df)
        if count == 0:
            subgroup_summary[name] = {
                "count": 0,
                "mean_dsc": 0.0,
                "mean_avd": 0.0,
                "mean_lcd": 0.0,
                "mean_f1": 0.0,
            }
        else:
            subgroup_summary[name] = {
                "count": count,
                "mean_dsc": float(sub_df["dsc"].mean()),
                "mean_avd": float(sub_df["avd_ml"].mean()),
                "mean_lcd": float(sub_df["lcd"].mean()),
                "mean_f1": float(sub_df["lesion_f1"].mean()),
            }
    return subgroup_summary


def evaluate_cohort(
    predictions_and_targets: List[Tuple[str, np.ndarray, np.ndarray, Sequence[float]]],
    output_dir: Path,
    run_name: str,
) -> pd.DataFrame:
    """Computes ISLES-2022 metrics for each case and produces comprehensive summary reports."""
    batch_evaluator = ISLESBatchEvaluator()

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

    # Subgroup stratification
    subgroups = compute_subgroup_metrics(df_cases)

    # Generate Markdown summary report
    md_path = output_dir / f"{run_name}_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mean_dsc = df_cases["dsc"].mean()
    std_dsc = df_cases["dsc"].std()
    med_dsc = df_cases["dsc"].median()

    mean_avd = df_cases["avd_ml"].mean()
    std_avd = df_cases["avd_ml"].std()
    med_avd = df_cases["avd_ml"].median()

    mean_lcd = df_cases["lcd"].mean()
    std_lcd = df_cases["lcd"].std()
    med_lcd = df_cases["lcd"].median()

    mean_f1 = df_cases["lesion_f1"].mean()
    std_f1 = df_cases["lesion_f1"].std()
    med_f1 = df_cases["lesion_f1"].median()

    subgroup_rows = ""
    for s_name, s_data in subgroups.items():
        subgroup_rows += (
            f"| {s_name} | {s_data['count']} | {s_data['mean_dsc']:.4f} | "
            f"{s_data['mean_avd']:.2f} mL | {s_data['mean_lcd']:.2f} | {s_data['mean_f1']:.4f} |\n"
        )

    report_content = f"""# ISLES-2022 Evaluation Report: {run_name}

**Evaluation Date:** {timestamp}  
**Total Cases Evaluated:** {len(df_cases)}  

---

## 1. Overall ISLES-2022 Benchmark Suite (Mean ± Std [Median])

| Metric | Mean ± Std | Median | Unit / Target |
| :--- | :--- | :--- | :--- |
| **Dice Similarity Coefficient (DSC)** | **{mean_dsc:.4f} ± {std_dsc:.4f}** | **{med_dsc:.4f}** | Higher is better (Max 1.0) |
| **Absolute Volume Difference (AVD)** | **{mean_avd:.2f} ± {std_avd:.2f}** | **{med_avd:.2f}** | mL (Lower is better) |
| **Lesion Count Difference (LCD)** | **{mean_lcd:.2f} ± {std_lcd:.2f}** | **{med_lcd:.2f}** | Count (Lower is better) |
| **Lesion Detection F1 Score** | **{mean_f1:.4f} ± {std_f1:.4f}** | **{med_f1:.4f}** | Higher is better (Max 1.0) |

---

## 2. Lesion Volume Subgroup Stratification

| Subgroup Category | Case Count | Mean DSC | Mean AVD | Mean LCD | Mean Lesion F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
{subgroup_rows}
---

## 3. Case-by-case Distribution
- Top 5 Best Cases (by DSC):
{df_cases.nlargest(5, 'dsc')[['case_id', 'dsc', 'avd_ml', 'lcd', 'lesion_f1']].to_markdown(index=False)}

- Top 5 Most Challenging Cases (by DSC):
{df_cases.nsmallest(5, 'dsc')[['case_id', 'dsc', 'avd_ml', 'lcd', 'lesion_f1']].to_markdown(index=False)}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Saved benchmark report to: %s", md_path)

    return df_cases


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="ISLES-2022 Multi-Model & Ensemble Evaluation CLI"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dual_segresnet",
        choices=["dual_segresnet", "dual_encoder_segresnet", "segresnet", "swin_unetr", "mednext"],
        help="Target model architecture to evaluate.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "oof", "ensemble", "hybrid"],
        help="Evaluation mode: single fold, 5-fold OOF, K-fold ensemble, or hybrid.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit path to a .pth checkpoint (overrides directory discovery for single mode).",
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        default=0,
        help="Validation fold index [0-4] for 'single' or 'ensemble' modes.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/ISLES-2022",
        help="Path to ISLES-2022 dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory to store evaluation reports and CSV files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for mask binarization.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding window inference overlap ratio.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=-1,
        help="Maximum cases to evaluate (-1 for all).",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI execution flow."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    output_dir = Path(args.output_dir)

    in_channels, modalities = get_model_modality_info(args.model)
    logger.info("===========================================================================")
    logger.info("ISLES-2022 EVALUATION ENGINE")
    logger.info("Model: %s | Channels: %d (%s) | Mode: %s | Device: %s", args.model, in_channels, "+".join(modalities), args.mode, device)
    logger.info("===========================================================================")

    # Determine patch size from config if available
    config_file = DEFAULT_MODEL_CONFIGS.get(args.model)
    roi_size = (96, 96, 32)
    if config_file and Path(config_file).is_file():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                c = yaml.safe_load(f)
            roi_size = tuple(c.get("training", {}).get("patch_size", [96, 96, 32]))
        except Exception as e:
            logger.warning("Could not read patch_size from config %s: %s", config_file, e)

    results_buffer: List[Tuple[str, np.ndarray, np.ndarray, Sequence[float]]] = []

    # -------------------------------------------------------------------------
    # MODE 1: SINGLE MODEL EVALUATION
    # -------------------------------------------------------------------------
    if args.mode == "single":
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint).resolve()
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Provided checkpoint does not exist: {ckpt_path}")
        else:
            checkpoints = find_checkpoints_for_model(args.model, PROJECT_ROOT)
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint found for model '{args.model}' in project.")
            ckpt_path = checkpoints[args.val_fold % len(checkpoints)]

        logger.info("Evaluating single checkpoint: %s on Fold %d", ckpt_path, args.val_fold)
        model = load_model_from_checkpoint(args.model, ckpt_path, device, in_channels=in_channels)

        _, val_loader = build_kfold_dataloaders(
            data_dir=args.data_dir,
            fold=args.val_fold,
            n_splits=5,
            patch_size=roi_size,
            num_workers=2,
            use_cache=False,
            in_channels=in_channels,
            modalities=modalities,
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
            logger.info("Evaluated Case %02d/%d: %s", case_idx + 1, len(val_loader), case_id)

            if args.max_cases > 0 and (case_idx + 1) >= args.max_cases:
                break

        run_name = f"{args.model}_single_fold{args.val_fold}"

    # -------------------------------------------------------------------------
    # MODE 2: K-FOLD SOFT ENSEMBLE ON A SPECIFIC VALIDATION FOLD
    # -------------------------------------------------------------------------
    elif args.mode == "ensemble":
        checkpoints = find_checkpoints_for_model(args.model, PROJECT_ROOT)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for ensemble of model '{args.model}'.")

        logger.info("Starting K-Fold Ensemble using %d models on Fold %d", len(checkpoints), args.val_fold)
        models = [
            load_model_from_checkpoint(args.model, cp, device, in_channels=in_channels)
            for cp in checkpoints
        ]

        _, val_loader = build_kfold_dataloaders(
            data_dir=args.data_dir,
            fold=args.val_fold,
            n_splits=5,
            patch_size=roi_size,
            num_workers=2,
            use_cache=False,
            in_channels=in_channels,
            modalities=modalities,
        )

        for case_idx, batch_data in enumerate(val_loader):
            case_id = f"ens_fold{args.val_fold}_case{case_idx:02d}"
            images = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["mask"]

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
            logger.info("Evaluated Case %02d/%d: %s", case_idx + 1, len(val_loader), case_id)

            if args.max_cases > 0 and (case_idx + 1) >= args.max_cases:
                break

        run_name = f"{args.model}_kfold_ensemble_valfold{args.val_fold}"

    # -------------------------------------------------------------------------
    # MODE 3: 5-FOLD OUT-OF-FOLD (OOF) FULL-COHORT EVALUATION
    # -------------------------------------------------------------------------
    elif args.mode == "oof":
        checkpoints = find_checkpoints_for_model(args.model, PROJECT_ROOT)
        if len(checkpoints) < 5:
            logger.warning("Found only %d checkpoints instead of 5 for full OOF evaluation.", len(checkpoints))

        total_cases_evaluated = 0
        for fold_idx, ckpt_path in enumerate(checkpoints):
            logger.info("--- Evaluating Fold %d / %d using %s ---", fold_idx, len(checkpoints) - 1, ckpt_path.name)
            model = load_model_from_checkpoint(args.model, ckpt_path, device, in_channels=in_channels)

            _, val_loader = build_kfold_dataloaders(
                data_dir=args.data_dir,
                fold=fold_idx,
                n_splits=5,
                patch_size=roi_size,
                num_workers=2,
                use_cache=False,
                in_channels=in_channels,
                modalities=modalities,
            )

            for case_idx, batch_data in enumerate(val_loader):
                case_id = f"oof_fold{fold_idx}_case{case_idx:02d}"
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

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.max_cases > 0 and total_cases_evaluated >= args.max_cases:
                break

        run_name = f"{args.model}_5fold_oof"

    # -------------------------------------------------------------------------
    # MODE 4: HYBRID MULTI-ARCHITECTURE BLENDING
    # -------------------------------------------------------------------------
    elif args.mode == "hybrid":
        seg_ckpts = find_checkpoints_for_model("segresnet", PROJECT_ROOT)
        swin_ckpts = find_checkpoints_for_model("swin_unetr", PROJECT_ROOT)

        if not seg_ckpts or not swin_ckpts:
            raise FileNotFoundError("Hybrid evaluation requires checkpoints for both SegResNet and SwinUNETR.")

        m_seg = load_model_from_checkpoint("segresnet", seg_ckpts[args.val_fold], device, in_channels=3)
        m_swin = load_model_from_checkpoint("swin_unetr", swin_ckpts[args.val_fold], device, in_channels=3)

        _, val_loader = build_kfold_dataloaders(
            data_dir=args.data_dir,
            fold=args.val_fold,
            n_splits=5,
            patch_size=roi_size,
            num_workers=2,
            use_cache=False,
            in_channels=3,
            modalities=["dwi", "adc", "flair"],
        )

        for case_idx, batch_data in enumerate(val_loader):
            case_id = f"hybrid_valfold{args.val_fold}_case{case_idx:02d}"
            images = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["mask"]

            p_seg = predict_sliding_window_prob(m_seg, images, roi_size, args.overlap, device, use_amp)
            p_swin = predict_sliding_window_prob(m_swin, images, roi_size, args.overlap, device, use_amp)

            blended_prob = 0.5 * p_seg[0, 0] + 0.5 * p_swin[0, 0]
            pred_mask = (blended_prob >= args.threshold).cpu().numpy().astype(np.uint8)
            gt_mask = (labels[0, 0] > 0).cpu().numpy().astype(np.uint8)
            voxel_spacing = (1.0, 1.0, 1.0)

            results_buffer.append((case_id, pred_mask, gt_mask, voxel_spacing))
            logger.info("Evaluated Case %02d/%d: %s", case_idx + 1, len(val_loader), case_id)

            if args.max_cases > 0 and (case_idx + 1) >= args.max_cases:
                break

        run_name = f"hybrid_segresnet_swin_valfold{args.val_fold}"

    # Generate results
    logger.info("Calculating official ISLES-2022 metrics for %d cases...", len(results_buffer))
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

