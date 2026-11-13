"""Pre-registered Paired Fold Evaluation Engine (V3 Pos/Neg vs V4-Fixed Component-Aware).

Evaluates two Dual-Encoder SegResNet checkpoints (e.g., Fold 0 V3 vs V4-Fixed) under
identical sliding-window inference settings on the verified ISLES-2022 250-case split.

Distinguishes:
1. Overall ISLES-2022 metrics (Dice, HD95, AVD, LCD, Lesion F1, Precision, Recall, FP lesions).
2. Case-level burden stratification (case total GT volume < 5 mL, 5-30 mL, > 30 mL).
3. Component-level lesion detection stratified by component volume, separately analyzing
   small satellite components embedded inside multi-lesion scans (gt_lesion_count >= 2).
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import monai.data as md
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance
import numpy as np
import pandas as pd
import torch
import cc3d

from src.dataset.parser import (
    compute_split_fingerprint,
    get_kfold_splits,
    parse_isles_dataset,
    verify_isles_dataset,
)
from src.dataset.transforms import get_val_transforms
from src.metrics.dice import compute_dice
from src.metrics.eval_utils import compute_lesion_f1_score
from src.metrics.lesion import analyze_gt_components, compute_lesion_metrics
from src.metrics.volume import (
    compute_absolute_volume_difference,
    compute_lesion_volume,
)
from src.models.builder import get_model
from src.utils.manifest import extract_checkpoint_manifest, verify_paired_manifests
from src.utils.postprocess import remove_small_lesion_islands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_paired_fold")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Paired Fold Evaluation: V3 Pos/Neg vs V4-Fixed Component-Aware"
    )
    parser.add_argument("--ckpt-v3", type=str, required=True, help="Path to V3 baseline checkpoint.")
    parser.add_argument("--ckpt-v4", type=str, required=True, help="Path to V4-fixed checkpoint.")
    parser.add_argument("--data-dir", type=str, default="data/raw/ISLES-2022", help="ISLES-2022 root.")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold index (default: 0).")
    parser.add_argument("--n-splits", type=int, default=5, help="Total folds (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="K-Fold split seed (default: 42).")
    parser.add_argument("--overlap", type=float, default=0.25, help="Sliding-window overlap.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold.")
    parser.add_argument(
        "--postprocess-modes",
        nargs="+",
        default=["raw", "0.008ml"],
        choices=["raw", "0.008ml", "15vox"],
        help="Post-processing modes evaluated from the same forward pass ('raw' is always primary).",
    )
    parser.add_argument(
        "--sweep-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Optional exploratory probability thresholds (e.g. 0.45 0.50 0.55 0.60 0.65) evaluated from the same forward pass.",
    )
    parser.add_argument(
        "--sweep-postprocess",
        type=str,
        default="0.008ml",
        choices=["raw", "0.008ml", "15vox"],
        help="Post-processing mode used for exploratory threshold sweep (default: 0.008ml).",
    )
    parser.add_argument(
        "--sweep-target-fp",
        type=float,
        default=None,
        help="Optional reference FP/case target for exploratory operating-point comparison (defaults to V3 at --threshold).",
    )
    parser.add_argument(
        "--cache-probs-dir",
        type=str,
        default=None,
        help="Optional directory to cache predicted probability volumes (float16) to avoid re-inferring.",
    )
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Report output dir.")
    parser.add_argument("--max-cases", type=int, default=-1, help="Optional cap for smoke checks.")
    parser.add_argument("--strict-isles250", action="store_true", help="Require exact 250 ISLES cases.")
    parser.add_argument(
        "--allow-dry-run",
        action="store_true",
        help="Allow evaluating --dry-run smoke checkpoints (disabled by default to prevent accidental reporting).",
    )
    parser.add_argument(
        "--allow-code-mismatch",
        action="store_true",
        help="Allow differing code hashes/commits between V3 and V4 checkpoints.",
    )
    parser.add_argument(
        "--allow-env-mismatch",
        action="store_true",
        help="Allow differing Python/package dependency versions between V3 and V4 checkpoints.",
    )
    parser.add_argument(
        "--allow-param-mismatch",
        action="store_true",
        help="Allow comparing checkpoints with differing hyperparameters (e.g. patch size, epochs, or two V3 variants) while still strictly enforcing identical 250-case dataset, Fold 0 split hash, and test cases.",
    )
    parser.add_argument(
        "--sw-roi-size",
        nargs=3,
        type=int,
        default=None,
        metavar=("H", "W", "D"),
        help="Explicit sliding-window ROI size for inference (e.g. 96 96 64). If not specified, uses val_roi_size or roi_size from manifest.",
    )
    parser.add_argument(
        "--model-v3-name",
        type=str,
        default="V3_PosNeg",
        help="Display label for Model A (default: V3_PosNeg).",
    )
    parser.add_argument(
        "--model-v4-name",
        type=str,
        default="V4_Component",
        help="Display label for Model B (default: V4_Component).",
    )
    parser.add_argument(
        "--skip-manifest-check",
        action="store_true",
        help="Skip run_manifest parity verification (only for legacy checkpoints without manifests).",
    )
    return parser.parse_args()


def load_checkpoint_and_manifest(
    ckpt_path: Path,
    device: torch.device,
    require_manifest: bool = True,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Loads raw checkpoint dict and optional validated run_manifest."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload type at '{ckpt_path}': {type(checkpoint)}")
    manifest = (
        extract_checkpoint_manifest(checkpoint, ckpt_path)
        if require_manifest
        else (
            extract_checkpoint_manifest(checkpoint, ckpt_path)
            if "run_manifest" in checkpoint or (ckpt_path.parent / "run_manifest.json").is_file()
            else None
        )
    )
    return checkpoint, manifest


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
    manifest: Optional[Dict[str, Any]] = None,
) -> torch.nn.Module:
    """Instantiates architecture from manifest settings (or DualEncoderSegResNet defaults) and loads weights."""
    resolved = manifest.get("config", {}).get("resolved", {}) if manifest else {}
    model = get_model(
        model_name=str(resolved.get("model_name", "dual_segresnet")),
        in_channels=int(resolved.get("in_channels", 2)),
        out_channels=1,
        init_filters=int(resolved.get("init_filters", 32)),
        fusion_mode=str(resolved.get("fusion_mode", "bidirectional")),
        deep_supervision=bool(resolved.get("deep_supervision", True)),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def compute_case_hd95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    case_id: str = "",
    empty_mismatch_penalty: float = 100.0,
) -> Tuple[float, bool]:
    """Calculates 95th percentile Hausdorff Distance (mm) without swallowing runtime errors.

    Returns:
        Tuple of (hd95_mm, is_empty_mismatch_penalty).
        - If both masks are empty: returns (0.0, False).
        - If exactly one mask is empty: returns (empty_mismatch_penalty, True).
        - If both masks are non-empty: computes MONAI HD95 directly and raises on errors/non-finite output.
    """
    pred_bin = np.asarray(pred_mask) > 0
    gt_bin = np.asarray(gt_mask) > 0
    if pred_bin.shape != gt_bin.shape:
        raise ValueError(
            f"Shape mismatch in compute_case_hd95 for '{case_id}': pred={pred_bin.shape} vs gt={gt_bin.shape}"
        )

    pred_sum = int(np.sum(pred_bin))
    gt_sum = int(np.sum(gt_bin))
    if pred_sum == 0 and gt_sum == 0:
        return 0.0, False
    if pred_sum == 0 or gt_sum == 0:
        return float(empty_mismatch_penalty), True

    pred_t = torch.from_numpy(pred_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    hd95 = compute_hausdorff_distance(y_pred=pred_t, y=gt_t, percentile=95, spacing=voxel_spacing)
    val = float(hd95[0, 0].item())
    if not np.isfinite(val):
        raise RuntimeError(
            f"Non-finite HD95 computed for case '{case_id}' (pred_voxels={pred_sum}, gt_voxels={gt_sum}): {val}"
        )
    return val, False


def apply_postprocess_mode(pred_bin: np.ndarray, mode: str, spacing: Sequence[float]) -> np.ndarray:
    """Applies standardized post-processing cutoff without re-running model inference."""
    if mode == "raw":
        return pred_bin
    if mode == "0.008ml":
        return remove_small_lesion_islands(
            pred_bin, min_size_voxels=None, min_size_ml=0.008, voxel_spacing=spacing
        )
    if mode == "15vox":
        return remove_small_lesion_islands(
            pred_bin, min_size_voxels=15, voxel_spacing=spacing
        )
    raise ValueError(f"Unsupported postprocess mode: {mode}")


def order_postprocess_modes(modes: Sequence[str]) -> List[str]:
    """Deduplicates modes while guaranteeing 'raw' appears first as primary endpoint."""
    seen: List[str] = []
    for m in modes:
        if m not in seen:
            seen.append(m)
    if "raw" in seen:
        seen = ["raw"] + [m for m in seen if m != "raw"]
    return seen


def evaluate_prediction_volume(
    model_tag: str,
    pp_mode: str,
    case_id: str,
    pred_clean: np.ndarray,
    gt_bin: np.ndarray,
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Computes both case-level metrics (including FP counts) and component-level detection."""
    dsc = compute_dice(pred_clean, gt_bin)
    hd95, hd95_is_empty_penalty = compute_case_hd95(
        pred_clean,
        gt_bin,
        voxel_spacing=voxel_spacing,
        case_id=f"{model_tag}/{pp_mode}/{case_id}",
    )
    pred_vol = compute_lesion_volume(pred_clean, voxel_spacing)
    gt_vol = compute_lesion_volume(gt_bin, voxel_spacing)
    avd = compute_absolute_volume_difference(pred_clean, gt_bin, voxel_spacing)
    lesion_res = compute_lesion_metrics(pred_clean, gt_bin, iou_threshold=0.2, connectivity=26)

    # Official ISLES-2022 Scorer metrics:
    official_f1 = compute_lesion_f1_score(gt_bin, pred_clean, empty_value=1.0, connectivity=26)
    pred_labeled, num_pred_cc = cc3d.connected_components(pred_clean > 0, connectivity=26, return_N=True)
    if num_pred_cc > 0:
        overlapping_labels = set(np.unique(pred_labeled[gt_bin > 0]))
        overlapping_labels.discard(0)
        official_fp = int(num_pred_cc - len(overlapping_labels))
    else:
        official_fp = 0

    case_row: Dict[str, Any] = {
        "model": model_tag,
        "postprocess": pp_mode,
        "case_id": case_id,
        "gt_vol_ml": float(gt_vol),
        "pred_vol_ml": float(pred_vol),
        "gt_is_empty": bool(np.sum(gt_bin > 0) == 0),
        "pred_is_empty": bool(np.sum(pred_clean > 0) == 0),
        "hd95_is_empty_penalty": bool(hd95_is_empty_penalty),
        "gt_lesions": int(lesion_res["gt_lesion_count"]),
        "pred_lesions": int(lesion_res["pred_lesion_count"]),
        "tp_gt": int(lesion_res["tp_gt"]),
        "tp_pred": int(lesion_res["tp_pred"]),
        "fp_lesions": int(lesion_res["fp_lesions"]),
        "fn_lesions": int(lesion_res["fn_lesions"]),
        "dsc": float(dsc),
        "hd95_mm": float(hd95),
        "avd_ml": float(avd),
        "lcd": int(lesion_res["lcd"]),
        "lesion_f1": float(lesion_res["lesion_f1"]),
        "lesion_precision": float(lesion_res["lesion_precision"]),
        "lesion_recall": float(lesion_res["lesion_recall"]),
        "official_lesion_f1": float(official_f1),
        "official_fp_lesions": int(official_fp),
    }

    comp_rows = analyze_gt_components(pred_clean, gt_bin, voxel_spacing=voxel_spacing)
    for row in comp_rows:
        row["model"] = model_tag
        row["postprocess"] = pp_mode
        row["case_id"] = case_id

    return case_row, comp_rows


def _summarize_case_df(sub_df: pd.DataFrame) -> str:
    """Formats summary statistics string row for a subset DataFrame."""
    if sub_df.empty:
        return "0 | - | - | - | - | - | -"
    n_pen = int(sub_df["hd95_is_empty_penalty"].sum()) if "hd95_is_empty_penalty" in sub_df.columns else 0
    pen_note = f" [{n_pen} pen]" if n_pen > 0 else ""
    return (
        f"{len(sub_df)} | "
        f"{sub_df['dsc'].mean():.4f} ({sub_df['dsc'].median():.4f}) | "
        f"{sub_df['hd95_mm'].mean():.2f} ({sub_df['hd95_mm'].median():.2f}){pen_note} | "
        f"{sub_df['avd_ml'].mean():.2f} | "
        f"{sub_df['lcd'].mean():.2f} | "
        f"{sub_df['lesion_f1'].mean():.4f} | "
        f"{sub_df['fp_lesions'].mean():.2f} ({int(sub_df['fp_lesions'].sum())})"
    )


def _format_component_bins(comp_df: pd.DataFrame, multi_lesion_only: bool = False) -> str:
    """Formats component-level detection recall table across lesion sizes."""
    df_target = comp_df[comp_df["is_multi_lesion_case"]] if multi_lesion_only else comp_df
    bins = [
        ("Micro (< 0.1 mL)", df_target[df_target["comp_vol_ml"] < 0.1]),
        ("Very Small (0.1 - 1.0 mL)", df_target[(df_target["comp_vol_ml"] >= 0.1) & (df_target["comp_vol_ml"] < 1.0)]),
        ("Small (1.0 - 5.0 mL)", df_target[(df_target["comp_vol_ml"] >= 1.0) & (df_target["comp_vol_ml"] < 5.0)]),
        ("Medium/Large (>= 5.0 mL)", df_target[df_target["comp_vol_ml"] >= 5.0]),
    ]
    lines = [
        "| Nhóm Thể Tích Ổ (Component Vol) | Số Ổ GT | V3 Recall (IoU>=0.2) | V4-Fixed Recall (IoU>=0.2) | V3 Any-Hit | V4-Fixed Any-Hit |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label_name, sub in bins:
        v3_s = sub[sub["model"] == "V3_PosNeg"]
        v4_s = sub[sub["model"] == "V4_Component"]
        n_gt = max(len(v3_s), len(v4_s))
        v3_r = f"{v3_s['detected_iou20'].mean():.4f}" if len(v3_s) else "-"
        v4_r = f"{v4_s['detected_iou20'].mean():.4f}" if len(v4_s) else "-"
        v3_h = f"{v3_s['touched_any'].mean():.4f}" if len(v3_s) else "-"
        v4_h = f"{v4_s['touched_any'].mean():.4f}" if len(v4_s) else "-"
        lines.append(f"| {label_name} | {n_gt} | {v3_r} | {v4_r} | {v3_h} | {v4_h} |")
    return "\n".join(lines)


def normalize_sweep_thresholds(thresholds: Optional[Sequence[float]]) -> List[float]:
    """Validates, deduplicates, and sorts exploratory probability thresholds."""
    if not thresholds:
        return []
    cleaned = sorted({round(float(t), 6) for t in thresholds})
    for t in cleaned:
        if not (0.0 < t < 1.0):
            raise ValueError(f"Sweep threshold must be strictly in (0, 1), got {t}")
    return cleaned


def _attach_threshold(
    case_row: Dict[str, Any],
    comp_rows: Sequence[Dict[str, Any]],
    threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Creates shallow copies of case/component evaluation rows tagged with threshold."""
    c_copy = dict(case_row)
    c_copy["threshold"] = float(threshold)
    k_copies: List[Dict[str, Any]] = []
    for row in comp_rows:
        r_copy = dict(row)
        r_copy["threshold"] = float(threshold)
        k_copies.append(r_copy)
    return c_copy, k_copies


def summarize_threshold_sweep(
    df_sweep_cases: pd.DataFrame,
    df_sweep_comps: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregates case-level and micro-lesion (<0.1 mL) metrics across probability thresholds."""
    if df_sweep_cases.empty:
        return pd.DataFrame()

    model_order = ["V3_PosNeg", "V4_Component"]
    thresholds = sorted(df_sweep_cases["threshold"].unique().tolist())
    rows: List[Dict[str, Any]] = []

    for model_tag in model_order:
        for thr in thresholds:
            sub_c = df_sweep_cases[
                (df_sweep_cases["model"] == model_tag) & (df_sweep_cases["threshold"] == thr)
            ]
            if sub_c.empty:
                continue
            sub_k = (
                df_sweep_comps[
                    (df_sweep_comps["model"] == model_tag) & (df_sweep_comps["threshold"] == thr)
                ]
                if not df_sweep_comps.empty
                else pd.DataFrame()
            )
            micro_k = sub_k[sub_k["comp_vol_ml"] < 0.1] if not sub_k.empty else pd.DataFrame()
            micro_total = int(len(micro_k))
            micro_iou = int(micro_k["detected_iou20"].sum()) if micro_total > 0 else 0
            micro_any = int(micro_k["touched_any"].sum()) if micro_total > 0 else 0

            all_total = int(len(sub_k))
            all_iou = int(sub_k["detected_iou20"].sum()) if all_total > 0 else 0
            all_any = int(sub_k["touched_any"].sum()) if all_total > 0 else 0

            official_fp_col = "official_fp_lesions" if "official_fp_lesions" in sub_c.columns else "fp_lesions"
            official_f1_col = "official_lesion_f1" if "official_lesion_f1" in sub_c.columns else "lesion_f1"

            rows.append(
                {
                    "model": model_tag,
                    "threshold": float(thr),
                    "postprocess": str(sub_c["postprocess"].iloc[0]),
                    "n_cases": int(len(sub_c)),
                    # Nhóm metrics (IoU >= 0.2):
                    "fp_per_case": float(sub_c["fp_lesions"].mean()),
                    "total_fp": int(sub_c["fp_lesions"].sum()),
                    "micro_total": micro_total,
                    "micro_detected_iou20": micro_iou,
                    "micro_recall_iou20": float(micro_iou / micro_total) if micro_total > 0 else 0.0,
                    "lesion_f1_mean": float(sub_c["lesion_f1"].mean()),
                    "lesion_f1_median": float(sub_c["lesion_f1"].median()),
                    "lesion_precision_mean": float(sub_c["lesion_precision"].mean()),
                    "lesion_recall_mean": float(sub_c["lesion_recall"].mean()),
                    # Scorer ISLES-2022 metrics (Official overlap):
                    "official_fp_per_case": float(sub_c[official_fp_col].mean()),
                    "official_total_fp": int(sub_c[official_fp_col].sum()),
                    "micro_touched_any": micro_any,
                    "micro_any_hit_rate": float(micro_any / micro_total) if micro_total > 0 else 0.0,
                    "official_lesion_f1_mean": float(sub_c[official_f1_col].mean()),
                    "official_lesion_f1_median": float(sub_c[official_f1_col].median()),
                    # Common:
                    "all_comp_total": all_total,
                    "all_comp_detected_iou20": all_iou,
                    "all_comp_recall_iou20": float(all_iou / all_total) if all_total > 0 else 0.0,
                    "all_comp_touched_any": all_any,
                    "all_comp_any_hit_rate": float(all_any / all_total) if all_total > 0 else 0.0,
                    "dsc_mean": float(sub_c["dsc"].mean()),
                    "dsc_median": float(sub_c["dsc"].median()),
                    "hd95_mean": float(sub_c["hd95_mm"].mean()),
                    "hd95_median": float(sub_c["hd95_mm"].median()),
                    "avd_mean": float(sub_c["avd_ml"].mean()),
                    "lcd_mean": float(sub_c["lcd"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _format_sweep_row_label(model_tag: str) -> str:
    return "V3 Pos/Neg" if model_tag == "V3_PosNeg" else "V4-Fixed"


def _format_threshold_sweep_section(
    df_summary: pd.DataFrame,
    sweep_pp_mode: str,
    default_threshold: float,
    target_fp: Optional[float] = None,
) -> str:
    """Formats exploratory threshold-sweep tables, operating-point matching, and recall-FP curve."""
    blocks: List[str] = [
        "---",
        "",
        f"## Phân Tích Thăm Dò (Exploratory Analysis): Quét Ngưỡng Xác Suất Trên `{sweep_pp_mode}`",
        "",
        (
            "> **Ghi chú phương pháp (Exploratory Post-Hoc Sensitivity):** Bảng dưới đây là **phân tích thăm dò** "
            "chạy từ **cùng một lượt suy luận xác suất** cho mỗi checkpoint (áp dụng chung bộ lọc "
            f"`{sweep_pp_mode}` và 26-connectivity). Kết quả chính thức theo giao thức "
            "đã đăng ký vẫn là bảng **`raw` (`prob >= 0.50`)** ở phần trên."
        ),
        "",
        "### 1. Bảng Quét Ngưỡng Đối Chiếu Hai Cách Chấm",
        "",
        "#### A. Cách Chấm Của Nhóm (26-connectivity, Ghép Ổ Theo IoU ≥ 0.2)",
        "| Mô Hình | Ngưỡng | FP/Ca (Tổng FP) ↓ | Ổ < 0.1 mL (IoU>=0.2) ↑ | Tỷ Lệ Phát Hiện ↑ | Dice (Med) ↑ | Lesion F1 (Med) ↑ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, r in df_summary.iterrows():
        micro_n = int(r["micro_total"])
        blocks.append(
            f"| {_format_sweep_row_label(str(r['model']))} | {r['threshold']:.2f} | "
            f"{r['fp_per_case']:.2f} ({int(r['total_fp'])}) | "
            f"{int(r['micro_detected_iou20'])}/{micro_n} | "
            f"{r['micro_recall_iou20']:.4f} | "
            f"{r['dsc_mean']:.4f} ({r['dsc_median']:.4f}) | "
            f"{r['lesion_f1_mean']:.4f} ({r['lesion_f1_median']:.4f}) |"
        )

    blocks.extend(
        [
            "",
            "#### B. Scorer ISLES-2022 (Mã Chính Thức, Quy Tắc Chồng Lấn cc3d)",
            "| Mô Hình | Ngưỡng | FP/Ca (Tổng FP) ↓ | Ổ < 0.1 mL (Any-Hit) ↑ | Tỷ Lệ Phát Hiện ↑ | Dice (Med) ↑ | Lesion F1 (Med) ↑ |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for _, r in df_summary.iterrows():
        micro_n = int(r["micro_total"])
        off_fp = r.get("official_fp_per_case", r["fp_per_case"])
        off_tot_fp = int(r.get("official_total_fp", r["total_fp"]))
        off_f1_m = r.get("official_lesion_f1_mean", r["lesion_f1_mean"])
        off_f1_med = r.get("official_lesion_f1_median", r["lesion_f1_median"])
        blocks.append(
            f"| {_format_sweep_row_label(str(r['model']))} | {r['threshold']:.2f} | "
            f"{off_fp:.2f} ({off_tot_fp}) | "
            f"{int(r['micro_touched_any'])}/{micro_n} | "
            f"{r['micro_any_hit_rate']:.4f} | "
            f"{r['dsc_mean']:.4f} ({r['dsc_median']:.4f}) | "
            f"{off_f1_m:.4f} ({off_f1_med:.4f}) |"
        )

    v3_df = df_summary[df_summary["model"] == "V3_PosNeg"].sort_values("threshold").reset_index(drop=True)
    v4_df = df_summary[df_summary["model"] == "V4_Component"].sort_values("threshold").reset_index(drop=True)

    ref_row = None
    if not v3_df.empty:
        v3_default = v3_df[(v3_df["threshold"] - default_threshold).abs() <= 1e-6]
        ref_row = v3_default.iloc[0] if not v3_default.empty else v3_df.iloc[0]

    effective_target_fp = (
        float(target_fp)
        if target_fp is not None
        else (float(ref_row["fp_per_case"]) if ref_row is not None else None)
    )

    if effective_target_fp is not None and not v4_df.empty:
        blocks.extend(
            [
                "",
                f"### 2. So Sánh Gần Mốc Tham Chiếu `{effective_target_fp:.2f} FP/ca` & Điểm Kẹp",
                "",
                "#### A. Theo Cách Chấm Của Nhóm (IoU ≥ 0.2)",
                "",
                "| Điểm Vận Hành | Mô Hình | Ngưỡng | FP/Ca thực tế (Tổng FP) | Chênh FP vs Mốc | Ổ < 0.1 mL (IoU>=0.2) | Tỷ Lệ Nhóm | Dice (Med) | Lesion F1 (Med) |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        v4_sorted_fp = v4_df.assign(fp_diff=(v4_df["fp_per_case"] - effective_target_fp).abs())
        nearest_v4 = v4_sorted_fp.sort_values(["fp_diff", "threshold"]).iloc[0]

        if ref_row is not None:
            micro_n = int(ref_row["micro_total"])
            blocks.append(
                f"| Mốc chuẩn V3 | V3 Pos/Neg | {ref_row['threshold']:.2f} | "
                f"{ref_row['fp_per_case']:.2f} ({int(ref_row['total_fp'])}) | "
                f"{ref_row['fp_per_case'] - effective_target_fp:+.2f} | "
                f"{int(ref_row['micro_detected_iou20'])}/{micro_n} | "
                f"{ref_row['micro_recall_iou20']:.4f} | "
                f"{ref_row['dsc_mean']:.4f} ({ref_row['dsc_median']:.4f}) | "
                f"{ref_row['lesion_f1_mean']:.4f} ({ref_row['lesion_f1_median']:.4f}) |"
            )

        above = v4_df[v4_df["fp_per_case"] >= effective_target_fp].sort_values("fp_per_case")
        below = v4_df[v4_df["fp_per_case"] <= effective_target_fp].sort_values("fp_per_case", ascending=False)
        has_bracket = not above.empty and not below.empty
        hi_pt = above.iloc[0] if has_bracket else None
        lo_pt = below.iloc[0] if has_bracket else None

        nearest_label = "V4 gần mốc nhất"
        if has_bracket and float(hi_pt["threshold"]) != float(lo_pt["threshold"]):
            if float(nearest_v4["threshold"]) == float(hi_pt["threshold"]):
                nearest_label = f"V4 gần mốc nhất (Kẹp trên >= {effective_target_fp:.2f})"
            elif float(nearest_v4["threshold"]) == float(lo_pt["threshold"]):
                nearest_label = f"V4 gần mốc nhất (Kẹp dưới <= {effective_target_fp:.2f})"

        micro_n4 = int(nearest_v4["micro_total"])
        blocks.append(
            f"| {nearest_label} | V4-Fixed | {nearest_v4['threshold']:.2f} | "
            f"{nearest_v4['fp_per_case']:.2f} ({int(nearest_v4['total_fp'])}) | "
            f"{nearest_v4['fp_per_case'] - effective_target_fp:+.2f} | "
            f"{int(nearest_v4['micro_detected_iou20'])}/{micro_n4} | "
            f"{nearest_v4['micro_recall_iou20']:.4f} | "
            f"{nearest_v4['dsc_mean']:.4f} ({nearest_v4['dsc_median']:.4f}) | "
            f"{nearest_v4['lesion_f1_mean']:.4f} ({nearest_v4['lesion_f1_median']:.4f}) |"
        )

        if has_bracket:
            if float(hi_pt["threshold"]) != float(lo_pt["threshold"]):
                for label_b, pt in [
                    (f"Kẹp trên (>= {effective_target_fp:.2f})", hi_pt),
                    (f"Kẹp dưới (<= {effective_target_fp:.2f})", lo_pt),
                ]:
                    if float(pt["threshold"]) == float(nearest_v4["threshold"]):
                        continue
                    blocks.append(
                        f"| {label_b} | V4-Fixed | {pt['threshold']:.2f} | "
                        f"{pt['fp_per_case']:.2f} ({int(pt['total_fp'])}) | "
                        f"{pt['fp_per_case'] - effective_target_fp:+.2f} | "
                        f"{int(pt['micro_detected_iou20'])}/{int(pt['micro_total'])} | "
                        f"{pt['micro_recall_iou20']:.4f} | "
                        f"{pt['dsc_mean']:.4f} ({pt['dsc_median']:.4f}) | "
                        f"{pt['lesion_f1_mean']:.4f} ({pt['lesion_f1_median']:.4f}) |"
                    )
        else:
            direction = "cao hơn" if v4_df["fp_per_case"].min() > effective_target_fp else "thấp hơn"
            blocks.append(
                f"\n> *Lưu ý biên (Cách chấm nhóm):* Toàn bộ các ngưỡng V4 vừa quét đều có FP/ca **{direction}** mốc `{effective_target_fp:.2f}` "
                f"(dải quan sát của V4: `{v4_df['fp_per_case'].min():.2f}` – `{v4_df['fp_per_case'].max():.2f} FP/ca`)."
            )

        # Official Scorer Bracketing Section
        if "official_fp_per_case" in v4_df.columns and ref_row is not None:
            off_ref_fp = float(ref_row.get("official_fp_per_case", ref_row["fp_per_case"]))
            blocks.extend(
                [
                    "",
                    f"#### B. Theo Scorer ISLES-2022 (Mã Chính Thức, Mốc V3 = `{off_ref_fp:.2f} FP/ca`)",
                    "",
                    "| Điểm Vận Hành | Mô Hình | Ngưỡng | FP/Ca thực tế (Tổng FP) | Chênh FP vs Mốc | Ổ < 0.1 mL (Any-Hit) | Tỷ Lệ Scorer | Dice (Med) | Lesion F1 (Med) |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            v4_sorted_off_fp = v4_df.assign(off_fp_diff=(v4_df["official_fp_per_case"] - off_ref_fp).abs())
            nearest_off_v4 = v4_sorted_off_fp.sort_values(["off_fp_diff", "threshold"]).iloc[0]

            blocks.append(
                f"| Mốc chuẩn V3 (Scorer) | V3 Pos/Neg | {ref_row['threshold']:.2f} | "
                f"{off_ref_fp:.2f} ({int(ref_row['official_total_fp'])}) | "
                f"+0.00 | "
                f"{int(ref_row['micro_touched_any'])}/{int(ref_row['micro_total'])} | "
                f"{ref_row['micro_any_hit_rate']:.4f} | "
                f"{ref_row['dsc_mean']:.4f} ({ref_row['dsc_median']:.4f}) | "
                f"{ref_row['official_lesion_f1_mean']:.4f} ({ref_row['official_lesion_f1_median']:.4f}) |"
            )

            above_off = v4_df[v4_df["official_fp_per_case"] >= off_ref_fp].sort_values("official_fp_per_case")
            below_off = v4_df[v4_df["official_fp_per_case"] <= off_ref_fp].sort_values("official_fp_per_case", ascending=False)
            has_off_bracket = not above_off.empty and not below_off.empty
            hi_off = above_off.iloc[0] if has_off_bracket else None
            lo_off = below_off.iloc[0] if has_off_bracket else None

            nearest_off_label = "V4 gần mốc nhất (Scorer)"
            if has_off_bracket and float(hi_off["threshold"]) != float(lo_off["threshold"]):
                if float(nearest_off_v4["threshold"]) == float(hi_off["threshold"]):
                    nearest_off_label = f"V4 gần mốc nhất (Kẹp trên >= {off_ref_fp:.2f})"
                elif float(nearest_off_v4["threshold"]) == float(lo_off["threshold"]):
                    nearest_off_label = f"V4 gần mốc nhất (Kẹp dưới <= {off_ref_fp:.2f})"

            blocks.append(
                f"| {nearest_off_label} | V4-Fixed | {nearest_off_v4['threshold']:.2f} | "
                f"{nearest_off_v4['official_fp_per_case']:.2f} ({int(nearest_off_v4['official_total_fp'])}) | "
                f"{nearest_off_v4['official_fp_per_case'] - off_ref_fp:+.2f} | "
                f"{int(nearest_off_v4['micro_touched_any'])}/{int(nearest_off_v4['micro_total'])} | "
                f"{nearest_off_v4['micro_any_hit_rate']:.4f} | "
                f"{nearest_off_v4['dsc_mean']:.4f} ({nearest_off_v4['dsc_median']:.4f}) | "
                f"{nearest_off_v4['official_lesion_f1_mean']:.4f} ({nearest_off_v4['official_lesion_f1_median']:.4f}) |"
            )

            if has_off_bracket:
                if float(hi_off["threshold"]) != float(lo_off["threshold"]):
                    for label_b, pt in [
                        (f"Kẹp trên (>= {off_ref_fp:.2f})", hi_off),
                        (f"Kẹp dưới (<= {off_ref_fp:.2f})", lo_off),
                    ]:
                        if float(pt["threshold"]) == float(nearest_off_v4["threshold"]):
                            continue
                        blocks.append(
                            f"| {label_b} | V4-Fixed | {pt['threshold']:.2f} | "
                            f"{pt['official_fp_per_case']:.2f} ({int(pt['official_total_fp'])}) | "
                            f"{pt['official_fp_per_case'] - off_ref_fp:+.2f} | "
                            f"{int(pt['micro_touched_any'])}/{int(pt['micro_total'])} | "
                            f"{pt['micro_any_hit_rate']:.4f} | "
                            f"{pt['dsc_mean']:.4f} ({pt['dsc_median']:.4f}) | "
                            f"{pt['official_lesion_f1_mean']:.4f} ({pt['official_lesion_f1_median']:.4f}) |"
                        )
            else:
                direction_off = "cao hơn" if v4_df["official_fp_per_case"].min() > off_ref_fp else "thấp hơn"
                blocks.append(
                    f"\n> *Lưu ý biên (Scorer chính thức):* Toàn bộ các ngưỡng V4 vừa quét đều có FP/ca **{direction_off}** mốc `{off_ref_fp:.2f}` "
                    f"(dải quan sát của V4: `{v4_df['official_fp_per_case'].min():.2f}` – `{v4_df['official_fp_per_case'].max():.2f} FP/ca`)."
                )

    blocks.extend(
        [
            "",
            "### 3. Đường Cong Recall–FP Đầy Đủ Theo Ngưỡng",
            "",
            "| Ngưỡng | V3 FP/Ca | V3 Recall <0.1 mL (IoU>=0.2 / Any) | V3 Lesion Recall / Prec | V4 FP/Ca | V4 Recall <0.1 mL (IoU>=0.2 / Any) | V4 Lesion Recall / Prec |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    thresholds = sorted(df_summary["threshold"].unique().tolist())
    for thr in thresholds:
        r3 = v3_df[v3_df["threshold"] == thr]
        r4 = v4_df[v4_df["threshold"] == thr]
        s3 = (
            f"{r3.iloc[0]['fp_per_case']:.2f} | "
            f"{r3.iloc[0]['micro_recall_iou20']:.4f} / {r3.iloc[0]['micro_any_hit_rate']:.4f} | "
            f"{r3.iloc[0]['lesion_recall_mean']:.4f} / {r3.iloc[0]['lesion_precision_mean']:.4f}"
            if not r3.empty
            else "- | - | -"
        )
        s4 = (
            f"{r4.iloc[0]['fp_per_case']:.2f} | "
            f"{r4.iloc[0]['micro_recall_iou20']:.4f} / {r4.iloc[0]['micro_any_hit_rate']:.4f} | "
            f"{r4.iloc[0]['lesion_recall_mean']:.4f} / {r4.iloc[0]['lesion_precision_mean']:.4f}"
            if not r4.empty
            else "- | - | -"
        )
        blocks.append(f"| {thr:.2f} | {s3} | {s4} |")

    blocks.append("")
    return "\n".join(blocks)


def write_comparison_report(
    md_path: Path,
    df_cases: pd.DataFrame,
    df_comps: pd.DataFrame,
    args: argparse.Namespace,
    split_fp: Dict[str, Any],
    ordered_modes: Sequence[str],
    roi_size: Tuple[int, int, int],
    manifest_summary: Optional[Dict[str, Any]] = None,
    df_sweep_summary: Optional[pd.DataFrame] = None,
) -> None:
    """Writes structured Markdown comparison report distinguishing case vs component burden."""
    blocks: List[str] = [
        f"# Báo Cáo Đối Chứng Cặp Fold {args.fold}: V3 (Voxel Pos/Neg) vs V4-Fixed (Component-Aware)",
        "",
        "> **Vai trò thực nghiệm:** Thử nghiệm có đối chứng trên Fold 0 để kiểm chứng hiệu quả thực của Component-Aware Sampling sau khi sửa bug đóng băng cache (`Randomizable`) và đồng nhất negative sampling / checkpoint selection.",
        f"> **Lưu ý diễn giải:** Đây là kết quả trên 1 fold (Fold {args.fold}, `n={split_fp['val_count']}` ca validation, `val_sha256={split_fp['val_sha256']}`), dùng để kiểm chứng giả thuyết trước khi mở rộng sang các fold khác.",
        f"> **V3 Checkpoint:** `{args.ckpt_v3}`  ",
        f"> **V4-Fixed Checkpoint:** `{args.ckpt_v4}`  ",
        f"> **Thời điểm đánh giá:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "",
    ]

    blocks.append("## 0. Xác Minh Manifest & Quy Ước Metrics\n")
    if manifest_summary is not None:
        env3 = manifest_summary.get("environment_v3", {})
        pkgs3 = env3.get("packages", {}) if isinstance(env3, dict) else {}
        env_summary_str = (
            f"py={env3.get('python_major_minor', '?')}, torch={pkgs3.get('torch', '?')} "
            f"(cu={env3.get('torch_cuda', '?')}), monai={pkgs3.get('monai', '?')}, "
            f"numpy={pkgs3.get('numpy', '?')}, nibabel={pkgs3.get('nibabel', '?')}, "
            f"scipy={pkgs3.get('scipy', '?')}"
            if env3
            else "legacy_no_env_recorded"
        )
        blocks.extend(
            [
                "| Hạng Mục Xác Minh | Giá Trị |",
                "| :--- | :--- |",
                f"| Dataset SHA-256 (`dataset_ids_sha256`) | `{manifest_summary['dataset_ids_sha256']}` (`{manifest_summary['train_count']}` Train / `{manifest_summary['val_count']}` Val) |",
                f"| Train Split SHA-256 (`train_sha256`) | `{manifest_summary['train_sha256']}` (khớp 2 nhánh) |",
                f"| Val Split SHA-256 (`val_sha256`) | `{manifest_summary['val_sha256']}` (khớp 2 nhánh & tập đánh giá) |",
                f"| Sampling Mode | V3=`{manifest_summary['v3_sampling_mode']}` vs V4=`{manifest_summary['v4_sampling_mode']}` |",
                f"| Code Version (`git_commit`) | V3=`{str(manifest_summary['v3_git_commit'])[:12]}` vs V4=`{str(manifest_summary['v4_git_commit'])[:12]}` (`core_files_matched={manifest_summary['code_matched']}`) |",
                f"| Environment (`env_matched={manifest_summary.get('env_matched', True)}`) | `{env_summary_str}` |",
                f"| Dry-Run Checkpoints | `{manifest_summary['is_dry_run']}` |",
                "",
            ]
        )
        if manifest_summary.get("warnings"):
            for warn in manifest_summary["warnings"]:
                blocks.append(f"> **Cảnh báo manifest:** {warn}")
            blocks.append("")
    else:
        blocks.append("> **Lưu ý:** Đánh giá chạy với `--skip-manifest-check` (chưa xác minh manifest tự động).\n")

    blocks.extend(
        [
            "### Quy Ước Đánh Giá & Metrics (Pre-registered Protocol)",
            "",
            f"- **Phân cấp kết quả:** **`raw` (không lọc hậu xử lý)** là **kết quả chính** (đồng bộ trực tiếp với tiêu chí chọn checkpoint `postprocess_min_size: 0`); `0.008ml` (`< 8 voxels` ở `1x1x1 mm³`) là **phân tích độ nhạy hậu xử lý bổ sung** từ cùng một lượt dự đoán xác suất.",
            f"- **Giao thức suy luận:** Không gian đẳng hướng `1.0 x 1.0 x 1.0 mm³`, sliding-window `roi_size={tuple(roi_size)}`, `overlap={args.overlap}`, `mode=\"gaussian\"`, nhị phân hóa `prob >= {args.threshold}`.",
            "- **Dice (DSC):** $2|P \\cap G| / (|P| + |G|)$. Quy ước edge case: cả `Pred` và `GT` rỗng = `1.0`; đúng một bên rỗng = `0.0`.",
            "- **HD95 (mm):** 95th percentile Hausdorff Distance (`spacing=(1,1,1)`). Cả `Pred` và `GT` rỗng = `0.0 mm`; đúng một bên rỗng (`|P|=0` XOR `|G|=0`) = phạt cố định `100.0 mm` (hiển thị kèm `[k pen]` ở bảng tổng hợp nếu xảy ra). **Tuyệt đối không nuốt exception hoặc NaN/Inf thành `100.0 mm`** khi cả hai mask đều khác rỗng.",
            "- **AVD (mL):** $|V_{pred} - V_{gt}|$ tính bằng mL (`1 voxel = 0.001 mL`).",
            "- **Lesion Metrics & FP (`connectivity=26`, `iou_threshold=0.2`):** Đánh số thành phần liên thông 3D 26-connectivity. `TP_gt` đếm số ổ GT có `max_iou >= 0.2`; `TP_pred` đếm số ổ dự đoán có `max_iou >= 0.2`; `FP lesions = N_pred - TP_pred`; `FN lesions = N_gt - TP_gt`; `LCD = |N_pred - N_gt|`. Nếu cả hai rỗng: `LCD=0, F1=1.0, FP=0`; nếu chỉ `Pred` rỗng (`N_gt>0`): `LCD=N_gt, F1=0.0, FP=0`; nếu chỉ `GT` rỗng (`N_pred>0`): `LCD=N_pred, F1=0.0, FP=N_pred`.",
            "- **Component-Level Recall (Bảng 3A & 3B):** Báo cáo cả `Recall (IoU>=0.2)` và `Any-Hit (overlap > 0 voxels)` trên toàn bộ ổ GT và riêng các ổ nằm trong ca đa ổ (`gt_lesion_count >= 2`), tách biệt khỏi phân tầng gánh nặng ca bệnh (`Case Total < 5 mL`, `5 - 30 mL`, `> 30 mL`).",
            "",
        ]
    )

    for pp_mode in ordered_modes:
        sub_c = df_cases[df_cases["postprocess"] == pp_mode]
        sub_k = df_comps[df_comps["postprocess"] == pp_mode] if not df_comps.empty else pd.DataFrame()

        v3_c = sub_c[sub_c["model"] == "V3_PosNeg"]
        v4_c = sub_c[sub_c["model"] == "V4_Component"]

        if pp_mode == "raw":
            mode_heading = "## Kết Quả Chính: `raw` (Không Lọc Hậu Xử Lý — Tiêu Chí Chính)"
        elif pp_mode == "0.008ml":
            mode_heading = "## Phân Tích Bổ Sung: `0.008ml` (Hậu Xử Lý < 0.008 mL ≈ 8 Voxels)"
        else:
            mode_heading = f"## Phân Tích Bổ Sung: `{pp_mode}`"
        blocks.append(f"---\n\n{mode_heading}\n")
        blocks.append("### 1. Toàn Tập Validation Fold 0 (Mean [Median] & False Positives)\n")
        blocks.append("| Mô Hình | N | Dice (DSC) ↑ | HD95 (mm) ↓ | AVD (mL) ↓ | LCD ↓ | Lesion F1 ↑ | FP/Ca (Tổng FP) ↓ |")
        blocks.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        blocks.append(f"| V3 Pos/Neg | {_summarize_case_df(v3_c)} |")
        blocks.append(f"| V4-Fixed Component | {_summarize_case_df(v4_c)} |")
        if not v3_c.empty and not v4_c.empty:
            merged = pd.merge(v3_c, v4_c, on="case_id", suffixes=("_v3", "_v4"))
            blocks.append(
                f"| **Delta (V4 - V3)** | {len(merged)} | "
                f"**{(merged['dsc_v4'] - merged['dsc_v3']).mean():+.4f}** | "
                f"**{(merged['hd95_mm_v4'] - merged['hd95_mm_v3']).mean():+.2f}** | "
                f"**{(merged['avd_ml_v4'] - merged['avd_ml_v3']).mean():+.2f}** | "
                f"**{(merged['lcd_v4'] - merged['lcd_v3']).mean():+.2f}** | "
                f"**{(merged['lesion_f1_v4'] - merged['lesion_f1_v3']).mean():+.4f}** | "
                f"**{(merged['fp_lesions_v4'] - merged['fp_lesions_v3']).mean():+.2f} "
                f"({int(merged['fp_lesions_v4'].sum() - merged['fp_lesions_v3'].sum()):+d})** |"
            )

        blocks.append("\n### 2. Phân Tầng Theo Tổng Thể Tích Bệnh Nhân (Case-Level Burden)\n")
        blocks.append("| Nhóm Bệnh Nhân | Mô Hình | N | Dice (Med) | HD95 (Med) | AVD | LCD | Lesion F1 | FP/Ca (Tổng FP) |")
        blocks.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        case_bins = [
            ("Case Total < 5 mL", lambda d: d["gt_vol_ml"] < 5.0),
            ("Case Total 5 - 30 mL", lambda d: (d["gt_vol_ml"] >= 5.0) & (d["gt_vol_ml"] <= 30.0)),
            ("Case Total > 30 mL", lambda d: d["gt_vol_ml"] > 30.0),
        ]
        for b_name, b_fn in case_bins:
            for m_label, m_df in [("V3 Pos/Neg", v3_c), ("V4-Fixed", v4_c)]:
                blocks.append(f"| {b_name} | {m_label} | {_summarize_case_df(m_df[b_fn(m_df)])} |")

        if not sub_k.empty:
            blocks.append("\n### 3A. Phát Hiện Theo Từng Ổ Tổn Thương — Toàn Bộ Các Ổ GT (Component-Level)\n")
            blocks.append(_format_component_bins(sub_k, multi_lesion_only=False))
            blocks.append("\n### 3B. Phát Hiện Ổ Nhỏ Nằm Trong Ca Đa Ổ (`gt_lesion_count >= 2`)\n")
            blocks.append(_format_component_bins(sub_k, multi_lesion_only=True))
        blocks.append("")

    if df_sweep_summary is not None and not df_sweep_summary.empty:
        blocks.append(
            _format_threshold_sweep_section(
                df_summary=df_sweep_summary,
                sweep_pp_mode=args.sweep_postprocess,
                default_threshold=args.threshold,
                target_fp=args.sweep_target_fp,
            )
        )

    md_path.write_text("\n".join(blocks), encoding="utf-8")
    logger.info("Saved paired Fold %d comparison report to: %s", args.fold, md_path)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_modes = order_postprocess_modes(args.postprocess_modes)
    sweep_thresholds = normalize_sweep_thresholds(args.sweep_thresholds)

    cases = parse_isles_dataset(args.data_dir, require_mask=True)
    if args.strict_isles250:
        verify_isles_dataset(
            cases=cases,
            expected_num_cases=250,
            require_exact_isles250_ids=True,
            required_modalities=("dwi", "adc"),
            require_mask=True,
        )

    splits = get_kfold_splits(cases, n_splits=args.n_splits, seed=args.seed)
    train_cases, val_cases = splits[args.fold]
    split_fp = compute_split_fingerprint(train_cases, val_cases)
    logger.info(
        "Evaluating Fold %d Val Set: %d subjects (val_sha256=%s)",
        args.fold,
        split_fp["val_count"],
        split_fp["val_sha256"],
    )

    require_manifest = not args.skip_manifest_check
    ckpt_v3, manifest_v3 = load_checkpoint_and_manifest(
        Path(args.ckpt_v3), device=device, require_manifest=require_manifest
    )
    ckpt_v4, manifest_v4 = load_checkpoint_and_manifest(
        Path(args.ckpt_v4), device=device, require_manifest=require_manifest
    )

    manifest_summary: Optional[Dict[str, Any]] = None
    if require_manifest:
        assert manifest_v3 is not None and manifest_v4 is not None
        manifest_summary = verify_paired_manifests(
            manifest_v3=manifest_v3,
            manifest_v4=manifest_v4,
            expected_val_cases=val_cases,
            expected_fold=args.fold,
            expected_n_splits=args.n_splits,
            expected_split_seed=args.seed,
            strict_isles250=args.strict_isles250,
            allow_code_mismatch=args.allow_code_mismatch,
            allow_dry_run=args.allow_dry_run,
            allow_env_mismatch=args.allow_env_mismatch,
            allow_param_mismatch=args.allow_param_mismatch,
        )
        logger.info(
            "Manifest Parity PASSED | dataset=%s | train=%s | val=%s | code_matched=%s | env_matched=%s | modes=(%s vs %s)",
            manifest_summary["dataset_ids_sha256"],
            manifest_summary["train_sha256"],
            manifest_summary["val_sha256"],
            manifest_summary["code_matched"],
            manifest_summary["env_matched"],
            manifest_summary["v3_sampling_mode"],
            manifest_summary["v4_sampling_mode"],
        )
        for warn in manifest_summary.get("warnings", []):
            logger.warning("%s", warn)

    resolved = manifest_v3.get("config", {}).get("resolved", {}) if manifest_v3 else {}
    modalities = list(resolved.get("modalities", ["dwi", "adc"]))
    use_edwi = bool(resolved.get("use_edwi", False))
    if args.sw_roi_size is not None:
        roi_size: Tuple[int, int, int] = tuple(int(v) for v in args.sw_roi_size)
    else:
        roi_tuple = tuple(int(v) for v in resolved.get("val_roi_size", resolved.get("roi_size", [96, 96, 64])))
        roi_size: Tuple[int, int, int] = (roi_tuple[0], roi_tuple[1], roi_tuple[2])
    logger.info("Sliding-window inference ROI size: %s | overlap: %.2f", roi_size, args.overlap)

    val_transforms = get_val_transforms(
        target_spacing=(1.0, 1.0, 1.0),
        use_edwi=use_edwi,
        modalities=modalities,
    )
    val_ds = md.Dataset(data=val_cases, transform=val_transforms)
    val_loader = md.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    models = {
        "V3_PosNeg": build_model_from_checkpoint(ckpt_v3, device=device, manifest=manifest_v3),
        "V4_Component": build_model_from_checkpoint(ckpt_v4, device=device, manifest=manifest_v4),
    }

    case_records: List[Dict[str, Any]] = []
    comp_records: List[Dict[str, Any]] = []
    sweep_case_records: List[Dict[str, Any]] = []
    sweep_comp_records: List[Dict[str, Any]] = []

    cache_probs_dir = Path(args.cache_probs_dir) if args.cache_probs_dir else None
    if cache_probs_dir:
        cache_probs_dir.mkdir(parents=True, exist_ok=True)

    for idx, batch_data in enumerate(val_loader):
        if args.max_cases > 0 and idx >= args.max_cases:
            break
        case_id = val_cases[idx]["case_id"]
        images = batch_data["image"].to(device, non_blocking=True)
        gt_bin = (batch_data["mask"][0, 0] > 0).cpu().numpy().astype(np.uint8)

        for model_tag, model in models.items():
            t0 = time.time()
            cache_file = (cache_probs_dir / f"{model_tag}_{case_id}.npy") if cache_probs_dir else None
            from_cache = False
            if cache_file and cache_file.is_file():
                probs_np = np.load(cache_file).astype(np.float32)
                from_cache = True
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        logits = sliding_window_inference(
                            inputs=images,
                            roi_size=roi_size,
                            sw_batch_size=1,
                            predictor=model,
                            overlap=args.overlap,
                            mode="gaussian",
                        )
                        probs_np = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()
                if cache_file:
                    np.save(cache_file, probs_np.astype(np.float16))

            cached_default_eval: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]] = {}
            pred_raw = (probs_np >= args.threshold).astype(np.uint8)
            for pp_mode in ordered_modes:
                pred_clean = apply_postprocess_mode(pred_raw, pp_mode, spacing=(1.0, 1.0, 1.0))
                c_row, k_rows = evaluate_prediction_volume(
                    model_tag=model_tag,
                    pp_mode=pp_mode,
                    case_id=case_id,
                    pred_clean=pred_clean,
                    gt_bin=gt_bin,
                    voxel_spacing=(1.0, 1.0, 1.0),
                )
                case_records.append(c_row)
                comp_records.extend(k_rows)
                cached_default_eval[pp_mode] = (c_row, k_rows)

            for thr in sweep_thresholds:
                if abs(thr - args.threshold) <= 1e-6 and args.sweep_postprocess in cached_default_eval:
                    c_base, k_base = cached_default_eval[args.sweep_postprocess]
                    c_sw, k_sw = _attach_threshold(c_base, k_base, threshold=thr)
                else:
                    pred_thr = (probs_np >= thr).astype(np.uint8)
                    pred_sw = apply_postprocess_mode(
                        pred_thr, args.sweep_postprocess, spacing=(1.0, 1.0, 1.0)
                    )
                    c_base, k_base = evaluate_prediction_volume(
                        model_tag=model_tag,
                        pp_mode=args.sweep_postprocess,
                        case_id=case_id,
                        pred_clean=pred_sw,
                        gt_bin=gt_bin,
                        voxel_spacing=(1.0, 1.0, 1.0),
                    )
                    c_sw, k_sw = _attach_threshold(c_base, k_base, threshold=thr)
                sweep_case_records.append(c_sw)
                sweep_comp_records.extend(k_sw)

            logger.info(
                "  [Fold %d - %02d/%d | %s] %s %s in %.1fs",
                args.fold,
                idx + 1,
                len(val_loader),
                case_id,
                model_tag,
                "loaded from cache" if from_cache else "finished",
                time.time() - t0,
            )

    df_cases = pd.DataFrame(case_records)
    df_comps = pd.DataFrame(comp_records)
    df_sweep_summary: Optional[pd.DataFrame] = None

    case_csv = output_dir / f"paired_fold{args.fold}_case_metrics.csv"
    comp_csv = output_dir / f"paired_fold{args.fold}_component_metrics.csv"
    md_path = output_dir / f"paired_fold{args.fold}_comparison_report.md"

    df_cases.to_csv(case_csv, index=False)
    df_comps.to_csv(comp_csv, index=False)

    if sweep_thresholds:
        df_sweep_cases = pd.DataFrame(sweep_case_records)
        df_sweep_comps = pd.DataFrame(sweep_comp_records)
        df_sweep_summary = summarize_threshold_sweep(df_sweep_cases, df_sweep_comps)
        df_sweep_cases.to_csv(
            output_dir / f"paired_fold{args.fold}_threshold_sweep_cases.csv", index=False
        )
        df_sweep_comps.to_csv(
            output_dir / f"paired_fold{args.fold}_threshold_sweep_components.csv", index=False
        )
        df_sweep_summary.to_csv(
            output_dir / f"paired_fold{args.fold}_threshold_sweep_summary.csv", index=False
        )

    write_comparison_report(
        md_path=md_path,
        df_cases=df_cases,
        df_comps=df_comps,
        args=args,
        split_fp=split_fp,
        ordered_modes=ordered_modes,
        roi_size=roi_size,
        manifest_summary=manifest_summary,
        df_sweep_summary=df_sweep_summary,
    )


if __name__ == "__main__":
    main()

