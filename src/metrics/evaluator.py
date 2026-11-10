"""Unified evaluation engine for the complete ISLES-2022 metrics suite."""

from dataclasses import asdict, dataclass
import logging
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from src.metrics.dice import compute_dice
from src.metrics.lesion import compute_lesion_metrics
from src.metrics.volume import (
    compute_absolute_volume_difference,
    compute_lesion_volume,
)

logger = logging.getLogger(__name__)


@dataclass
class ISLESMetricsResult:
    """Stores the standard 4-metrics evaluation results for a single 3D case."""

    dsc: float
    avd_ml: float
    lcd: int
    lesion_f1: float
    lesion_precision: float
    lesion_recall: float
    pred_volume_ml: float
    gt_volume_ml: float
    pred_lesion_count: int
    gt_lesion_count: int

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Converts results dataclass to a dictionary."""
        return asdict(self)


def evaluate_isles_case(
    pred: Union[np.ndarray, "torch.Tensor"],
    gt: Union[np.ndarray, "torch.Tensor"],
    voxel_spacing: Sequence[float],
    iou_threshold: float = 0.2,
    connectivity: int = 26,
) -> ISLESMetricsResult:
    """Evaluates a single 3D MRI volume against all 4 ISLES-2022 standard metrics.

    Metrics computed:
    1. Dice Similarity Coefficient (DSC)
    2. Absolute Volume Difference (AVD) in mL
    3. Absolute Lesion Count Difference (LCD)
    4. Lesion-wise Detection F1 Score (with Precision and Recall)

    Args:
        pred: Predicted 3D binary mask.
        gt: Ground truth 3D binary mask.
        voxel_spacing: Voxel size in mm along each axis (sx, sy, sz).
        iou_threshold: Minimum IoU threshold for lesion detection (default: 0.2).
        connectivity: 3D neighborhood connectivity (default: 26).

    Returns:
        ISLESMetricsResult object containing all metrics.
    """
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    if hasattr(gt, "detach"):
        gt = gt.detach().cpu().numpy()

    # 1. Dice Score
    dsc = compute_dice(pred, gt)

    # 2. Volumes and AVD
    pred_vol = compute_lesion_volume(pred, voxel_spacing)
    gt_vol = compute_lesion_volume(gt, voxel_spacing)
    avd = compute_absolute_volume_difference(pred, gt, voxel_spacing)

    # 3. Lesion Count & Detection F1
    lesion_res = compute_lesion_metrics(
        pred,
        gt,
        iou_threshold=iou_threshold,
        connectivity=connectivity,
    )

    return ISLESMetricsResult(
        dsc=dsc,
        avd_ml=avd,
        lcd=int(lesion_res["lcd"]),
        lesion_f1=float(lesion_res["lesion_f1"]),
        lesion_precision=float(lesion_res["lesion_precision"]),
        lesion_recall=float(lesion_res["lesion_recall"]),
        pred_volume_ml=pred_vol,
        gt_volume_ml=gt_vol,
        pred_lesion_count=int(lesion_res["pred_lesion_count"]),
        gt_lesion_count=int(lesion_res["gt_lesion_count"]),
    )


class ISLESBatchEvaluator:
    """Accumulates and summarizes ISLES-2022 metrics across multiple cases."""

    def __init__(self, iou_threshold: float = 0.2, connectivity: int = 26):
        self.iou_threshold = iou_threshold
        self.connectivity = connectivity
        self.case_results: List[Dict[str, Union[str, float, int]]] = []

    def add_case(
        self,
        case_id: str,
        pred: Union[np.ndarray, "torch.Tensor"],
        gt: Union[np.ndarray, "torch.Tensor"],
        voxel_spacing: Sequence[float],
    ) -> ISLESMetricsResult:
        """Evaluates and records a single case.

        Args:
            case_id: Identifier of the case (e.g., 'sub-strokecase0001').
            pred: Predicted binary mask.
            gt: Ground truth binary mask.
            voxel_spacing: Voxel dimensions in mm.

        Returns:
            ISLESMetricsResult for this case.
        """
        res = evaluate_isles_case(
            pred=pred,
            gt=gt,
            voxel_spacing=voxel_spacing,
            iou_threshold=self.iou_threshold,
            connectivity=self.connectivity,
        )
        record = {"case_id": case_id, **res.to_dict()}
        self.case_results.append(record)
        return res

    def to_dataframe(self) -> pd.DataFrame:
        """Returns all individual case results as a pandas DataFrame."""
        return pd.DataFrame(self.case_results)

    def compute_summary(self) -> Dict[str, Dict[str, float]]:
        """Computes aggregate summary statistics (mean, median, std) for all metrics."""
        if not self.case_results:
            return {}

        df = self.to_dataframe()
        numeric_cols = [
            "dsc",
            "avd_ml",
            "lcd",
            "lesion_f1",
            "lesion_precision",
            "lesion_recall",
            "pred_volume_ml",
            "gt_volume_ml",
        ]

        summary = {}
        for col in numeric_cols:
            if col in df.columns:
                summary[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std(ddof=1)) if len(df) > 1 else 0.0,
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }

        return summary
