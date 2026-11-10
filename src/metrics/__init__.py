"""ISLES-2022 Medical Segmentation Metrics Suite.

Standard 4-metrics evaluation:
- Dice Similarity Coefficient (DSC)
- Absolute Volume Difference (AVD) in mL
- Absolute Lesion Count Difference (LCD)
- Lesion-wise Detection F1 Score
"""

from src.metrics.dice import compute_dice
from src.metrics.evaluator import (
    ISLESBatchEvaluator,
    ISLESMetricsResult,
    evaluate_isles_case,
)
from src.metrics.lesion import compute_lesion_metrics
from src.metrics.volume import (
    compute_absolute_volume_difference,
    compute_lesion_volume,
    compute_relative_volume_difference,
)

__all__ = [
    "compute_dice",
    "compute_lesion_volume",
    "compute_absolute_volume_difference",
    "compute_relative_volume_difference",
    "compute_lesion_metrics",
    "evaluate_isles_case",
    "ISLESMetricsResult",
    "ISLESBatchEvaluator",
]
