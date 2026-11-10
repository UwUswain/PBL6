"""Dice Similarity Coefficient (DSC) implementation for 3D binary masks."""

from typing import Union
import numpy as np


def compute_dice(
    pred: Union[np.ndarray, "torch.Tensor"],
    gt: Union[np.ndarray, "torch.Tensor"],
    empty_score: float = 1.0,
) -> float:
    """Computes the Dice Similarity Coefficient (DSC) between two binary masks.

    DSC = 2 * |P ∩ G| / (|P| + |G|)

    Args:
        pred: Predicted binary mask (boolean or int, 0/1).
        gt: Ground truth binary mask (boolean or int, 0/1).
        empty_score: Score returned when both pred and gt are empty (default: 1.0).

    Returns:
        Dice score as a float in [0.0, 1.0].

    Raises:
        ValueError: If pred and gt have mismatched shapes.
    """
    # Convert PyTorch tensor to numpy if applicable
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    if hasattr(gt, "detach"):
        gt = gt.detach().cpu().numpy()

    pred_arr = np.asarray(pred) > 0
    gt_arr = np.asarray(gt) > 0

    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"Shape mismatch in compute_dice: pred {pred_arr.shape} vs gt {gt_arr.shape}"
        )

    intersection = np.logical_and(pred_arr, gt_arr).sum()
    pred_sum = pred_arr.sum()
    gt_sum = gt_arr.sum()

    total = pred_sum + gt_sum
    if total == 0:
        return float(empty_score)

    return float(2.0 * intersection / total)
