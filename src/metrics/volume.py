"""Volume metrics implementation for 3D binary masks (ISLES-2022)."""

from typing import Sequence, Tuple, Union
import numpy as np


def compute_lesion_volume(
    mask: Union[np.ndarray, "torch.Tensor"],
    voxel_spacing: Sequence[float],
) -> float:
    """Computes total lesion volume in milliliters (mL / cm^3).

    V = N_voxels * (spacing_x * spacing_y * spacing_z) / 1000.0

    Args:
        mask: 3D binary mask array (boolean or 0/1).
        voxel_spacing: Tuple or list of voxel sizes in mm (sx, sy, sz).

    Returns:
        Lesion volume in mL (float).
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    mask_arr = np.asarray(mask) > 0
    voxel_vol_mm3 = float(np.prod(voxel_spacing))
    # 1 mm^3 = 0.001 mL
    volume_ml = float(mask_arr.sum() * voxel_vol_mm3 / 1000.0)
    return volume_ml


def compute_absolute_volume_difference(
    pred: Union[np.ndarray, "torch.Tensor"],
    gt: Union[np.ndarray, "torch.Tensor"],
    voxel_spacing: Sequence[float],
) -> float:
    """Computes Absolute Volume Difference (AVD) in milliliters (mL).

    AVD = |V_pred - V_gt|

    Args:
        pred: Predicted binary mask.
        gt: Ground truth binary mask.
        voxel_spacing: Voxel spacing in mm along each spatial dimension.

    Returns:
        Absolute difference in lesion volume in mL.

    Raises:
        ValueError: If pred and gt have different shapes or spacing length is invalid.
    """
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    if hasattr(gt, "detach"):
        gt = gt.detach().cpu().numpy()

    pred_arr = np.asarray(pred) > 0
    gt_arr = np.asarray(gt) > 0

    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"Shape mismatch in compute_absolute_volume_difference: pred {pred_arr.shape} vs gt {gt_arr.shape}"
        )

    if len(voxel_spacing) != pred_arr.ndim:
        raise ValueError(
            f"voxel_spacing length ({len(voxel_spacing)}) must match mask dimensions ({pred_arr.ndim})"
        )

    pred_vol = compute_lesion_volume(pred_arr, voxel_spacing)
    gt_vol = compute_lesion_volume(gt_arr, voxel_spacing)

    return float(abs(pred_vol - gt_vol))


def compute_relative_volume_difference(
    pred: Union[np.ndarray, "torch.Tensor"],
    gt: Union[np.ndarray, "torch.Tensor"],
) -> float:
    """Computes Relative Volume Difference (RVD).

    RVD = (V_pred - V_gt) / V_gt

    Args:
        pred: Predicted binary mask.
        gt: Ground truth binary mask.

    Returns:
        Relative difference (float). If both are empty returns 0.0; if gt is empty and pred is not, returns float('inf').
    """
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    if hasattr(gt, "detach"):
        gt = gt.detach().cpu().numpy()

    pred_voxels = float((np.asarray(pred) > 0).sum())
    gt_voxels = float((np.asarray(gt) > 0).sum())

    if gt_voxels == 0:
        return 0.0 if pred_voxels == 0 else float("inf")

    return float((pred_voxels - gt_voxels) / gt_voxels)
