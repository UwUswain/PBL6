"""Lesion-wise detection metrics and lesion count analysis (ISLES-2022)."""

from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np
from scipy import ndimage


def compute_lesion_metrics(
    pred: Union[np.ndarray, "torch.Tensor"],
    gt: Union[np.ndarray, "torch.Tensor"],
    iou_threshold: float = 0.2,
    connectivity: int = 26,
) -> Dict[str, Union[float, int]]:
    """Computes lesion count difference (LCD) and lesion-wise detection F1 score.

    Uses 3D Connected Component Labeling to identify individual lesion instances.
    A ground-truth lesion is counted as detected (TP) if at least one predicted
    lesion overlaps with it with IoU >= iou_threshold.

    Args:
        pred: Predicted 3D binary mask.
        gt: Ground truth 3D binary mask.
        iou_threshold: Minimum IoU required to consider a lesion detected (default: 0.2).
        connectivity: Neighborhood connectivity: 6, 18, or 26 (default: 26).

    Returns:
        Dictionary containing:
            - 'lcd': Absolute Lesion Count Difference |N_pred - N_gt| (int).
            - 'lesion_f1': Lesion-wise detection F1 score (float, 0.0 to 1.0).
            - 'lesion_precision': Lesion-wise precision (float, 0.0 to 1.0).
            - 'lesion_recall': Lesion-wise recall / sensitivity (float, 0.0 to 1.0).
            - 'pred_lesion_count': Number of detected lesion components in pred (int).
            - 'gt_lesion_count': Number of lesion components in gt (int).
    """
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    if hasattr(gt, "detach"):
        gt = gt.detach().cpu().numpy()

    pred_arr = np.asarray(pred) > 0
    gt_arr = np.asarray(gt) > 0

    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"Shape mismatch in compute_lesion_metrics: pred {pred_arr.shape} vs gt {gt_arr.shape}"
        )

    if pred_arr.ndim != 3:
        raise ValueError(
            f"compute_lesion_metrics expects 3D volume, got {pred_arr.ndim}D"
        )

    # Define structuring element based on connectivity
    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        struct = ndimage.generate_binary_structure(3, 2)
    elif connectivity == 26:
        struct = ndimage.generate_binary_structure(3, 3)
    else:
        raise ValueError(f"Unsupported connectivity: {connectivity}. Choose from [6, 18, 26].")

    # Label connected components
    pred_labeled, num_pred = ndimage.label(pred_arr, structure=struct)
    gt_labeled, num_gt = ndimage.label(gt_arr, structure=struct)

    lcd = abs(num_pred - num_gt)

    # Edge cases: Both empty
    if num_pred == 0 and num_gt == 0:
        return {
            "lcd": 0,
            "lesion_f1": 1.0,
            "lesion_precision": 1.0,
            "lesion_recall": 1.0,
            "pred_lesion_count": 0,
            "gt_lesion_count": 0,
            "tp_pred": 0,
            "tp_gt": 0,
            "fp_lesions": 0,
            "fn_lesions": 0,
        }

    # One is empty, the other is not
    if num_pred == 0 and num_gt > 0:
        return {
            "lcd": num_gt,
            "lesion_f1": 0.0,
            "lesion_precision": 0.0,
            "lesion_recall": 0.0,
            "pred_lesion_count": 0,
            "gt_lesion_count": num_gt,
            "tp_pred": 0,
            "tp_gt": 0,
            "fp_lesions": 0,
            "fn_lesions": num_gt,
        }

    if num_pred > 0 and num_gt == 0:
        return {
            "lcd": num_pred,
            "lesion_f1": 0.0,
            "lesion_precision": 0.0,
            "lesion_recall": 0.0,
            "pred_lesion_count": num_pred,
            "gt_lesion_count": 0,
            "tp_pred": 0,
            "tp_gt": 0,
            "fp_lesions": num_pred,
            "fn_lesions": 0,
        }

    # Both have components: compute component sizes
    pred_sizes = np.bincount(pred_labeled.ravel())
    gt_sizes = np.bincount(gt_labeled.ravel())

    # Find overlapping voxels
    overlap = (pred_labeled > 0) & (gt_labeled > 0)
    p_indices = pred_labeled[overlap]
    g_indices = gt_labeled[overlap]

    # Map pairs of (p_idx, g_idx) to intersection voxel counts
    # Using 2D sparse/dense pairing
    pair_keys = p_indices.astype(np.int64) * (num_gt + 1) + g_indices.astype(np.int64)
    unique_pairs, pair_counts = np.unique(pair_keys, return_counts=True)

    # Track max IoU for each GT component and each Pred component
    max_iou_gt = np.zeros(num_gt + 1, dtype=np.float32)
    max_iou_pred = np.zeros(num_pred + 1, dtype=np.float32)

    for pair_key, count in zip(unique_pairs, pair_counts):
        p_idx = int(pair_key // (num_gt + 1))
        g_idx = int(pair_key % (num_gt + 1))

        union = pred_sizes[p_idx] + gt_sizes[g_idx] - count
        iou = float(count) / float(union) if union > 0 else 0.0

        if iou > max_iou_gt[g_idx]:
            max_iou_gt[g_idx] = iou
        if iou > max_iou_pred[p_idx]:
            max_iou_pred[p_idx] = iou

    # Count true positives
    tp_gt = int(np.sum(max_iou_gt[1:] >= iou_threshold))
    tp_pred = int(np.sum(max_iou_pred[1:] >= iou_threshold))

    precision = float(tp_pred / num_pred) if num_pred > 0 else 0.0
    recall = float(tp_gt / num_gt) if num_gt > 0 else 0.0

    if (precision + recall) > 0.0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0

    return {
        "lcd": int(lcd),
        "lesion_f1": float(f1),
        "lesion_precision": float(precision),
        "lesion_recall": float(recall),
        "pred_lesion_count": int(num_pred),
        "gt_lesion_count": int(num_gt),
        "tp_pred": int(tp_pred),
        "tp_gt": int(tp_gt),
        "fp_lesions": int(num_pred - tp_pred),
        "fn_lesions": int(num_gt - tp_gt),
    }


def analyze_gt_components(
    pred: Union[np.ndarray, "torch.Tensor"],
    gt: Union[np.ndarray, "torch.Tensor"],
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    iou_threshold: float = 0.2,
    connectivity: int = 26,
) -> List[Dict[str, Any]]:
    """Evaluates per-component detection status for every ground-truth lesion island.

    Used to distinguish patient-level small total lesion burden from individual
    small satellite components co-occurring inside multi-lesion scans.
    """
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    if hasattr(gt, "detach"):
        gt = gt.detach().cpu().numpy()

    pred_arr = np.asarray(pred) > 0
    gt_arr = np.asarray(gt) > 0

    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"Shape mismatch in analyze_gt_components: pred {pred_arr.shape} vs gt {gt_arr.shape}"
        )

    if connectivity == 6:
        struct = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        struct = ndimage.generate_binary_structure(3, 2)
    elif connectivity == 26:
        struct = ndimage.generate_binary_structure(3, 3)
    else:
        raise ValueError(f"Unsupported connectivity: {connectivity}. Choose from [6, 18, 26].")

    gt_labeled, num_gt = ndimage.label(gt_arr, structure=struct)
    if num_gt == 0:
        return []

    pred_labeled, num_pred = ndimage.label(pred_arr, structure=struct)
    voxel_vol_ml = float(np.prod(voxel_spacing)) / 1000.0
    gt_sizes = np.bincount(gt_labeled.ravel())
    gt_total_vol_ml = float(gt_arr.sum()) * voxel_vol_ml

    max_iou_gt = np.zeros(num_gt + 1, dtype=np.float32)
    max_inter_gt = np.zeros(num_gt + 1, dtype=np.int64)

    if num_pred > 0:
        pred_sizes = np.bincount(pred_labeled.ravel())
        overlap = (pred_labeled > 0) & (gt_labeled > 0)
        p_indices = pred_labeled[overlap]
        g_indices = gt_labeled[overlap]
        pair_keys = p_indices.astype(np.int64) * (num_gt + 1) + g_indices.astype(np.int64)
        unique_pairs, pair_counts = np.unique(pair_keys, return_counts=True)

        for pair_key, count in zip(unique_pairs, pair_counts):
            p_idx = int(pair_key // (num_gt + 1))
            g_idx = int(pair_key % (num_gt + 1))
            union = int(pred_sizes[p_idx] + gt_sizes[g_idx] - count)
            iou = float(count) / float(union) if union > 0 else 0.0
            if iou > max_iou_gt[g_idx]:
                max_iou_gt[g_idx] = iou
            if int(count) > max_inter_gt[g_idx]:
                max_inter_gt[g_idx] = int(count)

    records: List[Dict[str, Any]] = []
    for g_idx in range(1, num_gt + 1):
        vox = int(gt_sizes[g_idx])
        vol_ml = float(vox) * voxel_vol_ml
        iou_val = float(max_iou_gt[g_idx])
        inter_val = int(max_inter_gt[g_idx])
        records.append(
            {
                "comp_id": g_idx,
                "comp_voxels": vox,
                "comp_vol_ml": vol_ml,
                "gt_total_vol_ml": gt_total_vol_ml,
                "gt_lesion_count": int(num_gt),
                "is_multi_lesion_case": bool(num_gt >= 2),
                "max_iou": iou_val,
                "overlap_voxels": inter_val,
                "detected_iou20": bool(iou_val >= iou_threshold),
                "touched_any": bool(inter_val > 0),
            }
        )
    return records
