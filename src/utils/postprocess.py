"""Post-processing utilities for 3D ischemic stroke lesion segmentation."""

import logging
from typing import Optional, Sequence, Union
import numpy as np
import torch

try:
    import cc3d
except ImportError:
    cc3d = None

from scipy.ndimage import label

logger = logging.getLogger(__name__)


def remove_small_lesion_islands(
    mask: Union[np.ndarray, torch.Tensor],
    min_size_voxels: Optional[int] = None,
    min_size_ml: float = 0.008,
    voxel_spacing: Optional[Sequence[float]] = None,
    connectivity: int = 26,
) -> Union[np.ndarray, torch.Tensor]:
    """Removes spurious isolated micro-clusters below the physical volume or voxel threshold.

    Clinical Rationale:
        ISLES-2022 dataset pathology (Hernandez Petzsche et al., Nature Sci Data 2022) demonstrates
        that micro-embolic stroke showers frequently manifest as genuine ischemic foci between 8 and
        15 voxels (approx 0.008 - 0.015 mL at 1mm isotropic resolution). A rigid threshold of 15 voxels
        wipes out true punctate infarcts, causing false negative lesion detection (Lesion F1 = 0).
        This function dynamically scales thresholds according to physical voxel spacing (in mm³ / mL)
        to preserve true micro-infarcts while eliminating high-frequency acquisition noise.

    Args:
        mask: Binary segmentation mask (boolean or int), as NumPy array or PyTorch Tensor.
        min_size_voxels: Explicit minimum number of contiguous foreground voxels. If specified,
            takes precedence over min_size_ml.
        min_size_ml: Physical minimum volume in milliliters (default: 0.008 mL = 8 mm³).
        voxel_spacing: Sequence of (sx, sy, sz) voxel dimensions in mm. Used to compute
            physical volume threshold when min_size_voxels is None.
        connectivity: 3D neighborhood connectivity (default: 26 for full 3D neighborhood).

    Returns:
        Cleaned binary segmentation mask with identical type and shape.
    """
    is_torch = isinstance(mask, torch.Tensor)
    if is_torch:
        device = mask.device
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)

    orig_dtype = arr.dtype
    binary_arr = (arr > 0).astype(np.uint8)

    if binary_arr.sum() == 0:
        return mask

    # Determine effective cutoff in voxels
    if min_size_voxels is not None:
        effective_threshold = int(min_size_voxels)
    elif voxel_spacing is not None and len(voxel_spacing) >= 3:
        voxel_vol_ml = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]) / 1000.0
        if voxel_vol_ml > 0:
            effective_threshold = max(3, int(np.round(min_size_ml / voxel_vol_ml)))
        else:
            effective_threshold = 8
    else:
        # Default fallback for normalized 1mm³ isotropic space: 8 voxels = 0.008 mL
        effective_threshold = 8

    if effective_threshold <= 1:
        return mask

    if cc3d is not None:
        cleaned = cc3d.dust(
            binary_arr,
            threshold=effective_threshold,
            connectivity=connectivity,
            in_place=False,
        )
    else:
        labeled, num_features = label(binary_arr)
        if num_features == 0:
            cleaned = binary_arr
        else:
            component_sizes = np.bincount(labeled.ravel())
            too_small = component_sizes < effective_threshold
            too_small[0] = False
            cleaned = np.where(too_small[labeled], 0, binary_arr).astype(np.uint8)

    cleaned = cleaned.astype(orig_dtype)

    if is_torch:
        return torch.from_numpy(cleaned).to(device)
    return cleaned
