"""Post-processing utilities for 3D ischemic stroke lesion segmentation."""

import logging
from typing import Union
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
    min_size_voxels: int = 15,
    connectivity: int = 26,
) -> Union[np.ndarray, torch.Tensor]:
    """Removes spurious isolated micro-clusters below the voxel volume threshold.

    Clinical Rationale:
        Clinically relevant acute ischemic core lesions typically have volume >= 0.015 mL
        (~15 voxels at 1x1x1 mm³ resolution). Isolated clusters smaller than this threshold
        are primarily acquisition artifacts, skull edge noise, or gradient non-linearities.

    Args:
        mask: Binary segmentation mask (boolean or int), as NumPy array or PyTorch Tensor.
        min_size_voxels: Minimum number of contiguous foreground voxels to preserve (default: 15).
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

    if binary_arr.sum() == 0 or min_size_voxels <= 1:
        return mask

    if cc3d is not None:
        cleaned = cc3d.dust(
            binary_arr,
            threshold=min_size_voxels,
            connectivity=connectivity,
            in_place=False,
        )
    else:
        labeled, num_features = label(binary_arr)
        if num_features == 0:
            cleaned = binary_arr
        else:
            component_sizes = np.bincount(labeled.ravel())
            too_small = component_sizes < min_size_voxels
            too_small[0] = False
            cleaned = np.where(too_small[labeled], 0, binary_arr).astype(np.uint8)

    cleaned = cleaned.astype(orig_dtype)

    if is_torch:
        return torch.from_numpy(cleaned).to(device)
    return cleaned

