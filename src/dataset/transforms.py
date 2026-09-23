"""MONAI data transforms for multi-modal 3D MRI preprocessing in ISLES-2022."""

from typing import Sequence, Tuple, Union
import monai.transforms as mt


def _binarize_mask(m):
    """Ensures binary mask values {0, 1} while remaining picklable for multiprocessing."""
    return (m > 0).to(dtype=m.dtype) if hasattr(m, "to") else (m > 0).astype(m.dtype)


def get_train_transforms(
    patch_size: Tuple[int, int, int] = (96, 96, 32),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_samples: int = 2,
    pos_ratio: float = 1.0,
    neg_ratio: float = 1.0,
) -> mt.Compose:
    """Builds the training transform pipeline with patch-based sampling and augmentation.

    Pipeline:
        1. Load DWI, ADC, FLAIR, and MASK.
        2. Resample FLAIR to match DWI reference coordinate grid (ResampleToMatchd).
        3. Concatenate DWI, ADC, FLAIR into a single 3-channel tensor 'image' [3, D, H, W].
        4. Delete individual modality keys to minimize memory overhead.
        5. Guarantee strictly binary mask {0, 1}.
        6. Resample 'image' (bilinear) and 'mask' (nearest) to isometric target spacing (1.0mm).
        7. Channel-wise non-zero Z-score intensity normalization.
        8. Spatial zero-padding if volume dimensions are smaller than patch_size.
        9. Balanced 3D patch cropping (RandCropByPosNegLabeld) to overcome stroke class imbalance.
        10. Random 3D spatial flipping augmentations.
        11. Convert to PyTorch Tensor.

    Args:
        patch_size: 3D dimensions for patch cropping (H, W, D), e.g. (96, 96, 32).
        target_spacing: Isotropic voxel spacing in mm, default (1.0, 1.0, 1.0).
        num_samples: Number of cropped patches per 3D volume per iteration.
        pos_ratio: Proportion of patches centered on stroke lesion foreground voxels.
        neg_ratio: Proportion of patches centered on background voxels.

    Returns:
        mt.Compose: MONAI training transform pipeline.
    """
    return mt.Compose([
        # 1. Load multi-modal NIfTI files (channel-first)
        mt.LoadImaged(keys=["dwi", "adc", "flair", "mask"], ensure_channel_first=True),
        # 2. Resample high-resolution FLAIR to align with DWI coordinate space
        mt.ResampleToMatchd(keys=["flair"], key_dst="dwi", mode="bilinear"),
        # 3. Concatenate modalities along channel axis -> shape [3, D, H, W]
        mt.ConcatItemsd(keys=["dwi", "adc", "flair"], name="image", dim=0),
        # 4. Remove individual modality keys
        mt.DeleteItemsd(keys=["dwi", "adc", "flair"]),
        # 5. Ensure binary mask values {0, 1}
        mt.Lambdad(
            keys=["mask"],
            func=_binarize_mask,
        ),
        # 6. Standardize voxel spacing to isotropic (1.0, 1.0, 1.0) mm
        mt.Spacingd(
            keys=["image", "mask"],
            pixdim=target_spacing,
            mode=("bilinear", "nearest"),
        ),
        # 7. Independent Z-score normalization per channel on non-zero brain voxels
        mt.NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),
        # 8. Zero-pad volume if any dimension is smaller than patch_size
        mt.SpatialPadd(
            keys=["image", "mask"],
            spatial_size=patch_size,
            mode="constant",
            constant_values=0,
        ),
        # 9. Balanced Pos/Neg patch cropping to prevent VRAM OOM and address class imbalance
        mt.RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=patch_size,
            pos=pos_ratio,
            neg=neg_ratio,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        # 10. Data augmentations: random flipping along spatial axes
        mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        # 11. Ensure clean PyTorch tensors
        mt.EnsureTyped(keys=["image", "mask"], data_type="tensor", track_meta=False),
    ])


def get_val_transforms(
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> mt.Compose:
    """Builds the validation and inference transform pipeline for full 3D volumes.

    Pipeline:
        1. Load DWI, ADC, FLAIR, and MASK.
        2. Resample FLAIR to match DWI reference coordinate grid (ResampleToMatchd).
        3. Concatenate DWI, ADC, FLAIR into a single 3-channel tensor 'image' [3, D, H, W].
        4. Delete individual modality keys to minimize memory overhead.
        5. Guarantee strictly binary mask {0, 1}.
        6. Resample 'image' (bilinear) and 'mask' (nearest) to isometric target spacing (1.0mm).
        7. Channel-wise non-zero Z-score intensity normalization.
        8. Convert to PyTorch Tensor.

    Args:
        target_spacing: Isotropic voxel spacing in mm, default (1.0, 1.0, 1.0).

    Returns:
        mt.Compose: MONAI validation transform pipeline.
    """
    return mt.Compose([
        # 1. Load multi-modal NIfTI files (channel-first)
        mt.LoadImaged(keys=["dwi", "adc", "flair", "mask"], ensure_channel_first=True),
        # 2. Resample high-resolution FLAIR to align with DWI coordinate space
        mt.ResampleToMatchd(keys=["flair"], key_dst="dwi", mode="bilinear"),
        # 3. Concatenate modalities along channel axis -> shape [3, D, H, W]
        mt.ConcatItemsd(keys=["dwi", "adc", "flair"], name="image", dim=0),
        # 4. Remove individual modality keys
        mt.DeleteItemsd(keys=["dwi", "adc", "flair"]),
        # 5. Ensure binary mask values {0, 1}
        mt.Lambdad(
            keys=["mask"],
            func=_binarize_mask,
        ),
        # 6. Standardize voxel spacing to isotropic (1.0, 1.0, 1.0) mm
        mt.Spacingd(
            keys=["image", "mask"],
            pixdim=target_spacing,
            mode=("bilinear", "nearest"),
        ),
        # 7. Independent Z-score normalization per channel on non-zero brain voxels
        mt.NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),
        # 8. Ensure clean PyTorch tensors
        mt.EnsureTyped(keys=["image", "mask"], data_type="tensor", track_meta=False),
    ])


def get_isles_transforms(
    mode: str = "train",
    patch_size: Tuple[int, int, int] = (96, 96, 32),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_samples: int = 2,
    pos_ratio: float = 1.0,
    neg_ratio: float = 1.0,
) -> mt.Compose:
    """Convenience factory routing to get_train_transforms or get_val_transforms."""
    mode_normalized = mode.lower().strip()
    if mode_normalized == "train":
        return get_train_transforms(
            patch_size=patch_size,
            target_spacing=target_spacing,
            num_samples=num_samples,
            pos_ratio=pos_ratio,
            neg_ratio=neg_ratio,
        )
    elif mode_normalized in ("val", "infer"):
        return get_val_transforms(target_spacing=target_spacing)
    raise ValueError(f"Invalid mode: '{mode}'. Must be 'train', 'val', or 'infer'.")
