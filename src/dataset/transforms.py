"""MONAI data transforms for multi-modal 3D MRI preprocessing in ISLES-2022."""

from typing import Optional, Sequence, Tuple, Union
import cv2
import monai.data as md
import monai.transforms as mt
import numpy as np
import torch


def _binarize_mask(m):
    """Ensures binary mask values {0, 1} while remaining picklable for multiprocessing."""
    return (m > 0).to(dtype=m.dtype) if hasattr(m, "to") else (m > 0).astype(m.dtype)


class GenerateEnhancedDWId(mt.MapTransform):
    """Generates an enhanced DWI (eDWI) channel using slice-wise CLAHE.

    Applies Contrast Limited Adaptive Histogram Equalization across axial slices
    to amplify local contrast and delineate punctate ischemic stroke lesions.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("dwi",),
        output_key: str = "edwi",
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            dwi_obj = d[key]
            # Convert to numpy if tensor
            is_tensor = hasattr(dwi_obj, "cpu")
            if is_tensor:
                arr = dwi_obj.detach().cpu().numpy()
            else:
                arr = np.array(dwi_obj)

            channel_first = (arr.ndim == 4)
            work_arr = arr[0] if channel_first else arr

            # Create CLAHE instance per call to ensure safe multiprocessing pickling
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
            edwi = np.zeros_like(work_arr, dtype=np.float32)

            # Process slice-by-slice along axial dimension (axis 0 = D)
            for z in range(work_arr.shape[0]):
                sl = work_arr[z]
                sl_min, sl_max = float(sl.min()), float(sl.max())
                if (sl_max - sl_min) > 1e-5 and np.count_nonzero(sl > sl_min) > 20:
                    sl_u8 = ((sl - sl_min) / (sl_max - sl_min) * 255.0).clip(0, 255).astype(np.uint8)
                    edwi_sl = clahe.apply(sl_u8).astype(np.float32) / 255.0
                    edwi[z] = edwi_sl
                else:
                    edwi[z] = 0.0

            if channel_first:
                edwi = edwi[np.newaxis, ...]

            if hasattr(md, "MetaTensor") and isinstance(dwi_obj, md.MetaTensor):
                d[self.output_key] = md.MetaTensor(
                    torch.from_numpy(edwi).to(dtype=dwi_obj.dtype, device=dwi_obj.device),
                    meta=dwi_obj.meta,
                )
            elif is_tensor:
                d[self.output_key] = torch.from_numpy(edwi).to(
                    dtype=dwi_obj.dtype,
                    device=dwi_obj.device,
                )
            else:
                d[self.output_key] = edwi
        return d


def _build_load_and_fuse_steps(
    modalities: Optional[Sequence[str]] = None,
    use_edwi: bool = False,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> list:
    if use_edwi:
        return [
            mt.LoadImaged(keys=["dwi", "adc", "mask"], ensure_channel_first=True),
            GenerateEnhancedDWId(
                keys=["dwi"],
                output_key="edwi",
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
            ),
            mt.ConcatItemsd(keys=["dwi", "adc", "edwi"], name="image", dim=0),
            mt.DeleteItemsd(keys=["dwi", "adc", "edwi"]),
        ]

    mods = list(modalities) if modalities is not None else ["dwi", "adc", "flair"]

    if "flair" in mods and "dwi" in mods:
        return [
            mt.LoadImaged(keys=mods + ["mask"], ensure_channel_first=True),
            mt.ResampleToMatchd(keys=["flair"], key_dst="dwi", mode="bilinear"),
            mt.ConcatItemsd(keys=mods, name="image", dim=0),
            mt.DeleteItemsd(keys=mods),
        ]
    else:
        return [
            mt.LoadImaged(keys=mods + ["mask"], ensure_channel_first=True),
            mt.ConcatItemsd(keys=mods, name="image", dim=0),
            mt.DeleteItemsd(keys=mods),
        ]


def get_train_transforms(
    patch_size: Tuple[int, int, int] = (96, 96, 32),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_samples: int = 2,
    pos_ratio: float = 1.0,
    neg_ratio: float = 1.0,
    use_edwi: bool = False,
    modalities: Optional[Sequence[str]] = None,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> mt.Compose:
    """Builds the training transform pipeline with patch-based sampling and augmentation."""
    load_and_fuse_steps = _build_load_and_fuse_steps(
        modalities=modalities,
        use_edwi=use_edwi,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    common_steps = [
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
        # 9. Balanced Pos/Neg patch cropping to overcome stroke class imbalance
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
    ]

    return mt.Compose(load_and_fuse_steps + common_steps)


def get_val_transforms(
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_edwi: bool = False,
    modalities: Optional[Sequence[str]] = None,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> mt.Compose:
    """Builds the validation and inference transform pipeline for full 3D volumes."""
    load_and_fuse_steps = _build_load_and_fuse_steps(
        modalities=modalities,
        use_edwi=use_edwi,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    common_steps = [
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
    ]

    return mt.Compose(load_and_fuse_steps + common_steps)


def get_isles_transforms(
    mode: str = "train",
    patch_size: Tuple[int, int, int] = (96, 96, 32),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_samples: int = 2,
    pos_ratio: float = 1.0,
    neg_ratio: float = 1.0,
    use_edwi: bool = False,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
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
            use_edwi=use_edwi,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )
    elif mode_normalized in ("val", "infer"):
        return get_val_transforms(
            target_spacing=target_spacing,
            use_edwi=use_edwi,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )
    raise ValueError(f"Invalid mode: '{mode}'. Must be 'train', 'val', or 'infer'.")
