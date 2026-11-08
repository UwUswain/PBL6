"""MONAI data transforms for multi-modal 3D MRI preprocessing in ISLES-2022."""

from typing import Optional, Sequence, Tuple, Union
import cv2
import monai.data as md
import monai.transforms as mt
import numpy as np
import torch
from scipy.ndimage import label


def _binarize_mask(m):
    """Ensures binary mask values {0, 1} while remaining picklable for multiprocessing."""
    return (m > 0).to(dtype=m.dtype) if hasattr(m, "to") else (m > 0).astype(m.dtype)


def _is_positive(x):
    """Pickle-safe positive value selector for foreground brain cropping."""
    return x > 0


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


class RandCropByConnectedComponentsd(mt.MapTransform):
    """3D patch cropping centered uniformly across distinct connected stroke lesions.

    Clinical Rationale:
        Under standard RandCropByPosNegLabeld, massive core lesions (>50,000 voxels) consume
        over 99.9% of crop centers, leaving small punctate satellites (<50 voxels) virtually
        untrained. This transform labels individual 3D connected components (islands) via
        scipy.ndimage.label and assigns equal probability (1/K) to each component, guaranteeing
        that punctate and micro-ischemic infarcts receive robust representation in the training stream.
    """

    def __init__(
        self,
        keys: Sequence[str],
        label_key: str = "mask",
        spatial_size: Tuple[int, int, int] = (96, 96, 64),
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 2,
    ):
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size = np.asarray(spatial_size, dtype=int)
        self.pos = float(pos)
        self.neg = float(neg)
        self.num_samples = int(num_samples)

    def __call__(self, data: dict) -> list:
        d = dict(data)
        mask_obj = d[self.label_key]
        is_torch = isinstance(mask_obj, torch.Tensor)
        m_arr = mask_obj.detach().cpu().numpy() if is_torch else np.asarray(mask_obj)

        # Squeeze channel if channel-first [C, D, H, W]
        if m_arr.ndim == 4:
            m_spatial = m_arr[0]
        else:
            m_spatial = m_arr

        labeled, n_components = label((m_spatial > 0).astype(np.uint8))
        shape = np.array(m_spatial.shape)
        half_patch = self.spatial_size // 2

        samples = []
        total_ratio = self.pos + self.neg
        pos_prob = self.pos / total_ratio if total_ratio > 0 else 0.5

        for _ in range(self.num_samples):
            pick_pos = (np.random.rand() < pos_prob) and (n_components > 0)
            if pick_pos:
                comp_idx = np.random.randint(1, n_components + 1)
                coords = np.argwhere(labeled == comp_idx)
                center = coords[np.random.randint(len(coords))]
            else:
                # Random foreground/brain center
                center = np.array([np.random.randint(0, s) for s in shape])

            start = np.maximum(0, center - half_patch)
            end = start + self.spatial_size
            shift = np.maximum(0, end - shape)
            start = np.maximum(0, start - shift)
            end = start + self.spatial_size

            slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))

            sample = {}
            for k in self.keys:
                item = d[k]
                if is_torch and isinstance(item, torch.Tensor):
                    sample[k] = item[(slice(None),) + slices]
                else:
                    sample[k] = item[(slice(None),) + slices]
            samples.append(sample)

        return samples


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
            mt.SelectItemsd(keys=["image", "mask"]),
        ]

    # Explicit modality selection
    if modalities is not None and len(modalities) > 0:
        keys_to_load = [m.lower().strip() for m in modalities]
        concat_keys = list(keys_to_load)
        all_keys = concat_keys + ["mask"]
        return [
            mt.LoadImaged(keys=all_keys, ensure_channel_first=True),
            mt.ConcatItemsd(keys=concat_keys, name="image", dim=0),
            mt.SelectItemsd(keys=["image", "mask"]),
        ]

    # Default fallback: 3-channel (DWI, ADC, FLAIR)
    return [
        mt.LoadImaged(keys=["dwi", "adc", "flair", "mask"], ensure_channel_first=True),
        mt.ConcatItemsd(keys=["dwi", "adc", "flair"], name="image", dim=0),
        mt.SelectItemsd(keys=["image", "mask"]),
    ]


def get_train_transforms(
    patch_size: Tuple[int, int, int] = (96, 96, 64),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_samples: int = 2,
    pos_ratio: float = 1.0,
    neg_ratio: float = 1.0,
    use_edwi: bool = False,
    modalities: Optional[Sequence[str]] = None,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    sampling_mode: str = "standard",
) -> mt.Compose:
    """Builds the training transform pipeline with Paper 32 augmentations and foreground isolation."""
    load_and_fuse_steps = _build_load_and_fuse_steps(
        modalities=modalities,
        use_edwi=use_edwi,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    if sampling_mode == "component_aware":
        crop_step = RandCropByConnectedComponentsd(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=patch_size,
            pos=pos_ratio,
            neg=neg_ratio,
            num_samples=num_samples,
        )
    else:
        crop_step = mt.RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=patch_size,
            pos=pos_ratio,
            neg=neg_ratio,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
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
        # 7. Crop zero-intensity background to isolate brain foreground (Paper 32 Section 4.2)
        mt.CropForegroundd(
            keys=["image", "mask"],
            source_key="image",
            select_fn=_is_positive,
            margin=4,
        ),
        # 8. Clean up extra coordinate keys generated by CropForegroundd
        mt.SelectItemsd(keys=["image", "mask"]),
        # 9. Independent Z-score normalization per channel on non-zero brain voxels
        mt.NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),
        # 10. Zero-pad volume if any dimension is smaller than patch_size
        mt.SpatialPadd(
            keys=["image", "mask"],
            spatial_size=patch_size,
            mode="constant",
            constant_values=0,
        ),
        # 11. Balanced patch cropping (standard or component-aware)
        crop_step,
        # 12. Full 7 Data Augmentations from Paper 32 (Factorizer Appendix A):
        # 12.1 Random 3D Affine transform (rotation [-30, 30] deg, scaling [0.7, 1.3], prob 0.15)
        mt.RandAffined(
            keys=["image", "mask"],
            prob=0.15,
            rotate_range=(np.pi / 6, np.pi / 6, np.pi / 6),
            scale_range=(0.3, 0.3, 0.3),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        # 12.2 Random 3D Flips along all spatial axes (prob 0.5)
        mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        # 12.3 Additive white Gaussian noise (variance [0, 0.1], prob 0.15)
        mt.RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
        # 12.4 Gaussian smoothing (kernel width [0.5, 1.5] voxels, prob 0.15)
        mt.RandGaussianSmoothd(
            keys=["image"],
            prob=0.15,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        # 12.5 Random scale intensity (s in [0.7, 1.3], prob 0.15)
        mt.RandScaleIntensityd(keys=["image"], prob=0.15, factors=0.3),
        # 12.6 Random shift intensity (offsets in [-0.1, 0.1], prob 0.15)
        mt.RandShiftIntensityd(keys=["image"], prob=0.15, offsets=0.1),
        # 12.7 Adjust contrast / Gamma transform (gamma in [0.7, 1.5], prob 0.15)
        mt.RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.5)),
        # 13. Ensure clean PyTorch tensors
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
        # 7. Crop zero-intensity background to isolate brain foreground (Paper 32 Section 4.2)
        mt.CropForegroundd(
            keys=["image", "mask"],
            source_key="image",
            select_fn=_is_positive,
            margin=4,
        ),
        # 8. Clean up extra coordinate keys generated by CropForegroundd
        mt.SelectItemsd(keys=["image", "mask"]),
        # 9. Independent Z-score normalization per channel on non-zero brain voxels
        mt.NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),
        # 10. Ensure clean PyTorch tensors
        mt.EnsureTyped(keys=["image", "mask"], data_type="tensor", track_meta=False),
    ]

    return mt.Compose(load_and_fuse_steps + common_steps)


def get_isles_transforms(
    mode: str = "train",
    patch_size: Tuple[int, int, int] = (96, 96, 64),
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
