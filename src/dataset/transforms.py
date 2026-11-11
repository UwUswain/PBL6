"""MONAI data transforms for multi-modal 3D MRI preprocessing in ISLES-2022."""

from copy import deepcopy
from typing import Optional, Sequence, Tuple, Union
import cv2
import monai.data as md
import monai.transforms as mt
from monai.transforms.utils import correct_crop_centers, map_binary_to_indices
import numpy as np
import torch
from scipy import ndimage


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


class RandCropByConnectedComponentsd(mt.Randomizable, mt.MapTransform, mt.MultiSampleTrait):
    """3D patch cropping centered uniformly across distinct connected stroke lesions.

    Clinical Rationale:
        Under standard RandCropByPosNegLabeld, massive core lesions (>50,000 voxels) consume
        over 99.9% of crop centers, leaving small punctate satellites (<50 voxels) virtually
        untrained. This transform labels individual 3D connected components (islands) via
        scipy.ndimage.label and assigns equal probability (1/K) to each component, guaranteeing
        that punctate and micro-ischemic infarcts receive robust representation in the training stream.

    Controlled Parity with RandCropByPosNegLabeld:
        - Inherits monai.transforms.Randomizable and MultiSampleTrait so CacheDataset
          correctly caches ONLY up to the pre-crop deterministic transforms and re-crops
          dynamically on every epoch via self.R.
        - Uses MONAI's exact map_binary_to_indices(..., image_threshold) for negative/background
          center sampling and correct_crop_centers(...) for boundary clamping, ensuring the ONLY
          difference from RandCropByPosNegLabeld is uniform component-wise positive center selection.
    """

    def __init__(
        self,
        keys: Sequence[str],
        label_key: str = "mask",
        spatial_size: Tuple[int, int, int] = (96, 96, 64),
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 2,
        image_key: Optional[str] = "image",
        image_threshold: float = 0.0,
        allow_smaller: bool = False,
        connectivity: int = 26,
        allow_missing_keys: bool = False,
    ):
        mt.MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.image_key = image_key
        self.image_threshold = float(image_threshold)
        self.spatial_size = tuple(int(s) for s in spatial_size)
        self.pos = float(pos)
        self.neg = float(neg)
        self.num_samples = int(num_samples)
        self.allow_smaller = bool(allow_smaller)

        if connectivity == 6:
            self.connectivity_rank = 1
        elif connectivity == 18:
            self.connectivity_rank = 2
        elif connectivity == 26:
            self.connectivity_rank = 3
        else:
            raise ValueError(f"Unsupported connectivity: {connectivity}. Choose from [6, 18, 26].")

        self._structure = ndimage.generate_binary_structure(3, self.connectivity_rank)
        self.R = np.random.RandomState()
        self.centers: Optional[Tuple[Tuple[int, ...], ...]] = None

    def randomize(
        self,
        label_img: Union[np.ndarray, torch.Tensor],
        image: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """Selects balanced positive (uniform over components) and negative crop centers."""
        if isinstance(label_img, torch.Tensor) and label_img.ndim == 3:
            label_cf = label_img.unsqueeze(0)
        elif isinstance(label_img, np.ndarray) and label_img.ndim == 3:
            label_cf = label_img[np.newaxis, ...]
        else:
            label_cf = label_img

        if image is not None and isinstance(image, torch.Tensor) and image.ndim == 3:
            image_cf = image.unsqueeze(0)
        elif image is not None and isinstance(image, np.ndarray) and image.ndim == 3:
            image_cf = image[np.newaxis, ...]
        else:
            image_cf = image

        fg_indices, bg_indices = map_binary_to_indices(
            label_cf, image_cf, self.image_threshold
        )

        if isinstance(label_cf, md.MetaTensor):
            spatial_shape = tuple(int(x) for x in label_cf.peek_pending_shape())
        else:
            spatial_shape = tuple(int(x) for x in label_cf.shape[1:])

        n_fg = int(len(fg_indices))
        n_bg = int(len(bg_indices))
        if n_fg == 0 and n_bg == 0:
            raise ValueError("No sampling location available.")

        if n_fg == 0:
            eff_pos_ratio = 0.0
        elif n_bg == 0:
            eff_pos_ratio = 1.0
        else:
            total_ratio = self.pos + self.neg
            eff_pos_ratio = self.pos / total_ratio if total_ratio > 0 else 0.5

        if torch.is_tensor(bg_indices):
            bg_indices_np = bg_indices.detach().cpu().numpy()
        else:
            bg_indices_np = np.asarray(bg_indices)

        sample_is_pos = [bool(self.R.rand() < eff_pos_ratio) for _ in range(self.num_samples)]

        labeled = None
        n_components = 0
        obj_slices = []
        if any(sample_is_pos) and n_fg > 0:
            if isinstance(label_cf, torch.Tensor):
                m_np = label_cf.detach().cpu().numpy()
            else:
                m_np = np.asarray(label_cf)
            if m_np.shape[0] > 1:
                m_spatial = np.any(m_np[1:] > 0, axis=0)
            else:
                m_spatial = m_np[0] > 0
            labeled, n_components = ndimage.label(m_spatial.astype(np.uint8), structure=self._structure)
            if n_components > 0:
                obj_slices = ndimage.find_objects(labeled)

        centers = []
        for is_pos in sample_is_pos:
            if is_pos and n_components > 0 and labeled is not None:
                comp_idx = int(self.R.randint(1, n_components + 1))
                slc = obj_slices[comp_idx - 1]
                if slc is not None:
                    local_coords = np.argwhere(labeled[slc] == comp_idx)
                    picked = local_coords[int(self.R.randint(len(local_coords)))]
                    center = [
                        int(picked[dim] + slc[dim].start)
                        for dim in range(len(spatial_shape))
                    ]
                else:
                    coords = np.argwhere(labeled == comp_idx)
                    center = [int(c) for c in coords[int(self.R.randint(len(coords)))]]
            else:
                rand_idx = int(self.R.randint(len(bg_indices_np)))
                flat_idx = int(bg_indices_np[rand_idx])
                center = [int(c) for c in np.unravel_index(flat_idx, spatial_shape)]

            corrected = correct_crop_centers(
                center,
                self.spatial_size,
                spatial_shape,
                allow_smaller=self.allow_smaller,
            )
            centers.append(tuple(int(c) for c in corrected))

        self.centers = tuple(centers)

    def __call__(self, data: dict) -> list:
        d = dict(data)
        img_ref = d.get(self.image_key) if self.image_key is not None else None
        self.randomize(d[self.label_key], img_ref)

        assert self.centers is not None
        ret = [dict(d) for _ in range(self.num_samples)]
        for i in range(self.num_samples):
            for key in set(d.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(d[key])

        for i, center in enumerate(self.centers):
            cropper = mt.SpatialCrop(roi_center=center, roi_size=self.spatial_size)
            for key in self.key_iterator(d):
                ret[i][key] = cropper(d[key])

        return ret


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
            image_key="image",
            image_threshold=0,
            allow_smaller=False,
            connectivity=26,
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
