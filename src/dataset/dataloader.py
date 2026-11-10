"""PyTorch & MONAI DataLoader builders for ISLES-2022 multi-modal segmentation."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import monai.data as md

from src.dataset.parser import get_kfold_splits, parse_isles_dataset
from src.dataset.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


def build_kfold_dataloaders(
    data_dir: Optional[Union[str, Path]] = None,
    fold: int = 0,
    n_splits: int = 5,
    batch_size: int = 2,
    patch_size: Tuple[int, int, int] = (96, 96, 32),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_workers: int = 0,
    use_cache: bool = False,
    cache_rate: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> Tuple[md.DataLoader, md.DataLoader]:
    """Constructs reproducible K-Fold train and validation DataLoaders for ISLES-2022.

    Args:
        data_dir: Path to ISLES-2022 dataset root.
        fold: Current fold index for validation [0, n_splits - 1].
        n_splits: Total number of cross-validation folds (default: 5).
        batch_size: Batch size of patches for training (default: 2).
        patch_size: 3D patch dimensions (H, W, D), e.g. (96, 96, 32).
        target_spacing: Voxel spacing (sx, sy, sz) in mm, default (1.0, 1.0, 1.0).
        num_workers: Number of background subprocesses for data loading.
        use_cache: Whether to use MONAI CacheDataset for faster RAM access.
        cache_rate: Fraction of dataset cached in RAM if use_cache=True.
        seed: Random seed for fold splitting reproducibility.
        **kwargs: Backward-compatibility arguments ('dataset_dir', 'val_fold', etc.).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Support backward-compatible kwargs
    resolved_dir = data_dir or kwargs.get("dataset_dir")
    if resolved_dir is None:
        raise ValueError("data_dir must be provided.")
    active_fold = kwargs.get("val_fold", fold)

    # 1. Parse all available BIDS cases
    cases = parse_isles_dataset(resolved_dir, require_mask=True)
    if not cases:
        raise RuntimeError(f"No valid ISLES cases found in {resolved_dir}")

    # 2. Split into patient-level K-Fold cross-validation folds
    splits = get_kfold_splits(cases, n_splits=n_splits, seed=seed)
    train_cases, val_cases = splits[active_fold]

    logger.info(
        "DataLoaders (Fold %d/%d): Train=%d subjects, Val=%d subjects",
        active_fold,
        n_splits,
        len(train_cases),
        len(val_cases),
    )

    # 3. Instantiate transform pipelines from Task 2
    num_samples = kwargs.get("num_samples_per_volume", 2)
    train_transforms = get_train_transforms(
        patch_size=patch_size,
        target_spacing=target_spacing,
        num_samples=num_samples,
    )
    val_transforms = get_val_transforms(target_spacing=target_spacing)

    # 4. Construct Datasets
    if use_cache:
        train_dataset = md.CacheDataset(
            data=train_cases,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        val_dataset = md.CacheDataset(
            data=val_cases,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        train_dataset = md.Dataset(data=train_cases, transform=train_transforms)
        val_dataset = md.Dataset(data=val_cases, transform=val_transforms)

    # 5. Build PyTorch/MONAI DataLoaders
    train_loader = md.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=md.list_data_collate,
        pin_memory=False,
    )

    val_loader = md.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


# Backward-compatible alias
build_isles_dataloaders = build_kfold_dataloaders
