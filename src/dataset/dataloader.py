"""PyTorch & MONAI DataLoader builders for ISLES-2022 multi-modal segmentation."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import monai.data as md
import torch

from src.dataset.parser import (
    compute_split_fingerprint,
    get_kfold_splits,
    hash_id_list,
    parse_isles_dataset,
    verify_isles_dataset,
)
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
    split_info_out: Optional[Dict[str, Any]] = None,
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

    # Resolve modality expectations & optional strict ISLES-250 integrity check
    num_samples = kwargs.get("num_samples_per_volume", 2)
    use_edwi = kwargs.get("use_edwi", False)
    modalities = kwargs.get("modalities", None)
    in_channels = kwargs.get("in_channels", None)
    if modalities is None and in_channels == 2:
        modalities = ["dwi", "adc"]
    active_modalities = modalities if modalities is not None else ["dwi", "adc", "flair"]

    strict_isles250 = bool(kwargs.get("strict_isles250", False))
    expected_num_cases = kwargs.get("expected_num_cases", 250 if strict_isles250 else None)
    if strict_isles250 or expected_num_cases is not None:
        ds_summary = verify_isles_dataset(
            cases=cases,
            expected_num_cases=int(expected_num_cases) if expected_num_cases is not None else None,
            require_exact_isles250_ids=strict_isles250,
            required_modalities=active_modalities,
            require_mask=True,
        )
        logger.info(
            "Dataset Guard PASSED: %d/%s cases verified (%s..%s, ids_sha256=%s)",
            ds_summary["num_cases"],
            str(expected_num_cases),
            ds_summary["first_id"],
            ds_summary["last_id"],
            ds_summary["ids_sha256"],
        )

    # 2. Split into patient-level K-Fold cross-validation folds
    splits = get_kfold_splits(cases, n_splits=n_splits, seed=seed)
    train_cases, val_cases = splits[active_fold]
    split_fp = compute_split_fingerprint(train_cases, val_cases)

    logger.info(
        "DataLoaders (Fold %d/%d): Train=%d subjects (sha256=%s), Val=%d subjects (sha256=%s, head=%s)",
        active_fold,
        n_splits,
        split_fp["train_count"],
        split_fp["train_sha256"],
        split_fp["val_count"],
        split_fp["val_sha256"],
        split_fp["val_head"],
    )

    # 3. Instantiate transform pipelines from Task 2
    sampling_mode = kwargs.get("sampling_mode", "standard")
    pos_ratio = float(kwargs.get("pos_ratio", 1.0))
    neg_ratio = float(kwargs.get("neg_ratio", 1.0))

    train_transforms = get_train_transforms(
        patch_size=patch_size,
        target_spacing=target_spacing,
        pos_ratio=pos_ratio,
        neg_ratio=neg_ratio,
        num_samples=num_samples,
        use_edwi=use_edwi,
        modalities=modalities,
        sampling_mode=sampling_mode,
    )
    val_transforms = get_val_transforms(
        target_spacing=target_spacing,
        use_edwi=use_edwi,
        modalities=modalities,
    )

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
    loader_seed = int(kwargs.get("train_seed", seed))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(loader_seed)

    train_loader = md.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=md.list_data_collate,
        generator=loader_generator,
        pin_memory=False,
    )

    val_loader = md.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    dataset_ids_sha256 = hash_id_list([c["case_id"] for c in cases], sort_ids=True)
    split_manifest = {
        "fold": int(active_fold),
        "n_splits": int(n_splits),
        "split_seed": int(seed),
        "train_seed": int(loader_seed),
        "total_cases": len(cases),
        "dataset_ids_sha256": dataset_ids_sha256,
        **split_fp,
    }
    setattr(train_loader, "split_manifest", split_manifest)
    if split_info_out is not None:
        split_info_out.update(split_manifest)

    return train_loader, val_loader


# Backward-compatible alias
build_isles_dataloaders = build_kfold_dataloaders
