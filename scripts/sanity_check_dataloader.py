"""Sanity check script for ISLES-2022 dataset loader and MONAI preprocessing.

Loads a sample training batch and a validation volume, validates tensor shapes,
intensity normalization, and verifies there are no NaNs or corrupted values.
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is on sys.path for direct CLI execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.dataset.dataloader import build_isles_dataloaders
from src.dataset.parser import parse_isles_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sanity_check_dataloader")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="ISLES-2022 DataLoader Sanity Check")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/ISLES-2022",
        help="Path to ISLES-2022 dataset root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size of patches.",
    )
    parser.add_argument(
        "--patch-h",
        type=int,
        default=96,
        help="Patch height.",
    )
    parser.add_argument(
        "--patch-w",
        type=int,
        default=96,
        help="Patch width.",
    )
    parser.add_argument(
        "--patch-d",
        type=int,
        default=32,
        help="Patch depth (slices).",
    )
    return parser.parse_args()


def main() -> None:
    """Runs data loader sanity checks."""
    args = parse_args()
    data_path = Path(args.data_dir).resolve()
    patch_size = (args.patch_h, args.patch_w, args.patch_d)

    logger.info("=" * 70)
    logger.info("ISLES-2022 DATALOADER SANITY CHECK")
    logger.info("=" * 70)
    logger.info("Dataset path: %s", data_path)
    logger.info("Patch dimensions: %s", patch_size)
    logger.info("Batch size: %d", args.batch_size)

    # 1. Check dataset discovery
    cases = parse_isles_dataset(data_path, require_mask=True)
    if not cases:
        logger.error("No valid cases discovered in %s", data_path)
        sys.exit(1)
    logger.info("Total valid cases discovered: %d", len(cases))

    # 2. Build DataLoaders
    logger.info("Building DataLoaders (Fold 0 of 5)...")
    train_loader, val_loader = build_isles_dataloaders(
        dataset_dir=data_path,
        batch_size=args.batch_size,
        patch_size=patch_size,
        target_spacing=(1.0, 1.0, 1.0),
        val_fold=0,
        n_splits=5,
        num_workers=0,
        use_cache=False,
        num_samples_per_volume=2,
    )

    # 3. Check Training Batch
    logger.info("Fetching 1 training patch batch...")
    train_batch = next(iter(train_loader))
    images = train_batch["image"]
    masks = train_batch["mask"]

    logger.info("--- Training Batch Details ---")
    logger.info("Images tensor shape : %s (Expected: [%d, 3, %d, %d, %d])", images.shape, args.batch_size, *patch_size)
    logger.info("Masks tensor shape  : %s (Expected: [%d, 1, %d, %d, %d])", masks.shape, args.batch_size, *patch_size)
    logger.info("Tensor dtypes       : image=%s, mask=%s", images.dtype, masks.dtype)

    # Assert correct dimensions
    assert images.ndim == 5, f"Images must be 5D [B, C, H, W, D], got {images.ndim}D"
    assert images.shape[1] == 3, f"Images must have 3 channels (DWI, ADC, FLAIR), got {images.shape[1]}"
    assert masks.shape[1] == 1, f"Masks must have 1 channel, got {masks.shape[1]}"
    assert not torch.isnan(images).any(), "NaN detected in training images!"
    assert not torch.isinf(images).any(), "Inf detected in training images!"

    # Channel stats (DWI, ADC, FLAIR)
    modality_names = ["DWI (ch 0)", "ADC (ch 1)", "FLAIR (ch 2)"]
    for ch, name in enumerate(modality_names):
        ch_data = images[:, ch, ...]
        logger.info(
            "%s -> min: %7.3f | mean: %7.3f | max: %7.3f | non-zero voxels: %d",
            name,
            float(ch_data.min()),
            float(ch_data.mean()),
            float(ch_data.max()),
            int((ch_data != 0).sum()),
        )

    # Mask stats
    mask_unique = torch.unique(masks).tolist()
    logger.info("Mask unique values  : %s (Must be subset of [0.0, 1.0])", mask_unique)
    assert set(mask_unique).issubset({0.0, 1.0}), f"Mask values must be binary, got {mask_unique}"
    logger.info("Lesion voxels in batch: %d / %d (%4.2f%%)", int(masks.sum()), masks.numel(), 100.0 * float(masks.sum()) / masks.numel())

    # 4. Check Validation Full Volume
    logger.info("--- Validation Case Details ---")
    logger.info("Fetching 1 validation volume...")
    val_batch = next(iter(val_loader))
    val_image = val_batch["image"]
    val_mask = val_batch["mask"]
    logger.info("Val Image shape : %s", val_image.shape)
    logger.info("Val Mask shape  : %s", val_mask.shape)
    assert not torch.isnan(val_image).any(), "NaN detected in validation image!"

    logger.info("=" * 70)
    logger.info("SANITY CHECK PASSED SUCCESSFULLY! Ready for model training.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
