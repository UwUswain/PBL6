"""BIDS-format dataset parser and K-Fold cross-validation splitter for ISLES-2022."""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import os

logger = logging.getLogger(__name__)


@dataclass
class ISLESCase:
    """Represents a single multi-modal subject in ISLES-2022."""

    case_id: str
    dwi: str
    adc: str
    flair: str
    mask: str

    def to_dict(self) -> Dict[str, str]:
        """Converts case paths to a dictionary format expected by MONAI transforms."""
        return asdict(self)


def parse_isles_dataset(
    dataset_dir: Union[str, Path],
    require_mask: bool = True,
) -> List[Dict[str, str]]:
    """Scans and parses the raw ISLES-2022 BIDS dataset directory.

    Expected directory structure:
        dataset_dir/
        ├── sub-strokecaseXXXX/
        │   └── ses-0001/
        │       ├── dwi/
        │       │   ├── sub-strokecaseXXXX_ses-0001_dwi.nii.gz
        │       │   └── sub-strokecaseXXXX_ses-0001_adc.nii.gz
        │       └── anat/
        │           └── sub-strokecaseXXXX_ses-0001_FLAIR.nii.gz
        └── derivatives/
            └── sub-strokecaseXXXX/
                └── ses-0001/
                    └── sub-strokecaseXXXX_ses-0001_msk.nii.gz

    Args:
        dataset_dir: Path to the ISLES-2022 dataset root folder.
        require_mask: If True, only includes cases that have an existing ground truth mask.

    Returns:
        List of dictionaries with keys: 'case_id', 'dwi', 'adc', 'flair', 'mask'.
    """
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.exists():
        # Intelligent fallback: Auto-search in Kaggle /kaggle/input or common mounts
        found = False
        search_roots = [Path("/kaggle/input"), Path("data/raw")]
        for s_root in search_roots:
            if s_root.exists():
                for root, dirs, _ in os.walk(s_root):
                    if any(d.startswith("sub-strokecase") for d in dirs):
                        dataset_path = Path(root)
                        logger.info("Auto-detected ISLES dataset directory: %s", dataset_path)
                        found = True
                        break
            if found:
                break
        if not found:
            raise FileNotFoundError(f"ISLES dataset directory not found: {dataset_path}")

    # Check if subject folders are nested inside
    subject_dirs = sorted(
        [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("sub-strokecase")]
    )
    if not subject_dirs:
        for child in dataset_path.iterdir():
            if child.is_dir():
                sub_candidates = [d for d in child.iterdir() if d.is_dir() and d.name.startswith("sub-strokecase")]
                if sub_candidates:
                    dataset_path = child
                    subject_dirs = sorted(sub_candidates)
                    logger.info("Found subject directories in nested folder: %s", dataset_path)
                    break

    if (dataset_path / "derivatives").exists():
        derivatives_dir = dataset_path / "derivatives"
    elif (dataset_path.parent / "derivatives").exists():
        derivatives_dir = dataset_path.parent / "derivatives"
    else:
        deriv_candidates = list(dataset_path.glob("**/derivatives")) or list(dataset_path.parent.glob("**/derivatives"))
        derivatives_dir = deriv_candidates[0] if deriv_candidates else dataset_path / "derivatives"

    valid_cases: List[Dict[str, str]] = []
    skipped_count = 0

    for sub_dir in subject_dirs:
        case_id = sub_dir.name

        # Locate session folder (usually ses-0001)
        ses_dirs = [d for d in sub_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")]
        if not ses_dirs:
            logger.warning("No session folder found for %s, skipping.", case_id)
            skipped_count += 1
            continue
        ses_dir = ses_dirs[0]
        ses_name = ses_dir.name

        # Locate modalities
        dwi_path = ses_dir / "dwi" / f"{case_id}_{ses_name}_dwi.nii.gz"
        adc_path = ses_dir / "dwi" / f"{case_id}_{ses_name}_adc.nii.gz"
        flair_path = ses_dir / "anat" / f"{case_id}_{ses_name}_FLAIR.nii.gz"

        # Check raw image modalities existence
        missing_images = [
            name
            for name, path in [("DWI", dwi_path), ("ADC", adc_path), ("FLAIR", flair_path)]
            if not path.exists()
        ]
        if missing_images:
            logger.warning(
                "Case %s missing modalities: %s, skipping.", case_id, ", ".join(missing_images)
            )
            skipped_count += 1
            continue

        # Locate ground truth mask
        mask_path = (
            derivatives_dir / case_id / ses_name / f"{case_id}_{ses_name}_msk.nii.gz"
        )
        if require_mask and not mask_path.exists():
            logger.warning("Case %s missing ground truth mask at %s, skipping.", case_id, mask_path)
            skipped_count += 1
            continue

        case_obj = ISLESCase(
            case_id=case_id,
            dwi=str(dwi_path),
            adc=str(adc_path),
            flair=str(flair_path),
            mask=str(mask_path) if mask_path.exists() else "",
        )
        valid_cases.append(case_obj.to_dict())

    logger.info(
        "Dataset parsing complete: %d valid cases found (%d skipped) in %s",
        len(valid_cases),
        skipped_count,
        dataset_path,
    )
    return valid_cases


def get_kfold_splits(
    cases: List[Dict[str, str]],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
    """Splits a list of cases into K reproducible cross-validation folds.

    Args:
        cases: List of case dictionaries.
        n_splits: Number of folds (default: 5).
        seed: Random seed for reproducible shuffling.

    Returns:
        List of tuples: (train_cases, val_cases) for each fold index.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    n_samples = len(cases)
    if n_samples < n_splits:
        raise ValueError(f"Number of cases ({n_samples}) cannot be smaller than n_splits ({n_splits})")

    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    splits = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        train_cases = [cases[i] for i in train_idx]
        val_cases = [cases[i] for i in val_idx]
        splits.append((train_cases, val_cases))

        current = stop

    return splits
