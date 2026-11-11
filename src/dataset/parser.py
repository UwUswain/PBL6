"""BIDS-format dataset parser and K-Fold cross-validation splitter for ISLES-2022."""

from dataclasses import asdict, dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

import os

logger = logging.getLogger(__name__)

ISLES_2022_EXPECTED_COUNT = 250
ISLES_2022_EXPECTED_IDS = tuple(
    f"sub-strokecase{i:04d}" for i in range(1, ISLES_2022_EXPECTED_COUNT + 1)
)


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


def _find_file(folder: Optional[Path], *patterns: str) -> Optional[Path]:
    """Finds a file matching any pattern in a folder, checking direct path and rglob."""
    if folder is None or not folder.exists():
        return None
    for p in patterns:
        target = folder / p
        if target.is_file():
            return target
    for p in patterns:
        matches = [m for m in folder.rglob(p) if m.is_file()]
        if matches:
            return matches[0]
    return None


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

    # 1. Search for directory containing sub-strokecase folders
    subject_dirs: List[Path] = []
    if dataset_path.exists():
        subject_dirs = sorted(
            [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("sub-strokecase")]
        )

    # 2. If not found directly, walk recursively down dataset_path or common roots (e.g. /kaggle/input)
    if not subject_dirs:
        search_roots = [dataset_path] if dataset_path.exists() else []
        search_roots.extend([Path("/kaggle/input"), Path("data/raw")])
        for s_root in search_roots:
            if s_root.exists():
                for root, dirs, _ in os.walk(s_root):
                    candidates = [d for d in dirs if d.startswith("sub-strokecase")]
                    if candidates:
                        dataset_path = Path(root)
                        subject_dirs = sorted([dataset_path / d for d in candidates])
                        logger.info(
                            "Auto-detected ISLES dataset directory: %s (%d cases)",
                            dataset_path,
                            len(subject_dirs),
                        )
                        break
            if subject_dirs:
                break

    if not subject_dirs:
        raise FileNotFoundError(f"ISLES dataset directory with subject cases not found starting from {dataset_dir}")

    # 3. Locate derivatives directory (masks)
    if (dataset_path / "derivatives").exists():
        derivatives_dir = dataset_path / "derivatives"
    elif (dataset_path.parent / "derivatives").exists():
        derivatives_dir = dataset_path.parent / "derivatives"
    else:
        deriv_candidates = list(dataset_path.glob("**/derivatives")) or list(dataset_path.parent.glob("**/derivatives"))
        derivatives_dir = deriv_candidates[0] if deriv_candidates else dataset_path / "derivatives"

    logger.info("Using dataset_path: %s | derivatives_dir: %s", dataset_path, derivatives_dir)

    valid_cases: List[Dict[str, str]] = []
    skipped_count = 0

    for sub_dir in subject_dirs:
        case_id = sub_dir.name

        # Locate session folder (usually ses-0001)
        ses_dirs = [d for d in sub_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")]
        ses_dir = ses_dirs[0] if ses_dirs else sub_dir
        ses_name = ses_dir.name if ses_dirs else "ses-0001"

        # Locate DWI and ADC
        dwi_dir = ses_dir / "dwi" if (ses_dir / "dwi").exists() else ses_dir
        dwi_path = _find_file(
            dwi_dir,
            f"{case_id}_{ses_name}_dwi.nii.gz",
            f"{case_id}_{ses_name}_dwi.nii",
            "*dwi.nii.gz",
            "*dwi.nii",
            "*DWI.nii*",
        )
        adc_path = _find_file(
            dwi_dir,
            f"{case_id}_{ses_name}_adc.nii.gz",
            f"{case_id}_{ses_name}_adc.nii",
            "*adc.nii.gz",
            "*adc.nii",
            "*ADC.nii*",
        )

        if not dwi_path or not adc_path:
            logger.warning("Case %s missing DWI or ADC, skipping.", case_id)
            skipped_count += 1
            continue

        # Locate FLAIR (optional fallback to dwi_path if FLAIR is missing)
        anat_dir = ses_dir / "anat" if (ses_dir / "anat").exists() else ses_dir
        flair_path = _find_file(
            anat_dir,
            f"{case_id}_{ses_name}_FLAIR.nii.gz",
            f"{case_id}_{ses_name}_flair.nii.gz",
            f"{case_id}_{ses_name}_FLAIR.nii",
            f"{case_id}_{ses_name}_flair.nii",
            "*FLAIR.nii*",
            "*flair.nii*",
        )
        if not flair_path:
            flair_path = dwi_path

        # Locate ground truth mask
        mask_path = _find_file(
            derivatives_dir / case_id / ses_name,
            f"{case_id}_{ses_name}_msk.nii.gz",
            f"{case_id}_{ses_name}_msk.nii",
            "*msk.nii*",
            "*mask.nii*",
        )
        if not mask_path:
            mask_path = _find_file(
                derivatives_dir / case_id,
                f"{case_id}_{ses_name}_msk.nii.gz",
                f"{case_id}_{ses_name}_msk.nii",
                "*msk.nii*",
                "*mask.nii*",
            )

        if require_mask and not mask_path:
            logger.warning("Case %s missing ground truth mask in %s, skipping.", case_id, derivatives_dir)
            skipped_count += 1
            continue

        case_obj = ISLESCase(
            case_id=case_id,
            dwi=str(dwi_path),
            adc=str(adc_path),
            flair=str(flair_path),
            mask=str(mask_path) if mask_path else "",
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


def verify_isles_dataset(
    cases: List[Dict[str, str]],
    expected_num_cases: Optional[int] = ISLES_2022_EXPECTED_COUNT,
    require_exact_isles250_ids: bool = True,
    required_modalities: Sequence[str] = ("dwi", "adc"),
    require_mask: bool = True,
) -> Dict[str, Any]:
    """Validates dataset integrity (exact IDs, zero duplicates, existing non-empty files).

    Args:
        cases: Parsed list of case dictionaries from parse_isles_dataset().
        expected_num_cases: Required total number of subjects (None to skip count check).
        require_exact_isles250_ids: If True, requires exact set sub-strokecase0001..0250.
        required_modalities: Modalities whose paths must exist and be non-empty on disk.
        require_mask: If True, checks ground-truth mask file existence and non-zero size.

    Returns:
        Summary dictionary with verified count and SHA-256 IDs fingerprint.

    Raises:
        RuntimeError: If any count, ID, duplicate, or file-completeness check fails.
    """
    if not cases:
        raise RuntimeError("Dataset verification failed: 0 cases found.")

    case_ids = [c["case_id"] for c in cases]
    unique_ids = set(case_ids)
    if len(unique_ids) != len(case_ids):
        seen = set()
        dups = sorted({cid for cid in case_ids if cid in seen or seen.add(cid)})
        raise RuntimeError(
            f"Dataset verification failed: duplicate case_id entries detected ({len(dups)}): {dups[:10]}"
        )

    if expected_num_cases is not None and len(cases) != expected_num_cases:
        raise RuntimeError(
            f"Dataset verification failed: expected exactly {expected_num_cases} cases, found {len(cases)}."
        )

    if require_exact_isles250_ids:
        expected_set = set(ISLES_2022_EXPECTED_IDS)
        missing_ids = sorted(expected_set - unique_ids)
        extra_ids = sorted(unique_ids - expected_set)
        if missing_ids or extra_ids:
            raise RuntimeError(
                f"Dataset verification failed: IDs do not match canonical ISLES-2022 (0001..0250). "
                f"Missing ({len(missing_ids)}): {missing_ids[:10]} | Unexpected ({len(extra_ids)}): {extra_ids[:10]}"
            )

    check_keys = [m.lower() for m in required_modalities]
    if require_mask and "mask" not in check_keys:
        check_keys.append("mask")

    missing_files = []
    seen_paths: Dict[str, set] = {k: set() for k in check_keys}
    dup_paths = []

    for c in cases:
        cid = c["case_id"]
        for key in check_keys:
            raw_path = c.get(key, "")
            p = Path(raw_path) if raw_path else None
            if p is None or not p.is_file() or p.stat().st_size <= 0:
                missing_files.append((cid, key, raw_path))
            else:
                resolved = str(p.resolve())
                if resolved in seen_paths[key]:
                    dup_paths.append((cid, key, resolved))
                seen_paths[key].add(resolved)

    if missing_files:
        raise RuntimeError(
            f"Dataset verification failed: {len(missing_files)} missing or empty files. Sample: {missing_files[:5]}"
        )
    if dup_paths:
        raise RuntimeError(
            f"Dataset verification failed: {len(dup_paths)} duplicate file paths across subjects. Sample: {dup_paths[:5]}"
        )

    ids_sha256 = hash_id_list(case_ids, sort_ids=True)
    return {
        "num_cases": len(cases),
        "ids_sha256": ids_sha256,
        "first_id": min(case_ids),
        "last_id": max(case_ids),
    }


def hash_id_list(ids: Sequence[str], sort_ids: bool = False) -> str:
    """Computes a compact 16-char SHA-256 hex digest for a sequence of case IDs."""
    items = sorted(ids) if sort_ids else list(ids)
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]


def compute_split_fingerprint(
    train_cases: Sequence[Dict[str, str]],
    val_cases: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    """Computes deterministic SHA-256 fingerprints for train and validation case ID lists."""
    train_ids = [c["case_id"] for c in train_cases]
    val_ids = [c["case_id"] for c in val_cases]
    overlap = sorted(set(train_ids).intersection(set(val_ids)))
    if overlap:
        raise RuntimeError(f"Split leakage detected: {len(overlap)} overlapping IDs: {overlap[:5]}")

    train_sha256 = hash_id_list(train_ids, sort_ids=False)
    val_sha256 = hash_id_list(val_ids, sort_ids=False)
    return {
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_sha256": train_sha256,
        "val_sha256": val_sha256,
        "val_head": val_ids[:5],
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
