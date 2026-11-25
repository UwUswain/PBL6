"""Case files on the shared data volume (api writes uploads, the worker writes masks) and NIfTI checks."""

import hashlib
import math
import os
from pathlib import Path
import uuid

from fastapi import HTTPException, UploadFile
import nibabel as nib
import numpy as np

DATA_DIR = Path(os.environ.get("DATA_DIR", "/srv/data"))
MAX_UPLOAD_BYTES = 600 * 2**20  # uncompressed ISLES FLAIR is ~140 MB
MAX_VOXELS = 512 ** 3


def case_dir(case_id: uuid.UUID) -> Path:
    return DATA_DIR / "cases" / str(case_id)


def save_nifti_upload(upload: UploadFile, dest_dir: Path, stem: str, check=None):
    """Stream to a temp file with a size cap, check it, then move it into place.

    Returns (path, sha256, size, shape, spacing, affine). The client's filename is never used as a path, and a
    previous file is replaced only after the new one passed the checks (`check(shape, affine)` may raise).
    """
    name = (upload.filename or "").lower()
    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii" if name.endswith(".nii") else None
    if suffix is None:
        raise HTTPException(400, "Only .nii or .nii.gz files are accepted")
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f".upload-{uuid.uuid4().hex}{suffix}"
    digest, size = hashlib.sha256(), 0
    try:
        with open(tmp, "wb") as out:
            while chunk := upload.file.read(2**20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // 2**20} MB")
                digest.update(chunk)
                out.write(chunk)
        shape, spacing, affine = inspect_nifti(tmp)
        if check:
            check(shape, affine)
        final = dest_dir / f"{stem}{suffix}"
        for old in dest_dir.glob(f"{stem}.nii*"):  # this case's previous upload of the same sequence
            old.unlink()
        tmp.replace(final)
        return final, digest.hexdigest(), size, shape, spacing, affine
    finally:
        tmp.unlink(missing_ok=True)


def inspect_nifti(path: Path) -> tuple[list[int], list[float], np.ndarray]:
    """Header-only check (no full decompression, so a gzip bomb cannot exhaust memory)."""
    try:
        img = nib.load(str(path))
        shape = [int(s) for s in img.shape]
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
        affine = np.asarray(img.affine, dtype=float)
    except Exception:
        raise HTTPException(400, "Not a readable NIfTI file")
    if len(shape) == 4 and shape[3] == 1:
        shape = shape[:3]
    if len(shape) != 3 or min(shape) < 8 or math.prod(shape) > MAX_VOXELS:
        raise HTTPException(400, f"Expected one 3D volume, got shape {shape}")
    if not all(math.isfinite(z) and 0.1 <= z <= 10 for z in zooms) or not np.isfinite(affine).all():
        raise HTTPException(400, "Invalid voxel spacing or orientation in the header")
    return shape, zooms, affine
