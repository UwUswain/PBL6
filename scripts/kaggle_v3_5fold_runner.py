#!/usr/bin/env python3
"""
PBL6 — Kaggle V3 5-Fold Runner Cell
====================================
Single self-contained cell for training DualEncoderSegResNet V3 (standard
pos/neg sampling) on Folds 1–4 of ISLES-2022 (250 cases). Designed to match
the exact conditions of V3 Fold 0 trained at commit f06c53f.

Usage:
    1. Create a Kaggle notebook named pbl6-isles2022-v3-fold<N>.
    2. Attach dataset "isles-2022-250cases-sip".
    3. Set Accelerator = GPU T4 ×2.
    4. Paste this entire file as a single code cell.
    5. Change TARGET_FOLD below to 1, 2, 3, or 4.
    6. Click "Save & Run All (Commit)".

What this cell does NOT support:
    - Resume from interrupted training (commit f06c53f has no --resume-from).
      If a fold job is interrupted, it must be restarted from scratch.
    - Out-of-fold evaluation (separate script).
    - Modifying architecture, loss, sampler, optimizer, or LR schedule.

Estimated wall time per fold: ~3.5–4.5 hours on T4 (Fold 0 took 199.4 min).
This is an estimate, not a guarantee.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# USER CONFIGURATION — CHANGE ONLY THIS
# =============================================================================
TARGET_FOLD = 1  # <-- Set to 1, 2, 3, or 4

# =============================================================================
# LOCKED CONSTANTS (from V3 Fold 0 run_manifest.json)
# =============================================================================
PINNED_COMMIT = "f06c53fa4f66384ba1ce1ceaee7928db6a1b692d"
REPO_URL = "https://github.com/UwUswain/PBL6.git"
REPO_BRANCH = "long_exp"
CONFIG_FILE = "configs/paired_v3_posneg_250_fold0.yaml"
CONFIG_SHA256 = "8985264edd8aeb6bc39b1a686cd8370689cd6df4d15f62afdbbd60d86974df81"

# Environment lock from V3 Fold 0 manifest
PINNED_PYTHON_MAJOR_MINOR = "3.13"
PINNED_TORCH = "2.11.0+cu130"
PINNED_CUDA = "13.0"
PINNED_PACKAGES = {
    "monai": "1.6.1",
    "nibabel": "5.4.2",
    "numpy": "2.1.3",
    "scipy": "1.16.3",
    "pandas": "2.3.3",
    "scikit-learn": "1.6.1",
    "PyYAML": "6.0.3",
    # "connected-components-3d": null in manifest — NOT installed during Fold 0
}

# 10 core source file hashes from manifest
CORE_FILES_SHA256 = {
    "scripts/train.py": "27d67c810064f280179561440b684042a0be0fdec70c63c1c7d2229849e19930",
    "src/dataset/parser.py": "0c462ca408ea62b55925acd8f1750f780c158ebf92c8fd7cde7a7a71d9c6111b",
    "src/dataset/dataloader.py": "2bfc0e512a15e99aa10ac02ae6a8f18064d294081414f4e1b6dee1d6c852da7e",
    "src/dataset/transforms.py": "d2e7a64b964c664209c3db506b4ba8eacdc982dd412a086a7f58102bb1ce2c08",
    "src/models/builder.py": "7f552fd09fb2dbe925c14db01524fc14af104d67fea7c1a1610b73c569571eeb",
    "src/models/dual_segresnet.py": "dba0a0326999617c7f9991e070bfb37a56cc0c34b9b5f6c854875dbeab2b1e3b",
    "src/losses/builder.py": "a46fbe51305f63982343fbfe407abb64627e613c1109e6ecb1f1ac31bf03691b",
    "src/metrics/lesion.py": "51a1e51df6e7fa3d111ac4cba68618ba5743f7856a6f5114b9b0369883caf340",
    "src/utils/postprocess.py": "b98882802826fb50d8d811d8bfabc4da35aba3e458210cbe5ad2c181f29b9cbf",
    "src/utils/manifest.py": "d1dc1a7668d23b1c560c0e856e06387da891a39b8d71559b1073dad263b3a17b",
}

# Dataset fingerprint
EXPECTED_IDS_SHA256 = "d9e5eb71b2aae852"
EXPECTED_NUM_CASES = 250

# Split fingerprints (verified locally with check_5fold_splits.py)
SPLIT_FINGERPRINTS = {
    0: {"train_sha256": "5163d342f196ef3a", "val_sha256": "e505d2eb32670af7"},
    1: {"train_sha256": "5fc5204c454d7d0c", "val_sha256": "d5e62ad462bb2ada"},
    2: {"train_sha256": "2d11648c18dedee7", "val_sha256": "0241d17316891c16"},
    3: {"train_sha256": "8f4e5cbdc768f184", "val_sha256": "1cc914932b05035e"},
    4: {"train_sha256": "b4695f0f199bacce", "val_sha256": "2dfecd2f584bef38"},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    print(f"$ {cmd}", flush=True)
    return subprocess.run(
        cmd, shell=True, check=check, capture_output=capture,
        text=True if capture else None,
    )


def fatal(msg: str):
    """Print error and exit."""
    print(f"\n{'='*75}\n[FATAL] {msg}\n{'='*75}", flush=True)
    sys.exit(1)


# =============================================================================
# PHASE 0: VALIDATE TARGET_FOLD
# =============================================================================
if TARGET_FOLD not in (1, 2, 3, 4):
    fatal(f"TARGET_FOLD must be 1, 2, 3, or 4 (got {TARGET_FOLD}). "
          "Fold 0 is already trained and stored in reserve/V3_F0/.")

print(f"{'='*75}")
print(f"PBL6 V3 5-Fold Runner — Fold {TARGET_FOLD}")
print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
print(f"{'='*75}\n")

# =============================================================================
# PHASE 1: ENVIRONMENT VERIFICATION & LOCKING
# =============================================================================
print(">>> [1/6] Verifying and locking environment...\n")

# 1a. Python version
py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
if py_ver != PINNED_PYTHON_MAJOR_MINOR:
    fatal(f"Python major.minor mismatch: expected {PINNED_PYTHON_MAJOR_MINOR}, got {py_ver}")
print(f"  Python: {sys.version} ✓")

# 1b. PyTorch + CUDA (import early to check before spending time on pip)
import torch
torch_ver = torch.__version__
cuda_ver = torch.version.cuda
if torch_ver != PINNED_TORCH:
    fatal(f"PyTorch version mismatch: expected {PINNED_TORCH}, got {torch_ver}. "
          "Do NOT upgrade/downgrade torch — use the correct Kaggle base image.")
if cuda_ver != PINNED_CUDA:
    fatal(f"CUDA version mismatch: expected {PINNED_CUDA}, got {cuda_ver}")
print(f"  PyTorch: {torch_ver} ✓")
print(f"  CUDA: {cuda_ver} ✓")

# 1c. GPU check
if not torch.cuda.is_available():
    fatal("No CUDA GPU detected. Ensure the notebook is set to GPU T4 accelerator.")
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
print(f"  GPU 0: {gpu_name} ({gpu_mem:.1f} GB) ✓")

# 1d. Install/verify pinned packages
packages_to_install = []
import importlib
import importlib.metadata

# Map pip package names to importlib.metadata names
PKG_META_NAMES = {
    "monai": "monai",
    "nibabel": "nibabel",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "PyYAML": "PyYAML",
}

for pkg_name, expected_ver in PINNED_PACKAGES.items():
    meta_name = PKG_META_NAMES.get(pkg_name, pkg_name)
    try:
        installed_ver = importlib.metadata.version(meta_name)
        if installed_ver != expected_ver:
            packages_to_install.append(f"{pkg_name}=={expected_ver}")
            print(f"  {pkg_name}: {installed_ver} → need {expected_ver} (will install)")
        else:
            print(f"  {pkg_name}: {installed_ver} ✓")
    except importlib.metadata.PackageNotFoundError:
        packages_to_install.append(f"{pkg_name}=={expected_ver}")
        print(f"  {pkg_name}: NOT FOUND (will install)")

if packages_to_install:
    pip_cmd = f"{sys.executable} -m pip install --quiet --no-deps " + " ".join(packages_to_install)
    run_cmd(pip_cmd)
    # Re-verify after install
    for pkg_spec in packages_to_install:
        pkg_name = pkg_spec.split("==")[0]
        expected_ver = pkg_spec.split("==")[1]
        meta_name = PKG_META_NAMES.get(pkg_name, pkg_name)
        try:
            actual = importlib.metadata.version(meta_name)
        except importlib.metadata.PackageNotFoundError:
            fatal(f"Failed to install {pkg_name}=={expected_ver}")
        if actual != expected_ver:
            fatal(f"After pip install, {pkg_name} is {actual}, expected {expected_ver}")
    print("  Post-install verification passed ✓")

# 1e. Save environment snapshot
env_snapshot = {
    "python_version": sys.version,
    "python_major_minor": py_ver,
    "torch": torch_ver,
    "cuda": cuda_ver,
    "gpu": gpu_name,
    "packages": {},
}
for pkg_name in PINNED_PACKAGES:
    meta_name = PKG_META_NAMES.get(pkg_name, pkg_name)
    try:
        env_snapshot["packages"][pkg_name] = importlib.metadata.version(meta_name)
    except Exception:
        env_snapshot["packages"][pkg_name] = "NOT_FOUND"

print("\n  Environment verification PASSED ✓\n")

# =============================================================================
# PHASE 2: REPOSITORY CHECKOUT & CODE INTEGRITY
# =============================================================================
print(">>> [2/6] Setting up repository at pinned commit...\n")

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "PBL6_runner"  # Separate from any existing PBL6 checkout

# Clone fresh if needed, or verify existing
if REPO_DIR.exists():
    # Check if it's already at the right commit
    try:
        result = subprocess.run(
            f"cd {REPO_DIR} && git rev-parse HEAD",
            shell=True, capture_output=True, text=True, check=True,
        )
        current_sha = result.stdout.strip()
        if current_sha == PINNED_COMMIT:
            print(f"  Existing checkout at correct commit: {PINNED_COMMIT[:12]} ✓")
        else:
            print(f"  Existing checkout at wrong commit ({current_sha[:12]}), re-cloning...")
            # Do NOT rmtree — rename to preserve
            backup_name = f"PBL6_runner_old_{int(time.time())}"
            REPO_DIR.rename(WORK_DIR / backup_name)
            print(f"  Renamed old checkout to {backup_name}")
            REPO_DIR = WORK_DIR / "PBL6_runner"
            run_cmd(f"git clone --branch {REPO_BRANCH} {REPO_URL} {REPO_DIR}")
            run_cmd(f"cd {REPO_DIR} && git checkout {PINNED_COMMIT}")
    except subprocess.CalledProcessError:
        # Broken checkout
        backup_name = f"PBL6_runner_broken_{int(time.time())}"
        REPO_DIR.rename(WORK_DIR / backup_name)
        print(f"  Broken checkout renamed to {backup_name}")
        REPO_DIR = WORK_DIR / "PBL6_runner"
        run_cmd(f"git clone --branch {REPO_BRANCH} {REPO_URL} {REPO_DIR}")
        run_cmd(f"cd {REPO_DIR} && git checkout {PINNED_COMMIT}")
else:
    run_cmd(f"git clone --branch {REPO_BRANCH} {REPO_URL} {REPO_DIR}")
    run_cmd(f"cd {REPO_DIR} && git checkout {PINNED_COMMIT}")

# Verify commit
result = subprocess.run(
    f"cd {REPO_DIR} && git rev-parse HEAD",
    shell=True, capture_output=True, text=True, check=True,
)
actual_commit = result.stdout.strip()
if actual_commit != PINNED_COMMIT:
    fatal(f"Commit mismatch after checkout: {actual_commit} != {PINNED_COMMIT}")
print(f"  Commit verified: {actual_commit} ✓")

# Verify no dirty tracked files (untracked are OK)
result = subprocess.run(
    f"cd {REPO_DIR} && git diff --name-only HEAD",
    shell=True, capture_output=True, text=True, check=True,
)
if result.stdout.strip():
    fatal(f"Dirty tracked files detected after checkout:\n{result.stdout}")
print("  Working tree clean (tracked files) ✓")

# 2b. Verify 10 core file hashes
print("\n  Verifying core source file hashes...")
hash_failures = []
for rel_path, expected_hash in CORE_FILES_SHA256.items():
    full_path = REPO_DIR / rel_path
    if not full_path.is_file():
        hash_failures.append(f"  MISSING: {rel_path}")
        continue
    actual_hash = sha256_file(full_path)
    if actual_hash != expected_hash:
        hash_failures.append(
            f"  MISMATCH: {rel_path}\n"
            f"    expected: {expected_hash}\n"
            f"    actual:   {actual_hash}"
        )
    else:
        print(f"    {rel_path}: ✓")

if hash_failures:
    fatal("Core file hash verification FAILED:\n" + "\n".join(hash_failures))
print("  All 10 core file hashes match manifest ✓")

# 2c. Verify config file hash
config_path = REPO_DIR / CONFIG_FILE
if not config_path.is_file():
    fatal(f"Config file not found: {CONFIG_FILE}")
config_hash = sha256_file(config_path)
if config_hash != CONFIG_SHA256:
    fatal(f"Config file hash mismatch:\n  expected: {CONFIG_SHA256}\n  actual: {config_hash}")
print(f"  Config {CONFIG_FILE}: ✓\n")

# =============================================================================
# PHASE 3: DATASET DISCOVERY & VERIFICATION
# =============================================================================
print(">>> [3/6] Locating and verifying ISLES-2022 dataset...\n")

# Add repo to sys.path for imports
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Search for the .sip/.zip archive in /kaggle/input
archive_path = None
archive_candidates = []
input_dir = Path("/kaggle/input")
if input_dir.exists():
    for f in input_dir.rglob("*"):
        if f.is_file() and f.suffix in (".sip", ".zip"):
            if zipfile.is_zipfile(f):
                archive_candidates.append(f)
        # Also check files without extension
        elif f.is_file() and f.stat().st_size > 500_000_000:  # > 500MB
            if zipfile.is_zipfile(f):
                archive_candidates.append(f)

if len(archive_candidates) == 0:
    # Maybe already extracted? Check for sub-strokecase dirs
    pass
elif len(archive_candidates) == 1:
    archive_path = archive_candidates[0]
    print(f"  Found archive: {archive_path} ({archive_path.stat().st_size / 1e9:.2f} GB)")
elif len(archive_candidates) > 1:
    # Filter by name hint
    named = [a for a in archive_candidates if "isles" in a.name.lower() or "250" in a.name.lower()]
    if len(named) == 1:
        archive_path = named[0]
        print(f"  Found archive: {archive_path} ({archive_path.stat().st_size / 1e9:.2f} GB)")
    else:
        fatal(f"Multiple dataset archive candidates found, cannot disambiguate:\n"
              + "\n".join(f"  {a}" for a in archive_candidates))

# Extract if needed
DATA_DIR: Optional[Path] = None
EXTRACT_DIR = Path("/tmp/isles2022_extracted")

if archive_path is not None:
    if EXTRACT_DIR.exists():
        # Check if already extracted
        sub_dirs = [d for d in EXTRACT_DIR.rglob("sub-strokecase*") if d.is_dir()]
        if len(sub_dirs) >= 250:
            # Find the root
            first_sub = sub_dirs[0]
            DATA_DIR = first_sub.parent
            print(f"  Already extracted: {DATA_DIR} ({len(sub_dirs)} subject dirs)")
        else:
            print(f"  Extraction dir exists but incomplete ({len(sub_dirs)} dirs), re-extracting...")
            shutil.rmtree(EXTRACT_DIR)
    
    if DATA_DIR is None:
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting to {EXTRACT_DIR}...")
        t0 = time.time()
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(EXTRACT_DIR)
        print(f"  Extracted in {time.time() - t0:.1f}s")
        
        # Find data root
        sub_dirs = sorted([d for d in EXTRACT_DIR.rglob("sub-strokecase*") if d.is_dir()])
        if sub_dirs:
            DATA_DIR = sub_dirs[0].parent
        else:
            fatal(f"No sub-strokecase directories found after extraction in {EXTRACT_DIR}")

# Fallback: search /kaggle/input directly for already-extracted datasets
if DATA_DIR is None:
    for search_root in [Path("/kaggle/input"), Path("/tmp")]:
        if search_root.exists():
            sub_dirs = sorted([d for d in search_root.rglob("sub-strokecase*") if d.is_dir()])
            if len(sub_dirs) >= 250:
                DATA_DIR = sub_dirs[0].parent
                print(f"  Found pre-extracted dataset: {DATA_DIR}")
                break

if DATA_DIR is None:
    fatal("Could not find ISLES-2022 dataset. Attach the 'isles-2022-250cases-sip' dataset to this notebook.")

print(f"  Dataset root: {DATA_DIR}")

# 3b. Use repo's parser to verify dataset
from src.dataset.parser import (
    parse_isles_dataset,
    verify_isles_dataset,
    get_kfold_splits,
    compute_split_fingerprint,
)

cases = parse_isles_dataset(str(DATA_DIR), require_mask=True)
ds_info = verify_isles_dataset(
    cases,
    expected_num_cases=EXPECTED_NUM_CASES,
    require_exact_isles250_ids=True,
    required_modalities=["dwi", "adc"],
    require_mask=True,
)

actual_ids_sha = ds_info["ids_sha256"]
if actual_ids_sha != EXPECTED_IDS_SHA256:
    fatal(f"Dataset IDs SHA-256 mismatch: expected {EXPECTED_IDS_SHA256}, got {actual_ids_sha}")
print(f"  Verified {ds_info['num_cases']} cases, ids_sha256={actual_ids_sha} ✓")

# 3c. Verify split fingerprints for TARGET_FOLD
splits = get_kfold_splits(cases, n_splits=5, seed=42)
train_cases, val_cases = splits[TARGET_FOLD]
fp = compute_split_fingerprint(train_cases, val_cases)

expected_fp = SPLIT_FINGERPRINTS[TARGET_FOLD]
if fp["train_sha256"] != expected_fp["train_sha256"]:
    fatal(f"Fold {TARGET_FOLD} train_sha256 mismatch: "
          f"expected {expected_fp['train_sha256']}, got {fp['train_sha256']}")
if fp["val_sha256"] != expected_fp["val_sha256"]:
    fatal(f"Fold {TARGET_FOLD} val_sha256 mismatch: "
          f"expected {expected_fp['val_sha256']}, got {fp['val_sha256']}")

# Check no overlap
train_ids = set(c["case_id"] for c in train_cases)
val_ids = set(c["case_id"] for c in val_cases)
overlap = train_ids & val_ids
if overlap:
    fatal(f"Train/val overlap detected: {sorted(overlap)[:5]}")
if len(train_cases) != 200 or len(val_cases) != 50:
    fatal(f"Split size mismatch: {len(train_cases)} train, {len(val_cases)} val (expected 200/50)")

print(f"  Fold {TARGET_FOLD}: {len(train_cases)} train / {len(val_cases)} val ✓")
print(f"  train_sha256={fp['train_sha256']} ✓")
print(f"  val_sha256={fp['val_sha256']} ✓")
print(f"  No train/val overlap ✓\n")

# =============================================================================
# PHASE 4: PRE-TRAIN SUMMARY
# =============================================================================
print(">>> [4/6] Pre-train configuration summary\n")

import yaml
with open(config_path) as f:
    config_yaml = yaml.safe_load(f)

exp_cfg = config_yaml.get("experiment", {})
data_cfg = config_yaml.get("data", {})

summary = {
    "fold": TARGET_FOLD,
    "commit": PINNED_COMMIT,
    "config": CONFIG_FILE,
    "model": exp_cfg.get("model_name"),
    "init_filters": exp_cfg.get("init_filters"),
    "fusion_mode": exp_cfg.get("fusion_mode"),
    "deep_supervision": exp_cfg.get("deep_supervision"),
    "sampling_mode": exp_cfg.get("sampling_mode"),
    "loss": exp_cfg.get("loss"),
    "loss_gamma": exp_cfg.get("loss_gamma"),
    "max_epochs": exp_cfg.get("max_epochs"),
    "learning_rate": exp_cfg.get("learning_rate"),
    "weight_decay": exp_cfg.get("weight_decay"),
    "val_interval": exp_cfg.get("val_interval"),
    "seed": exp_cfg.get("seed"),
    "roi_size": data_cfg.get("roi_size"),
    "batch_size": data_cfg.get("batch_size"),
    "pos_ratio": data_cfg.get("pos_ratio"),
    "neg_ratio": data_cfg.get("neg_ratio"),
    "num_samples_per_volume": data_cfg.get("num_samples_per_volume"),
    "train_cases": len(train_cases),
    "val_cases": len(val_cases),
}

for k, v in summary.items():
    print(f"  {k}: {v}")
print()

# =============================================================================
# PHASE 5: TRAINING
# =============================================================================
print(">>> [5/6] Starting training...\n")

OUT_BASE = WORK_DIR / f"runs_v3_fold{TARGET_FOLD}"
OUT_BASE.mkdir(parents=True, exist_ok=True)

# Check if output dir already has results (do NOT overwrite)
existing_bests = list(OUT_BASE.rglob("best_metric_model.pth"))
if existing_bests:
    fatal(f"Output directory already contains results:\n"
          + "\n".join(f"  {p}" for p in existing_bests)
          + "\nRefusing to overwrite. Remove or rename the existing output "
          "directory before re-running.")

# Build the exact train command matching Fold 0
train_cmd = (
    f"cd {REPO_DIR} && "
    f"/usr/bin/python3 -u scripts/train.py "
    f"--config {CONFIG_FILE} "
    f"--data-dir {DATA_DIR} "
    f"--fold {TARGET_FOLD} "
    f"--save-dir {OUT_BASE} "
    f"--strict-isles250"
)

print(f"  Command: {train_cmd}")
print(f"  Start time: {datetime.now(timezone.utc).isoformat()}")
print(f"  Expected duration: ~3.5–4.5 hours (not guaranteed)")
print()

# Log files
artifacts_dir = WORK_DIR / f"artifacts_fold{TARGET_FOLD}"
artifacts_dir.mkdir(parents=True, exist_ok=True)
log_file = artifacts_dir / "train_stdout.log"

train_start = time.time()
train_returncode = None
train_error_msg = None

try:
    # Use Popen for real-time output + file logging (tee pattern)
    with open(log_file, "w", encoding="utf-8") as lf:
        process = subprocess.Popen(
            train_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line-buffered
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        process.wait()
        train_returncode = process.returncode

    if train_returncode != 0:
        train_error_msg = f"train.py exited with code {train_returncode}"

except Exception as e:
    train_error_msg = f"Exception during training: {type(e).__name__}: {e}"
    train_returncode = -1

finally:
    train_duration = time.time() - train_start
    
    # =========================================================================
    # PHASE 6: POST-TRAIN VERIFICATION & ARTIFACT ARCHIVAL
    # =========================================================================
    print(f"\n>>> [6/6] Post-train verification (duration: {train_duration/60:.1f} min)...\n")

    # Always save environment snapshot
    env_snapshot["train_duration_min"] = round(train_duration / 60, 2)
    env_snapshot["train_returncode"] = train_returncode
    env_snapshot["target_fold"] = TARGET_FOLD
    env_snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    env_file = artifacts_dir / "environment.json"
    with open(env_file, "w") as f:
        json.dump(env_snapshot, f, indent=2)
    print(f"  Environment snapshot saved to {env_file}")

    # Find the run directory (timestamped subdir created by train.py)
    run_dirs = sorted([
        d for d in OUT_BASE.iterdir()
        if d.is_dir() and d.name.startswith("dual_segresnet_fold")
    ]) if OUT_BASE.exists() else []

    if train_error_msg:
        print(f"\n  [ERROR] Training failed: {train_error_msg}")
        print(f"  Log saved to: {log_file}")
        
        # Archive whatever we have
        zip_name = f"v3_fold{TARGET_FOLD}_FAILED_{int(time.time())}.zip"
        zip_path = WORK_DIR / zip_name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add log
            if log_file.exists():
                zf.write(log_file, f"fold{TARGET_FOLD}/{log_file.name}")
            if env_file.exists():
                zf.write(env_file, f"fold{TARGET_FOLD}/{env_file.name}")
            # Add any partial run dirs
            for rd in run_dirs:
                for fp_item in rd.rglob("*"):
                    if fp_item.is_file():
                        arcname = f"fold{TARGET_FOLD}/{rd.name}/{fp_item.relative_to(rd)}"
                        zf.write(fp_item, arcname)
        print(f"  Partial artifacts archived to: {zip_path}")
        print(f"\n  Recovery: This fold must be restarted from scratch.")
        print(f"  Commit f06c53f does not support --resume-from.")
    else:
        # Training succeeded — verify outputs
        if not run_dirs:
            print("  [WARNING] No run directory found after training!")
        else:
            run_dir = run_dirs[-1]  # Latest
            print(f"  Run directory: {run_dir}")

            # 6a. Verify best_metric_model.pth
            best_ckpt = run_dir / "best_metric_model.pth"
            last_ckpt = run_dir / "last_checkpoint.pth"
            manifest_file = run_dir / "run_manifest.json"
            train_log = run_dir / "train.log"

            results_summary = {
                "fold": TARGET_FOLD,
                "commit": PINNED_COMMIT,
                "run_dir": str(run_dir),
                "train_duration_min": round(train_duration / 60, 2),
            }

            if best_ckpt.exists():
                ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
                best_epoch = ckpt.get("epoch", "?")
                best_dice = ckpt.get("best_dice", "?")
                best_sha = sha256_file(best_ckpt)
                
                # Verify manifest inside checkpoint matches our fold
                ckpt_manifest = ckpt.get("run_manifest", {})
                ckpt_fold = ckpt_manifest.get("split", {}).get("fold", None)
                if ckpt_fold is not None and ckpt_fold != TARGET_FOLD:
                    print(f"  [WARNING] Checkpoint manifest fold={ckpt_fold} != TARGET_FOLD={TARGET_FOLD}!")

                results_summary["best_epoch"] = best_epoch
                results_summary["best_dice"] = float(best_dice) if isinstance(best_dice, (int, float)) else best_dice
                results_summary["best_ckpt_sha256"] = best_sha
                
                print(f"  best_metric_model.pth: epoch={best_epoch}, dice={best_dice:.6f}")
                print(f"    SHA-256: {best_sha}")
            else:
                print("  [WARNING] best_metric_model.pth NOT FOUND!")
                results_summary["best_epoch"] = None
                results_summary["best_dice"] = None

            if last_ckpt.exists():
                last = torch.load(last_ckpt, map_location="cpu", weights_only=True)
                last_epoch = last.get("epoch", "?")
                last_sha = sha256_file(last_ckpt)
                results_summary["last_epoch"] = last_epoch
                results_summary["last_ckpt_sha256"] = last_sha
                
                if last_epoch != 100:
                    print(f"  [WARNING] last_checkpoint epoch={last_epoch}, expected 100!")
                else:
                    print(f"  last_checkpoint.pth: epoch={last_epoch} ✓")
                print(f"    SHA-256: {last_sha}")
            else:
                print("  [WARNING] last_checkpoint.pth NOT FOUND!")
                results_summary["last_epoch"] = None

            # Save summary
            summary_file = artifacts_dir / "results_summary.json"
            with open(summary_file, "w") as f:
                json.dump(results_summary, f, indent=2)
            print(f"\n  Results summary saved to: {summary_file}")

            # 6b. Archive all artifacts into a zip for download
            zip_name = f"v3_fold{TARGET_FOLD}_complete.zip"
            zip_path = WORK_DIR / zip_name
            print(f"\n  Archiving artifacts to {zip_path}...")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Run dir contents (checkpoints, manifest, train.log)
                for fp_item in run_dir.rglob("*"):
                    if fp_item.is_file():
                        arcname = f"fold{TARGET_FOLD}/{run_dir.name}/{fp_item.relative_to(run_dir)}"
                        zf.write(fp_item, arcname)
                # Artifacts (env, summary, stdout log)
                for fp_item in artifacts_dir.rglob("*"):
                    if fp_item.is_file():
                        arcname = f"fold{TARGET_FOLD}/artifacts/{fp_item.relative_to(artifacts_dir)}"
                        zf.write(fp_item, arcname)
            
            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"  Archive size: {zip_size_mb:.1f} MB")
            print(f"  Archive path: {zip_path}")

        print(f"\n{'='*75}")
        print(f"  FOLD {TARGET_FOLD} TRAINING COMPLETE")
        print(f"  Duration: {train_duration/60:.1f} minutes")
        if 'best_dice' in results_summary and results_summary['best_dice'] is not None:
            print(f"  Best Dice: {results_summary['best_dice']:.6f} at epoch {results_summary.get('best_epoch', '?')}")
        print(f"  Download: {zip_path}")
        print(f"{'='*75}")

