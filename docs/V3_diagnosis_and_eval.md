# HƯỚNG DẪN CHẨN ĐOÁN SUY GIẢM MÔ HÌNH V3 TRÊN KAGGLE (PHIÊN BẢN v2.1)

> **Dành cho Long:** Copy toàn bộ cell code bên dưới dán đè vào Kaggle Notebook `ISLES-2022 DATASET ANALYSIS`.
> - Chạy chẩn đoán đối chứng Epoch 70 vs Epoch 150.
> - Kiểm tra số học NaN/Inf, đối chiếu độ chính xác AMP FP16 vs FP32.
> - Quét ngưỡng 0.05–0.50 và đo độ méo aux target downsampling của Bài 22 (khoảng 3–4 phút).
> - Mỗi lượt chạy sẽ tự động lưu vào thư mục riêng có timestamp để không ghi đè artifact.

---

## BƯỚC 1: ĐƯA CHECKPOINT VERSION 23 VÀO NOTEBOOK KAGGLE

1. Trong giao diện Kaggle Notebook, nhìn sang panel **Input** bên phải.
2. Bấm nút **`+ Add Input`** $\rightarrow$ chọn tab **`Your Work`** (hoặc `Notebook Output Files`).
3. Tìm và chọn notebook **`ISLES-2022 DATASET ANALYSIS`** (Version 23 vừa chạy xong).
4. Kaggle sẽ tự động mount toàn bộ thư mục output chứa `best_metric_model.pth` và `last_checkpoint.pth` vào đường dẫn `/kaggle/input/isles-2022-dataset-analysis/`.

---

## BƯỚC 2: TOÀN BỘ KHỐI CODE 1 CELL DUY NHẤT (PASTE ĐÈ LÊN CELL 1 TRÊN KAGGLE)

```python
%pip install -q connected-components-3d==4.1.0

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

# ============================================================================
# 1. CẤU HÌNH PHIÊN CHẨN ĐOÁN & COMMIT SHA
# ============================================================================
REPO_URL = "https://github.com/UwUswain/PBL6.git"
BRANCH_NAME = "long_exp"
FOLD = 0

# SHA hợp lệ trên origin/long_exp (đã bao gồm sửa fingerprint, fatal CUDA check, run identity parity)
COMMIT_SHA = "277cdcd8801855fdfd75c25a422ee041d66a7ad2"

PINNED_ENV = {
    "python_major_minor": "3.13",
    "torch_cuda": "13.0",
    "packages": {
        "torch": "2.11.0+cu130",
        "monai": "1.6.1",
        "nibabel": "5.4.2",
        "numpy": "2.1.3",
        "scipy": "1.16.3",
        "pandas": "2.3.3",
        "scikit-learn": "1.6.1",
        "PyYAML": "6.0.3",
        "connected-components-3d": "4.1.0",
    },
}

# Đường dẫn tường minh đến Checkpoint Best (Epoch 70) và Last (Epoch 150)
CKPT_BEST_PATH = Path("/kaggle/input/isles-2022-dataset-analysis/runs_paired_v3_patch_96_96_80_150ep_fold0_fold0/dual_segresnet_fold0_20261113_091807/best_metric_model.pth")
CKPT_LAST_PATH = Path("/kaggle/input/isles-2022-dataset-analysis/runs_paired_v3_patch_96_96_80_150ep_fold0_fold0/dual_segresnet_fold0_20261113_091807/last_checkpoint.pth")

print(f"Target FOLD: {FOLD} | COMMIT: {COMMIT_SHA}")

# ============================================================================
# 2. KIỂM TRA MÔI TRƯỜNG NGHIÊM NGẶT & GIT CHECKOUT
# ============================================================================
def installed_ver(dist_name: str):
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None

print("\n>>> [1/5] Auditing GPU and Environment Constraints against PINNED_ENV...")
subprocess.run(["nvidia-smi"], check=False)

import torch
assert torch.cuda.is_available(), "CUDA GPU is required for Kaggle diagnostic run!"
gpu_name = torch.cuda.get_device_name(0)
gpu_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
torch_cuda_actual = getattr(torch.version, "cuda", None)
print(f"GPU: {gpu_name} ({gpu_mem_mb:.1f} MB VRAM) | PyTorch: {torch.__version__} (CUDA: {torch_cuda_actual})")

py_actual = f"{sys.version_info.major}.{sys.version_info.minor}"
if py_actual != PINNED_ENV["python_major_minor"]:
    raise RuntimeError(
        f"Python version mismatch! Expected {PINNED_ENV['python_major_minor']}, got {py_actual}"
    )

if PINNED_ENV.get("torch_cuda") and torch_cuda_actual != PINNED_ENV["torch_cuda"]:
    raise RuntimeError(
        f"CUDA version mismatch! Expected {PINNED_ENV['torch_cuda']}, got {torch_cuda_actual}"
    )

for pkg_name, exp_ver in PINNED_ENV["packages"].items():
    cur_ver = installed_ver(pkg_name)
    if cur_ver is None:
        if exp_ver is None:
            raise RuntimeError(f"Required package {pkg_name} is missing!")
        print(f"Package {pkg_name} is missing. Installing exact pinned version {exp_ver}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"{pkg_name}=={exp_ver}"])
        cur_ver = installed_ver(pkg_name)
        if cur_ver != exp_ver:
            raise RuntimeError(f"Failed to install {pkg_name}=={exp_ver}, got {cur_ver}")
    elif exp_ver is not None and cur_ver != exp_ver:
        raise RuntimeError(
            f"Package version mismatch for {pkg_name}: expected {exp_ver}, found {cur_ver}!"
        )
    print(f"  Verified {pkg_name:24s}: {cur_ver}")

repo_dir = Path("/kaggle/working/PBL6")
if not repo_dir.exists():
    print(f"\nCloning repository from {REPO_URL} (branch: {BRANCH_NAME})...")
    subprocess.check_call(["git", "clone", "--branch", BRANCH_NAME, REPO_URL, str(repo_dir)])
else:
    print(f"\nRepository exists at {repo_dir}. Fetching origin...")
    subprocess.check_call(["git", "-C", str(repo_dir), "fetch", "--all"])

print(f"Checking out pinned COMMIT_SHA: {COMMIT_SHA}...")
subprocess.check_call(["git", "-C", str(repo_dir), "checkout", COMMIT_SHA])

active_sha = subprocess.check_output(
    ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
).strip()
assert active_sha == COMMIT_SHA, f"Git SHA mismatch! Expected {COMMIT_SHA}, got {active_sha}"
print(f"Git checkout verified at: {active_sha}")

os.chdir(str(repo_dir))
sys.path.insert(0, str(repo_dir))

# ============================================================================
# 3. GIẢI NÉN VÀ XÁC THỰC DATASET ISLES-2022 (250 CASES)
# ============================================================================
raw_data_dir = Path("/kaggle/working/data/raw/ISLES-2022")
raw_data_dir.mkdir(parents=True, exist_ok=True)

existing_cases = list(raw_data_dir.glob("sub-strokecase*"))
if len(existing_cases) < 250:
    print("\n>>> [2/5] Locating and extracting ISLES-2022 archive (.sip / .zip)...")
    sip_candidates = [
        Path("/kaggle/input/datasets/ivesunderthiscobal/isles-2022-250cases-sip/ISLES-2022.sip"),
        Path("/kaggle/input/datasets/ivesunderthiscobal/isles-2022-250cases-sip/ISLES-2022.zip"),
    ] + list(Path("/kaggle/input").rglob("ISLES-2022.sip")) + list(Path("/kaggle/input").rglob("ISLES-2022.zip"))
    
    archive_to_extract = None
    for cand in sip_candidates:
        if cand.exists():
            archive_to_extract = cand
            break
            
    assert archive_to_extract is not None, "Could not locate ISLES-2022 archive (.sip or .zip) in /kaggle/input!"
    print(f"Extracting {archive_to_extract} to {raw_data_dir}...")
    with zipfile.ZipFile(archive_to_extract, "r") as zf:
        zf.extractall(raw_data_dir)

total_cases = len(list(raw_data_dir.glob("sub-strokecase*")))
print(f"ISLES-2022 dataset ready: {total_cases} cases verified.")
assert total_cases == 250, f"Expected 250 cases, found {total_cases}"
DATA_DIR = str(raw_data_dir)

# ============================================================================
# 4. CHỌN CHECKPOINT TƯỜNG MINH VÀ XÁC MINH MANIFEST / EPOCH 70-150
# ============================================================================
print("\n>>> [3/5] Explicit Checkpoint Discovery & Verification...")

def locate_explicit_run_dir() -> Path:
    # 1. Kiểm tra preferred paths nếu đã cấu hình
    if CKPT_BEST_PATH.exists() and CKPT_LAST_PATH.exists():
        if CKPT_BEST_PATH.parent == CKPT_LAST_PATH.parent:
            return CKPT_BEST_PATH.parent

    # 2. Kiểm tra danh sách các thư mục run khả dĩ chứa đồng thời cả best và last
    candidate_dirs = [
        Path("/kaggle/input/isles-2022-dataset-analysis/runs_paired_v3_patch_96_96_80_150ep_fold0_fold0/dual_segresnet_fold0_20261113_091807"),
        Path("/kaggle/input/isles-2022-v3-150ep-fold0/dual_segresnet_fold0_20261113_091807"),
        Path("/kaggle/input/isles-2022-v3-150ep-fold0/runs_paired_v3_patch_96_96_80_150ep_fold0_fold0/dual_segresnet_fold0_20261113_091807"),
        Path("/kaggle/working/runs_paired_v3_patch_96_96_80_150ep_fold0_fold0/dual_segresnet_fold0_20261113_091807"),
    ]
    for d in candidate_dirs:
        if (d / "best_metric_model.pth").exists() and (d / "last_checkpoint.pth").exists():
            return d

    # 3. Quét tìm tất cả các thư mục chứa đồng thời cả hai file
    valid_dirs = []
    for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
        for p in root.rglob("best_metric_model.pth"):
            parent = p.parent
            if (parent / "last_checkpoint.pth").exists() and parent not in valid_dirs:
                valid_dirs.append(parent)

    if len(valid_dirs) == 1:
        return valid_dirs[0]
    elif len(valid_dirs) > 1:
        dirs_str = "\n".join(f"  - {d}" for d in valid_dirs)
        raise RuntimeError(
            f"Tìm thấy nhiều thư mục chứa checkpoint 150ep! Không tự động lấy ngẫu nhiên.\n"
            f"Vui lòng gán trực tiếp CKPT_BEST_PATH và CKPT_LAST_PATH trong 1 thư mục cụ thể:\n{dirs_str}"
        )
    return None

run_dir = locate_explicit_run_dir()
if run_dir is None:
    raise FileNotFoundError(
        "CHƯA TÌM THẤY THƯ MỤC CHỨA CẢ 2 CHECKPOINT (best_metric_model.pth & last_checkpoint.pth)!\n"
        "Hãy kiểm tra lại đường dẫn CKPT_BEST_PATH và CKPT_LAST_PATH, hoặc mount output của notebook vào /kaggle/input/."
    )

ckpt_best = run_dir / "best_metric_model.pth"
ckpt_last = run_dir / "last_checkpoint.pth"

print(f"Verified Run Directory       : {run_dir}")
print(f"Checkpoint Best (Epoch 70)  : {ckpt_best}")
print(f"Checkpoint Last (Epoch 150) : {ckpt_last}")

# Xác minh metadata và epoch ngay trên CPU
data_b = torch.load(ckpt_best, map_location="cpu")
data_l = torch.load(ckpt_last, map_location="cpu")

assert data_b.get("epoch") == 70, f"Epoch mismatch in best checkpoint: expected 70, got {data_b.get('epoch')}"
assert data_l.get("epoch") == 150, f"Epoch mismatch in last checkpoint: expected 150, got {data_l.get('epoch')}"

mani_b = data_b.get("run_manifest")
mani_l = data_l.get("run_manifest")
assert mani_b is not None and mani_l is not None, "Both checkpoints MUST contain embedded run_manifest!"

# Đối chiếu code/config và thông tin nhận diện run giữa best và last
b_created = mani_b.get("created_at")
l_created = mani_l.get("created_at")
assert b_created == l_created, f"Run created_at mismatch! Best={b_created}, Last={l_created}. Checkpoints are from DIFFERENT runs!"

b_commit = mani_b.get("code_version", {}).get("git_commit")
l_commit = mani_l.get("code_version", {}).get("git_commit")
assert b_commit == l_commit, f"Git commit mismatch! Best={b_commit}, Last={l_commit}"

b_cfg_sha = mani_b.get("config", {}).get("config_sha256")
l_cfg_sha = mani_l.get("config", {}).get("config_sha256")
assert b_cfg_sha == l_cfg_sha, f"Config SHA256 mismatch! Best={b_cfg_sha}, Last={l_cfg_sha}"

b_train_sha = mani_b.get("split", {}).get("train_sha256")
l_train_sha = mani_l.get("split", {}).get("train_sha256")
assert b_train_sha == l_train_sha, f"Train split SHA256 mismatch! Best={b_train_sha}, Last={l_train_sha}"

b_val_sha = mani_b.get("split", {}).get("val_sha256")
l_val_sha = mani_l.get("split", {}).get("val_sha256")
assert b_val_sha == l_val_sha, f"Val split SHA256 mismatch! Best={b_val_sha}, Last={l_val_sha}"

print(f"Pre-check PASSED: Checkpoints confirmed from EXACT SAME RUN ({b_created})!")

# ============================================================================
# 5. THỰC THI CHẨN ĐOÁN VÀ LƯU VÀO THƯ MỤC RIÊNG CÓ TIMESTAMP
# ============================================================================
def run_streamed(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(p.stdout.readline, ""):
        print(line, end="", flush=True)
    p.stdout.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}")

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
diag_dir = Path(f"/kaggle/working/diagnostics_run_{run_timestamp}")
diag_dir.mkdir(parents=True, exist_ok=True)

print(f"\n>>> [4/5] Running Model Collapse Diagnostic Engine (v2.1)...")
print(f"Output directory: {diag_dir}")
print("Features: Manifest Cross-Check + NaN/Inf Pre-threshold Gating + AMP FP16 vs FP32 Parity + Aux Loss Distortion")

run_streamed([
    sys.executable, "-u", "scripts/diagnose_collapse.py",
    "--ckpt-best", str(ckpt_best),
    "--ckpt-last", str(ckpt_last),
    "--expected-best-epoch", "70",
    "--expected-last-epoch", "150",
    "--expected-cuda", "13.0",
    "--data-dir", DATA_DIR,
    "--fold", str(FOLD),
    "--sw-roi-size", "96", "96", "64",
    "--overlap", "0.25",
    "--num-cases", "5",
    "--output-dir", str(diag_dir),
])

report_file = diag_dir / "collapse_diagnosis_report.md"
if report_file.exists():
    print("\n" + "=" * 76)
    print("BÁO CÁO ĐO ĐẠC THỰC NGHIỆM CHI TIẾT (OBJECTIVE DATA ONLY):")
    print("=" * 76)
    print(report_file.read_text(encoding="utf-8"))
```
