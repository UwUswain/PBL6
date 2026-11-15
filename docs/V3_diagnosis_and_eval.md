# HƯỚNG DẪN CHẨN ĐOÁN SUY GIẢM MÔ HÌNH V3 VÀ ĐÁNH GIÁ ĐỐI ĐẦU TRÊN KAGGLE

> **Dành cho Long:** Copy toàn bộ cell code bên dưới dán đè vào Kaggle Notebook `ISLES-2022 DATASET ANALYSIS`.
> - Chọn `RUN_MODE = "diagnose"`: Chạy chẩn đoán đối chứng Epoch 70 vs Epoch 150, kiểm tra số học NaN/Inf, đối chiếu độ chính xác AMP FP16 vs FP32, quét ngưỡng 0.05–0.50, và đo độ méo aux target downsampling của Bài 22 (khoảng 3–4 phút).
> - Chọn `RUN_MODE = "eval_v3_old_vs_new"`: Chấm điểm đối đầu chuẩn 4 metrics ISLES-2022 giữa V3 cũ (100ep, Dice 0.6779) và V3 mới (Ep 70, Dice 0.6816) trên 50 ca Fold 0 (khoảng 12 phút).

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
from pathlib import Path

# ============================================================================
# 1. CẤU HÌNH CHẾ ĐỘ CHẠY & COMMIT SHA XÁC THỰC
# ============================================================================
# Chế độ chạy:
# - "diagnose": Chẩn đoán Ep 70 vs Ep 150 (NaN/Inf, AMP vs FP32, threshold sweep, aux downsampling) (~3-4 phút)
# - "eval_v3_old_vs_new": Đánh giá đối đầu V3 cũ (100ep) vs V3 mới (Ep 70) trên 50 ca Fold 0 (~12 phút)
RUN_MODE = "diagnose"  # "diagnose" | "eval_v3_old_vs_new"

REPO_URL = "https://github.com/UwUswain/PBL6.git"
BRANCH_NAME = "long_exp"
FOLD = 0

# SHA hợp lệ trên origin/long_exp (đã bao gồm diagnose_collapse v2.0 và AMP/FP32 parity)
COMMIT_SHA = "d61fb828d2896b7d19f38be82afc74090715f950"

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

print(f"Target RUN_MODE: {RUN_MODE} | FOLD: {FOLD} | COMMIT: {COMMIT_SHA}")

# ============================================================================
# 2. KIỂM TRA MÔI TRƯỜNG & GIT CHECKOUT
# ============================================================================
def installed_ver(dist_name: str):
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None

print("\n>>> [1/5] Auditing GPU and Environment Constraints...")
subprocess.run(["nvidia-smi"], check=False)

import torch
assert torch.cuda.is_available(), "CUDA GPU is required for Kaggle diagnostic/evaluation run!"
gpu_name = torch.cuda.get_device_name(0)
gpu_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
print(f"GPU: {gpu_name} ({gpu_mem_mb:.1f} MB VRAM) | PyTorch: {torch.__version__}")

py_actual = f"{sys.version_info.major}.{sys.version_info.minor}"
if py_actual != PINNED_ENV["python_major_minor"]:
    print(f"Notice: Python version is {py_actual} (pinned target: {PINNED_ENV['python_major_minor']})")

for pkg_name, exp_ver in PINNED_ENV["packages"].items():
    cur_ver = installed_ver(pkg_name)
    if cur_ver is None:
        print(f"Installing missing package: {pkg_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])
    else:
        print(f"  Package {pkg_name:24s}: {cur_ver} (expected: {exp_ver or 'any'})")

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
print("\n>>> [3/5] Explicit Checkpoint Discovery & Header Verification...")

def resolve_explicit_checkpoint(expected_epoch: int, filename: str) -> Path:
    # 1. Kiểm tra các đường dẫn tường minh
    explicit_candidates = [
        Path(f"/kaggle/input/isles-2022-dataset-analysis/runs_paired_v3_patch_96_96_80_150ep_fold0_fold0/dual_segresnet_fold0_20261113_091807/{filename}"),
        Path(f"/kaggle/input/isles-2022-v3-150ep-fold0/{filename}"),
        Path(f"/kaggle/working/runs_paired_v3_patch_96_96_80_150ep_fold0_fold0/dual_segresnet_fold0_20261113_091807/{filename}"),
    ]
    for p in explicit_candidates:
        if p.exists():
            ckpt = torch.load(p, map_location="cpu")
            if ckpt.get("epoch") == expected_epoch:
                return p
                
    # 2. Tìm kiếm trong các thư mục input / working nếu đường dẫn đổi tên
    for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
        for p in root.rglob("*.pth"):
            if filename in p.name:
                try:
                    ckpt = torch.load(p, map_location="cpu")
                    if ckpt.get("epoch") == expected_epoch:
                        return p
                except Exception:
                    pass
    return None

ckpt_best = resolve_explicit_checkpoint(expected_epoch=70, filename="best_metric_model.pth")
ckpt_last = resolve_explicit_checkpoint(expected_epoch=150, filename="last_checkpoint.pth")

# Tìm V3 cũ (100ep, Fold 0, Dice 0.6779)
ckpt_v3_old = None
old_candidates = [
    Path("/kaggle/input/isles-2022-v3-v4-fold0/v3_best.pth"),
    Path("/kaggle/input/datasets/ivesunderthiscobal/isles-2022-v3-v4-fold0/v3_best.pth"),
] + list(Path("/kaggle/input").rglob("v3_best.pth"))
for p in old_candidates:
    if p.exists():
        ckpt_v3_old = p
        break

print(f"Checkpoint Best (Epoch 70)  : {ckpt_best}")
print(f"Checkpoint Last (Epoch 150) : {ckpt_last}")
print(f"Checkpoint V3 Cũ (100ep)    : {ckpt_v3_old}")

if RUN_MODE == "diagnose":
    if ckpt_best is None or ckpt_last is None:
        raise FileNotFoundError(
            "CHƯA TÌM THẤY CHECKPOINT 150-EPOCH HỢP LỆ!\n"
            "Hãy nhìn sang cột phải của Kaggle -> Bấm '+ Add Input' -> Chọn tab 'Your Work' ->\n"
            "Chọn notebook 'ISLES-2022 DATASET ANALYSIS' (Version 23 vừa chạy xong) để mount output vào /kaggle/input/."
        )

# ============================================================================
# 5. THỰC THI CHẾ ĐỘ CHẨN ĐOÁN HOẶC ĐÁNH GIÁ ĐỐI ĐẦU
# ============================================================================
def run_streamed(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(p.stdout.readline, ""):
        print(line, end="", flush=True)
    p.stdout.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}")

if RUN_MODE == "diagnose":
    diag_dir = Path("/kaggle/working/diagnostics_out")
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n>>> [4/5] Running Model Collapse Diagnostic Engine (v2.0)...")
    print("Features: Manifest Check + NaN/Inf Pre-threshold Gating + AMP FP16 vs FP32 Parity + Aux Loss Distortion")
    run_streamed([
        sys.executable, "-u", "scripts/diagnose_collapse.py",
        "--ckpt-best", str(ckpt_best),
        "--ckpt-last", str(ckpt_last),
        "--expected-best-epoch", "70",
        "--expected-last-epoch", "150",
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

elif RUN_MODE == "eval_v3_old_vs_new":
    assert ckpt_v3_old is not None, "Chưa tìm thấy v3_best.pth (V3 cũ 100ep)!"
    assert ckpt_best is not None, "Chưa tìm thấy best_metric_model.pth (V3 mới Ep 70)!"
    
    eval_dir = Path("/kaggle/working/eval_v3_old_vs_new_fold0")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n>>> [4/5] Running Paired Evaluation: V3 Cũ (100ep) vs V3 Mới (150ep, Ep 70)...")
    run_streamed([
        sys.executable, "-u", "scripts/evaluate_paired_fold.py",
        "--ckpt-v3", str(ckpt_v3_old),
        "--ckpt-v4", str(ckpt_best),
        "--data-dir", DATA_DIR,
        "--fold", str(FOLD),
        "--strict-isles250",
        "--sw-roi-size", "96", "96", "64",
        "--allow-param-mismatch",
        "--postprocess-modes", "raw", "0.008ml",
        "--output-dir", str(eval_dir),
    ])
    
    report_file = eval_dir / f"paired_fold{FOLD}_comparison_report.md"
    if report_file.exists():
        print("\n" + "=" * 76)
        print("BÁO CÁO ĐỐI ĐẦU 4 METRICS ISLES-2022:")
        print("=" * 76)
        print(report_file.read_text(encoding="utf-8"))
```
