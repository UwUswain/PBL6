# HƯỚNG DẪN CHẨN ĐOÁN SỤP ĐỔ VÀ ĐÁNH GIÁ ĐỐI ĐẦU V3 (CŨ VS MỚI) TRÊN KAGGLE

> **Dành cho Long:** Copy toàn bộ cell bên dưới dán đè vào Kaggle Notebook `ISLES-2022 DATASET ANALYSIS`.
> - Chọn `RUN_MODE = "diagnose"` để chẩn đoán nguyên nhân sụp đổ Epoch 86–150 (kiểm tra phân bố xác suất, logits, quét ngưỡng và aux loss).
> - Chọn `RUN_MODE = "eval_v3_old_vs_new"` để chấm điểm công bằng giữa V3 cũ (Dice 0.6779) và V3 mới (Epoch 70, Dice 0.6816) trên 50 ca Fold 0.

---

## CÁCH ĐƯA CHECKPOINT 150-EPOCH VÀO KAGGLE (CHỌN 1 TRONG 2 CÁCH)

- **Cách 1 (Nhanh nhất - Không cần upload lại):**
  1. Trong notebook Kaggle, nhìn sang panel **Input** bên phải.
  2. Bấm nút **`+ Add Input`** -> chọn tab **`Your Work`** (hoặc `Notebook Output Files`).
  3. Chọn notebook **`ISLES-2022 DATASET ANALYSIS`** (Version 23 vừa chạy xong).
  4. Kaggle sẽ tự động mount toàn bộ artifact (chứa `best_metric_model.pth` và `last_checkpoint.pth`) vào thư mục `/kaggle/input/isles-2022-dataset-analysis/`.

- **Cách 2 (Upload thủ công nếu thích):**
  1. Bấm **`+ Add Input`** -> **`Upload`** -> Tạo New Dataset tên `isles-2022-v3-150ep-fold0`.
  2. Kéo thả 2 file `best_metric_model.pth` và `last_checkpoint.pth` từ `reserve/Fold0_150e/runs_paired_v3_patch_96_96_80_150ep_fold0_fold0/dual_segresnet_fold0_20261113_091807/` lên.

---

## TOÀN BỘ KHỐI CODE 1 CELL DUY NHẤT TRÊN KAGGLE

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
# 1. CẤU HÌNH CHẾ ĐỘ CHẠY & COMMIT_SHA
# ============================================================================
# Chọn 1 trong 2 chế độ:
# - "diagnose": Chẩn đoán tại sao Ep 86-150 sụp đổ, đo logits/probs, sweep threshold, test aux loss (5 ca, ~3 phút)
# - "eval_v3_old_vs_new": Chấm điểm đối đầu V3 cũ (100ep) vs V3 mới (Ep 70) trên toàn bộ 50 ca Fold 0 (~12 phút)
RUN_MODE = "diagnose"  # "diagnose" | "eval_v3_old_vs_new"

REPO_URL = "https://github.com/UwUswain/PBL6.git"
BRANCH_NAME = "long_exp"
FOLD = 0

# SHA mới nhất đã bao gồm công cụ chẩn đoán và sửa lỗi eval mismatch
COMMIT_SHA = "fe5dc00ee113b7885b46c576a3af863b89addf6c"

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
# 2. ĐỒNG BỘ MÔI TRƯỜNG & CLONE / CHECKOUT REPO
# ============================================================================
subprocess.run(["nvidia-smi"], check=False)

repo_dir = Path("/kaggle/working/PBL6")
if not repo_dir.exists():
    print(f"\nCloning {REPO_URL} (branch: {BRANCH_NAME})...")
    subprocess.check_call(["git", "clone", "--branch", BRANCH_NAME, REPO_URL, str(repo_dir)])
else:
    print(f"\nRepository exists at {repo_dir}. Fetching latest...")
    subprocess.check_call(["git", "-C", str(repo_dir), "fetch", "--all"])

print(f"Checking out pinned COMMIT_SHA: {COMMIT_SHA}...")
subprocess.check_call(["git", "-C", str(repo_dir), "checkout", COMMIT_SHA])

current_sha = subprocess.check_output(
    ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
).strip()
assert current_sha == COMMIT_SHA, f"Git SHA mismatch! Expected {COMMIT_SHA}, got {current_sha}"
print(f"Git checkout verified at: {current_sha}")

os.chdir(str(repo_dir))
sys.path.insert(0, str(repo_dir))

# ============================================================================
# 3. GIẢI NÉN VÀ XÁC THỰC DATASET ISLES-2022 (250 CASES)
# ============================================================================
raw_data_dir = Path("/kaggle/working/data/raw/ISLES-2022")
raw_data_dir.mkdir(parents=True, exist_ok=True)

sip_file = Path("/kaggle/input/datasets/ivesunderthiscobal/isles-2022-250cases-sip/ISLES-2022.sip")
zip_fallback = Path("/kaggle/input/datasets/ivesunderthiscobal/isles-2022-250cases-sip/ISLES-2022.zip")

existing_cases = list(raw_data_dir.glob("sub-strokecase*"))
if len(existing_cases) < 250:
    archive_to_extract = sip_file if sip_file.exists() else (zip_fallback if zip_fallback.exists() else None)
    if archive_to_extract is None:
        matches = list(Path("/kaggle/input").rglob("ISLES-2022.sip")) + list(Path("/kaggle/input").rglob("ISLES-2022.zip"))
        if matches:
            archive_to_extract = matches[0]
            
    assert archive_to_extract is not None, "Could not locate ISLES-2022 archive (.sip or .zip)!"
    print(f"Extracting {archive_to_extract} to {raw_data_dir}...")
    with zipfile.ZipFile(archive_to_extract, "r") as zf:
        zf.extractall(raw_data_dir)

total_cases = len(list(raw_data_dir.glob("sub-strokecase*")))
print(f"ISLES-2022 dataset ready: {total_cases} cases verified.")
assert total_cases == 250, f"Expected 250 cases, found {total_cases}"

DATA_DIR = str(raw_data_dir)

# ============================================================================
# 4. TỰ ĐỘNG TÌM ĐƯỜNG DẪN CHECKPOINTS
# ============================================================================
def find_ckpt(name_patterns: list) -> Path:
    search_roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    for root in search_roots:
        for p in root.rglob("*.pth"):
            for pat in name_patterns:
                if pat.lower() in p.name.lower() or pat.lower() in str(p).lower():
                    return p
    return None

# Tìm V3 Cũ
ckpt_v3_old = find_ckpt(["v3_best.pth", "dual_segresnet_fold0_20261111"])
# Tìm V3 Mới (Best Ep 70)
ckpt_v3_new_best = find_ckpt(["best_metric_model.pth", "dual_segresnet_fold0_20261113"])
# Tìm V3 Mới (Last Ep 150)
ckpt_v3_new_last = find_ckpt(["last_checkpoint.pth"])

print("\n--- Checkpoints Status ---")
print("V3 Cũ (100ep, 96x96x64)      :", ckpt_v3_old)
print("V3 Mới (150ep Best, Ep 70)   :", ckpt_v3_new_best)
print("V3 Mới (150ep Last, Ep 150)  :", ckpt_v3_new_last)

# ============================================================================
# 5. THỰC THI CHẾ ĐỘ
# ============================================================================
def run_streamed(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(p.stdout.readline, ""):
        print(line, end="", flush=True)
    p.stdout.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with return code {rc}")

if RUN_MODE == "diagnose":
    assert ckpt_v3_new_best is not None, "Missing best_metric_model.pth! Please add Version 23 output via '+ Add Input'."
    assert ckpt_v3_new_last is not None, "Missing last_checkpoint.pth! Please add Version 23 output via '+ Add Input'."
    
    diag_dir = Path("/kaggle/working/diagnostics_out")
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n>>> RUNNING COLLAPSE DIAGNOSTICS (5 cases, threshold sweep, probability dist, aux loss)...")
    run_streamed([
        sys.executable, "-u", "scripts/diagnose_collapse.py",
        "--ckpt-best", str(ckpt_v3_new_best),
        "--ckpt-last", str(ckpt_v3_new_last),
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
        print("BÁO CÁO CHẨN ĐOÁN CHI TIẾT:")
        print("=" * 76)
        print(report_file.read_text(encoding="utf-8"))

elif RUN_MODE == "eval_v3_old_vs_new":
    assert ckpt_v3_old is not None, "Missing v3_best.pth (old V3)!"
    assert ckpt_v3_new_best is not None, "Missing best_metric_model.pth (new V3 Ep 70)!"
    
    eval_dir = Path("/kaggle/working/eval_v3_old_vs_new_fold0")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n>>> RUNNING PAIRED EVALUATION: V3 CŨ (100ep) VS V3 MỚI (150ep, Ep 70)...")
    run_streamed([
        sys.executable, "-u", "scripts/evaluate_paired_fold.py",
        "--ckpt-v3", str(ckpt_v3_old),
        "--ckpt-v4", str(ckpt_v3_new_best),
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
        print("BÁO CÁO ĐỐI ĐẦU V3 CŨ VS V3 MỚI:")
        print("=" * 76)
        print(report_file.read_text(encoding="utf-8"))
```
