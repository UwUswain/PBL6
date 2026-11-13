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
# 1. CHỌN CHẾ ĐỘ CHẠY & CHỐT COMMIT_SHA + DEPENDENCIES DÙNG CHUNG
# ============================================================================
RUN_MODE = "smoke"  # "smoke" | "v3_full" | "v4_full" | "eval_full"

REPO_URL = "https://github.com/UwUswain/PBL6.git"
BRANCH_NAME = "long_exp"
FOLD = 0

# COMMIT_SHA cố định mới nhất trên origin/long_exp:
COMMIT_SHA = "c002afa500304737042d47acbe43fe5b6f4a1207"
# Giữ {} khi chạy "smoke" lần đầu; sau smoke lấy khối PINNED_ENV in ở cuối log dán vào đây:
PINNED_ENV = {}

# Điền 2 biến dưới đây CHỈ KHI RUN_MODE == "eval_full" (lấy từ Output v3_full & v4_full):
CKPT_V3_PATH = ""  # vd: "/kaggle/input/v3-fold0-ckpt/best_metric_model.pth"
CKPT_V4_PATH = ""  # vd: "/kaggle/input/v4-fold0-ckpt/best_metric_model.pth"

assert RUN_MODE in {"smoke", "v3_full", "v4_full", "eval_full"}, f"Invalid RUN_MODE: {RUN_MODE}"
clean_sha = COMMIT_SHA.strip()
if clean_sha:
    assert len(clean_sha) == 40 and all(ch in "0123456789abcdefABCDEF" for ch in clean_sha), (
        f"COMMIT_SHA must be a 40-char hex git SHA, got: {COMMIT_SHA!r}"
    )

if RUN_MODE in {"v3_full", "v4_full", "eval_full"}:
    assert len(clean_sha) == 40, (
        f"RUN_MODE='{RUN_MODE}' requires fixed 40-char COMMIT_SHA locked from smoke output!"
    )
    assert isinstance(PINNED_ENV, dict) and "packages" in PINNED_ENV and "python_major_minor" in PINNED_ENV, (
        f"RUN_MODE='{RUN_MODE}' requires PINNED_ENV locked from smoke output!"
    )

print(f"Selected RUN_MODE = {RUN_MODE} | FOLD = {FOLD} | BRANCH = {BRANCH_NAME} | PINNED_SHA = {clean_sha or '<HEAD of branch>'}")

# ============================================================================
# 2. CÀI / KHỚP DEPENDENCIES, CHECKOUT COMMIT_SHA & XÁC MINH DATASET 250 CA
# ============================================================================
def installed_ver(dist_name: str):
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None

BASE_PIP_PACKAGES = ("monai", "nibabel", "PyYAML", "scipy", "pandas", "scikit-learn")
PINNABLE_PIP_PACKAGES = ("monai", "nibabel", "numpy", "scipy", "pandas", "scikit-learn", "PyYAML")

print("\n>>> [1/4] Checking GPU & synchronizing shared dependencies...")
subprocess.run(["nvidia-smi"], check=False)

if not PINNED_ENV:
    missing = [pkg for pkg in BASE_PIP_PACKAGES if installed_ver(pkg) is None]
    if missing:
        print("Installing missing base packages:", missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    else:
        print("All base packages already installed.")
else:
    expected_pkgs = PINNED_ENV["packages"]
    py_actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if py_actual != PINNED_ENV["python_major_minor"]:
        raise RuntimeError(
            f"Python version drift: expected {PINNED_ENV['python_major_minor']}, got {py_actual}"
        )
    import torch
    if installed_ver("torch") != expected_pkgs.get("torch") or getattr(torch.version, "cuda", None) != PINNED_ENV.get("torch_cuda"):
        raise RuntimeError(
            f"PyTorch/CUDA drift on Kaggle image: expected torch={expected_pkgs.get('torch')} (cu={PINNED_ENV.get('torch_cuda')}), "
            f"got torch={installed_ver('torch')} (cu={getattr(torch.version, 'cuda', None)})"
        )
    mismatches = [
        f"{pkg}=={expected_pkgs[pkg]}"
        for pkg in PINNABLE_PIP_PACKAGES
        if pkg in expected_pkgs and expected_pkgs[pkg] is not None and installed_ver(pkg) != expected_pkgs[pkg]
    ]
    if mismatches:
        print("Pinning package versions to match smoke lock:", mismatches)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *mismatches])
    importlib.invalidate_caches()
    import torch as torch_recheck
    if installed_ver("torch") != expected_pkgs.get("torch") or getattr(torch_recheck.version, "cuda", None) != PINNED_ENV.get("torch_cuda"):
        raise RuntimeError("Pinned pip install altered torch/CUDA version! Aborting.")
    still_bad = [
        f"{pkg}: expected {expected_pkgs[pkg]}, got {installed_ver(pkg)}"
        for pkg in PINNABLE_PIP_PACKAGES
        if pkg in expected_pkgs and installed_ver(pkg) != expected_pkgs[pkg]
    ]
    if still_bad:
        raise RuntimeError("Failed to enforce PINNED_ENV versions: " + "; ".join(still_bad))
    print("PINNED_ENV VERIFIED OK across all tracked packages.")

work_dir = Path("/kaggle/working")
repo_dir = work_dir / "PBL6"

print(f"\n>>> [2/4] Syncing {REPO_URL} (branch={BRANCH_NAME}, commit={clean_sha or 'HEAD'})...")
if not (repo_dir / ".git").exists():
    if repo_dir.exists():
        subprocess.check_call(["rm", "-rf", str(repo_dir)])
    subprocess.check_call(["git", "clone", "-b", BRANCH_NAME, REPO_URL, str(repo_dir)])
else:
    subprocess.check_call(["git", "-C", str(repo_dir), "fetch", "--all", "--tags"])

if clean_sha:
    subprocess.check_call(["git", "-C", str(repo_dir), "checkout", "--detach", clean_sha])
    subprocess.check_call(["git", "-C", str(repo_dir), "reset", "--hard", clean_sha])
else:
    subprocess.check_call(["git", "-C", str(repo_dir), "checkout", BRANCH_NAME])
    subprocess.check_call(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{BRANCH_NAME}"])

os.chdir(str(repo_dir))
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
dirty_out = subprocess.check_output(["git", "status", "--short"], text=True).strip()
if clean_sha and commit_sha.lower() != clean_sha.lower():
    raise RuntimeError(f"Checked out SHA {commit_sha} != expected COMMIT_SHA {clean_sha}")
if dirty_out:
    raise RuntimeError(f"Repository working tree is dirty! Clean state required. Output:\n{dirty_out}")
print(f"ACTIVE COMMIT_SHA: {commit_sha} | dirty=False")

print("\n>>> [3/4] Locating & strictly verifying 250-case ISLES-2022 dataset in /kaggle/input...")
from src.dataset.parser import parse_isles_dataset, verify_isles_dataset
from src.utils.manifest import get_runtime_environment_info

def _find_verified_250_root(search_root: Path):
    cands = [p for p in search_root.rglob("derivatives") if p.is_dir()]
    errs = []
    for deriv_dir in sorted({c.parent for c in cands}):
        try:
            parsed = parse_isles_dataset(deriv_dir, require_mask=True)
            summary = verify_isles_dataset(
                cases=parsed,
                expected_num_cases=250,
                require_exact_isles250_ids=True,
                required_modalities=("dwi", "adc"),
                require_mask=True,
            )
            return deriv_dir, summary, errs
        except Exception as exc:
            errs.append(f"{deriv_dir}: {exc}")
    return None, None, errs

input_root = Path("/kaggle/input")
verified_data_dir, summary, inspection_report = _find_verified_250_root(input_root)

if verified_data_dir is None:
    sip_archives = sorted(list(input_root.rglob("*.sip")) + list(input_root.rglob("*.zip")))
    if sip_archives:
        free_tmp = shutil.disk_usage("/tmp").free
        extract_base = (
            Path("/tmp/isles2022_extracted")
            if free_tmp >= 4 * (1024 ** 3)
            else Path("/kaggle/working/_cache_isles2022")
        )
        if extract_base.exists():
            verified_data_dir, summary, cached_errs = _find_verified_250_root(extract_base)
            inspection_report.extend(cached_errs)
        if verified_data_dir is None:
            sip_path = sip_archives[0]
            print(f"Found archived dataset: {sip_path} ({sip_path.stat().st_size / (1024**3):.2f} GB)")
            print(f"Extracting to {extract_base} (disk free={shutil.disk_usage(extract_base.parent).free / (1024**3):.2f} GB)...")
            if extract_base.exists():
                shutil.rmtree(extract_base)
            extract_base.mkdir(parents=True, exist_ok=True)
            t_ext = time.time()
            with zipfile.ZipFile(sip_path, "r") as zf:
                zf.extractall(extract_base)
            print(f"Unpacked archive in {time.time() - t_ext:.1f}s")
            verified_data_dir, summary, ext_errs = _find_verified_250_root(extract_base)
            inspection_report.extend(ext_errs)

if verified_data_dir is None:
    raise RuntimeError(
        "No valid 250-case ISLES-2022 dataset found under /kaggle/input! Details:\n"
        + ("\n".join(inspection_report) if inspection_report else "No derivatives/ directories or .sip/.zip archives found.")
    )

print(
    f"VERIFIED 250-CASE DATASET: {verified_data_dir} "
    f"({summary['first_id']}..{summary['last_id']}, ids_sha256={summary['ids_sha256']})"
)

DATA_DIR = str(verified_data_dir)
ACTIVE_COMMIT_SHA = commit_sha
CURRENT_ENV_INFO = get_runtime_environment_info()
print("Active runtime packages:", json.dumps(CURRENT_ENV_INFO["packages"]))

print("\n>>> [4/4] Running quick unit sanity tests (losses/models, manifest, metrics)...")
subprocess.check_call(
    [sys.executable, "-m", "unittest", "tests.test_models_and_losses", "tests.test_manifest", "tests.test_metrics", "-v"]
)

# ============================================================================
# 3. THỰC THI THEO RUN_MODE, XUẤT LOCK ENV & NÉN ZIP LƯU TRỮ
# ============================================================================
def run_streamed(cmd):
    print("\n$ " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()
    elapsed = time.time() - t0
    print(f"\n[Finished in {elapsed/60.0:.2f} min | exit_code={rc}]")
    if rc != 0:
        raise RuntimeError(f"Command failed with exit_code={rc}: {' '.join(cmd)}")
    return elapsed

def zip_paths(zip_path: Path, paths):
    base_dir = Path("/kaggle/working")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in paths:
            p = Path(item)
            if p.is_file():
                zf.write(p, arcname=p.relative_to(base_dir) if p.is_relative_to(base_dir) else p.name)
            elif p.is_dir():
                for sub in sorted(p.rglob("*")):
                    if sub.is_file():
                        zf.write(sub, arcname=sub.relative_to(base_dir) if sub.is_relative_to(base_dir) else sub.name)
    print(f"Archived artifacts to: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

if RUN_MODE == "smoke":
    smoke_save = Path("/kaggle/working/smoke_runs")
    v3_save = smoke_save / "v3"
    v4_save = smoke_save / "v4"
    v3_save.mkdir(parents=True, exist_ok=True)
    v4_save.mkdir(parents=True, exist_ok=True)

    print("\n=== SMOKE 1/3: V3 Baseline (Pos/Neg) real-batch smoke train ===")
    run_streamed([
        sys.executable, "-u", "scripts/train.py",
        "--config", "configs/paired_v3_posneg_250_fold0.yaml",
        "--data-dir", DATA_DIR,
        "--fold", str(FOLD),
        "--save-dir", str(v3_save),
        "--strict-isles250",
        "--dry-run",
        "--dry-run-steps", "4",
        "--dry-run-val-cases", "2",
    ])

    print("\n=== SMOKE 2/3: V4-Fixed (Component-Aware) real-batch smoke train ===")
    run_streamed([
        sys.executable, "-u", "scripts/train.py",
        "--config", "configs/paired_v4_comp_250_fold0.yaml",
        "--data-dir", DATA_DIR,
        "--fold", str(FOLD),
        "--save-dir", str(v4_save),
        "--strict-isles250",
        "--dry-run",
        "--dry-run-steps", "4",
        "--dry-run-val-cases", "2",
    ])

    ckpt_v3 = sorted(v3_save.glob("dry_run_*/dry_run_checkpoint.pth"))[-1]
    ckpt_v4 = sorted(v4_save.glob("dry_run_*/dry_run_checkpoint.pth"))[-1]
    smoke_eval_dir = Path("/kaggle/working/smoke_eval_out")

    print("\n=== SMOKE 3/3: Paired Evaluation (manifest + env parity + raw & 0.008ml + HD95) ===")
    run_streamed([
        sys.executable, "-u", "scripts/evaluate_paired_fold.py",
        "--ckpt-v3", str(ckpt_v3),
        "--ckpt-v4", str(ckpt_v4),
        "--data-dir", DATA_DIR,
        "--fold", str(FOLD),
        "--strict-isles250",
        "--allow-dry-run",
        "--max-cases", "2",
        "--postprocess-modes", "raw", "0.008ml",
        "--output-dir", str(smoke_eval_dir),
    ])
    print((smoke_eval_dir / f"paired_fold{FOLD}_comparison_report.md").read_text(encoding="utf-8"))

    lock_payload = {
        "commit_sha": ACTIVE_COMMIT_SHA,
        "pinned_env": {
            "python_major_minor": CURRENT_ENV_INFO["python_major_minor"],
            "torch_cuda": CURRENT_ENV_INFO["torch_cuda"],
            "packages": CURRENT_ENV_INFO["packages"],
        },
    }
    lock_file = Path("/kaggle/working/kaggle_env_lock.json")
    lock_file.write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print("LOCK CHO FULL RUN (COPY DAN VAO CELL 1 CHO v3_full, v4_full, eval_full):")
    print("=" * 76)
    print(f'COMMIT_SHA = "{ACTIVE_COMMIT_SHA}"')
    print("PINNED_ENV = " + repr(lock_payload["pinned_env"]))
    print("=" * 76)
    zip_paths(Path("/kaggle/working/paired_fold0_smoke_artifacts.zip"), [smoke_save, smoke_eval_dir, lock_file])

elif RUN_MODE == "v3_full":
    out_dir = Path("/kaggle/working/runs_v3_fold0")
    out_dir.mkdir(parents=True, exist_ok=True)
    elapsed = run_streamed([
        sys.executable, "-u", "scripts/train.py",
        "--config", "configs/paired_v3_posneg_250_fold0.yaml",
        "--data-dir", DATA_DIR,
        "--fold", str(FOLD),
        "--save-dir", str(out_dir),
        "--strict-isles250",
    ])
    print(f"V3 Fold {FOLD} Runtime: {elapsed/3600.0:.2f} hours ({elapsed/60.0:.1f} min) | COMMIT={ACTIVE_COMMIT_SHA}")
    zip_paths(Path(f"/kaggle/working/v3_posneg_fold{FOLD}_artifacts.zip"), [out_dir])

elif RUN_MODE == "v4_full":
    out_dir = Path("/kaggle/working/runs_v4_fold0")
    out_dir.mkdir(parents=True, exist_ok=True)
    elapsed = run_streamed([
        sys.executable, "-u", "scripts/train.py",
        "--config", "configs/paired_v4_comp_250_fold0.yaml",
        "--data-dir", DATA_DIR,
        "--fold", str(FOLD),
        "--save-dir", str(out_dir),
        "--strict-isles250",
    ])
    print(f"V4-Fixed Fold {FOLD} Runtime: {elapsed/3600.0:.2f} hours ({elapsed/60.0:.1f} min) | COMMIT={ACTIVE_COMMIT_SHA}")
    zip_paths(Path(f"/kaggle/working/v4_fixed_fold{FOLD}_artifacts.zip"), [out_dir])

elif RUN_MODE == "eval_full":
    assert CKPT_V3_PATH and CKPT_V4_PATH, "Set CKPT_V3_PATH and CKPT_V4_PATH in Cell 1!"
    eval_dir = Path(f"/kaggle/working/paired_fold{FOLD}_eval_results")
    run_streamed([
        sys.executable, "-u", "scripts/evaluate_paired_fold.py",
        "--ckpt-v3", CKPT_V3_PATH,
        "--ckpt-v4", CKPT_V4_PATH,
        "--data-dir", DATA_DIR,
        "--fold", str(FOLD),
        "--strict-isles250",
        "--postprocess-modes", "raw", "0.008ml",
        "--output-dir", str(eval_dir),
    ])
    print((eval_dir / f"paired_fold{FOLD}_comparison_report.md").read_text(encoding="utf-8"))
    zip_paths(Path(f"/kaggle/working/paired_fold{FOLD}_eval_results.zip"), [eval_dir])