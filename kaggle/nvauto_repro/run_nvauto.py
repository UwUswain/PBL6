"""Tái hiện NVAUTO (bài 22, ISLES 2022) bằng Auto3DSeg SegResNet, chạy được trên Kaggle.

Hai lệnh:
    python run_nvauto.py train     # train các fold, tự dừng trước giới hạn thời gian, lần sau chạy tiếp
    python run_nvauto.py evaluate  # Dice từng ca ở độ phân giải gốc: trung bình + trung vị

Xem README.md để biết cách dùng trên Kaggle.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "segresnet"
CONFIG = TEMPLATE / "configs" / "hyper_parameters.yaml"
FOLDS = json.loads((HERE / "folds.json").read_text())

PAPER_CV_MEAN = 0.8086  # bài 22, Bảng 1: trung bình Dice 5-fold, 1 model mỗi fold
TEAM_CV_MEAN = 0.7235   # Dual-Encoder SegResNet V3, experiments/results/dual_segresnet_v3_5fold_summary.md


# ---------------------------------------------------------------- dữ liệu
def is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return p.is_file() and (n.endswith(".nii") or n.endswith(".nii.gz"))


def find_one(folder: Path, keyword: str) -> Path:
    # Bộ ISLES của nhóm có dạng ".../xxx_dwi.nii/<tên gốc>_dwi.nii" (thư mục mang đuôi .nii), nên chỉ nhận file thật
    hits = sorted(p for p in folder.rglob("*") if is_nifti(p) and keyword in p.name.lower())
    if not hits:
        raise FileNotFoundError(f"No '{keyword}' NIfTI file under {folder}")
    return hits[0]


def find_data_root(hint: str) -> Path:
    if hint != "auto":
        return Path(hint)
    for base in [Path("/kaggle/input"), HERE.parent.parent / "ISLES-2022"]:
        if not base.exists():
            continue
        for deriv in base.rglob("derivatives"):
            if deriv.is_dir() and any(deriv.parent.glob("sub-strokecase*")):
                return deriv.parent
    raise FileNotFoundError("ISLES-2022 not found; pass --data-root (folder containing sub-strokecase*/ and derivatives/)")


def build_datalist(data_root: Path, out: Path, smoke: bool) -> dict:
    training = []
    for case_dir in sorted(data_root.glob("sub-strokecase*")):
        cid = case_dir.name
        if cid not in FOLDS:
            raise KeyError(f"{cid} not in folds.json")
        training.append({
            "image": [str(find_one(case_dir, "adc")), str(find_one(case_dir, "dwi"))],  # ADC, DWI như process.py của NVAUTO
            "label": str(find_one(data_root / "derivatives" / cid, "msk")),
            "fold": FOLDS[cid],
            "case_id": cid,
        })
    if smoke:  # 3 ca mỗi fold; 2 ca cuối có tên file chung "adc_skull-stripped.nii" để kiểm tra dự đoán không ghi đè nhau
        training = [t for k in range(5) for xs in [[x for x in training if x["fold"] == k]] for t in xs[:1] + xs[-2:]]
    elif len(training) != 250:
        raise RuntimeError(f"Expected 250 ISLES cases, found {len(training)} under {data_root}")
    datalist = {"training": training}
    out.write_text(json.dumps(datalist, indent=1))
    return datalist


# ---------------------------------------------------------------- chạy tiến trình
def run_logged(cmd: list, log_path: Path, timeout_s: float) -> str:
    """Chạy lệnh, in log ra màn hình và ghi file; dừng cả cây tiến trình khi hết giờ. Trả 'ok' | 'failed' | 'timeout'."""
    import psutil

    print("$", " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env, cwd=str(HERE))

        def pump():
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            code = proc.wait(timeout=max(1.0, timeout_s))
        except subprocess.TimeoutExpired:
            print(f"\n>>> Time budget reached, stopping training (progress is saved every epoch).", flush=True)
            for child in psutil.Process(proc.pid).children(recursive=True):
                child.kill()
            proc.kill()
            proc.wait()
            reader.join(timeout=10)
            return "timeout"
        reader.join(timeout=10)
    return "ok" if code == 0 else "failed"


def sync_best_metric(model_dir: Path) -> None:
    # Template lưu model_final.pt kèm Dice của epoch cuối chứ không phải Dice tốt nhất; khi train tiếp
    # mà không sửa thì model.pt có thể bị ghi đè bằng model kém hơn.
    import torch

    best, final = model_dir / "model.pt", model_dir / "model_final.pt"
    if not (best.is_file() and final.is_file()):
        return
    best_metric = torch.load(best, map_location="cpu", weights_only=False).get("best_metric")
    ckpt = torch.load(final, map_location="cpu", weights_only=False)
    if best_metric is not None and ckpt.get("best_metric") != best_metric:
        ckpt["best_metric"] = best_metric
        torch.save(ckpt, final)


def copy_previous_output(src: Path, work: Path) -> None:
    for fold_src in sorted(src.rglob("fold[0-4]")):
        if not (fold_src / "model").is_dir():
            continue
        dst = work / fold_src.name
        if (dst / "model" / "model_final.pt").exists() or (dst / "DONE.json").exists():
            continue  # đã có bản mới hơn trong thư mục làm việc
        print(f"Restoring {fold_src} -> {dst}")
        shutil.copytree(fold_src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("prediction_*", "mlruns"))


def base_overrides(args, fold: int, fold_dir: Path, datalist_path: Path, data_root: Path) -> list:
    ov = [
        f"--bundle_root={fold_dir}",
        f"--fold={fold}",
        f"--data_list_file_path={datalist_path}",
        # Datalist paths are absolute so loading ignores this; saving uses it to keep one sub-folder per case,
        # otherwise the 52 cases named "dwi/adc_skull-stripped.nii" overwrite each other's prediction.
        f"--data_file_base_dir={data_root}",
        f"--num_epochs={args.epochs}",
        f"--num_epochs_per_validation={min(10, args.epochs)}",  # chạy ngắn vẫn có ít nhất 1 lần kiểm định
    ]
    if args.roi:
        ov.append("--roi_size=[" + ",".join(str(r) for r in args.roi) + "]")
    if args.workers is not None:
        ov.append(f"--num_workers={args.workers}")
    return ov


# ---------------------------------------------------------------- lệnh train
def cmd_train(args) -> None:
    deadline = time.time() + args.hours * 3600
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if args.resume_from:
        copy_previous_output(Path(args.resume_from), work)
    data_root = find_data_root(args.data_root)
    datalist_path = work / "datalist.json"
    build_datalist(data_root, datalist_path, args.smoke)
    print(f"Data: {data_root} | folds {args.folds} | epochs {args.epochs} | budget {args.hours} h", flush=True)

    for fold in args.folds:
        fold_dir = work / f"fold{fold}"
        model_dir = fold_dir / "model"
        if (fold_dir / "DONE.json").exists():
            print(f"fold {fold}: already finished, skipping")
            continue
        remaining = deadline - time.time() - args.margin_min * 60
        if remaining < args.min_session_min * 60:
            print(f"fold {fold}: not enough time left in this session ({remaining / 60:.0f} min), stopping")
            break
        model_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(TEMPLATE / "scripts" / "train.py"), "run", f"--config_file={CONFIG}"]
        cmd += base_overrides(args, fold, fold_dir, datalist_path, data_root)
        if (model_dir / "model_final.pt").exists():
            sync_best_metric(model_dir)
            cmd += ["--finetune#enabled=True", f"--finetune#ckpt_name={model_dir / 'model_final.pt'}", "--continue=True"]
            print(f"fold {fold}: resuming from {model_dir / 'model_final.pt'}")
        status = run_logged(cmd, fold_dir / "train_stdout.log", remaining)
        if status == "ok":
            progress = (model_dir / "progress.yaml").read_text(encoding="utf-8") if (model_dir / "progress.yaml").exists() else ""
            (fold_dir / "DONE.json").write_text(json.dumps({"epochs": args.epochs, "finished": time.ctime(), "progress_tail": progress[-600:]}))
            print(f"fold {fold}: finished")
        elif status == "timeout":
            print(f"fold {fold}: paused; run again with --resume-from to continue")
            break
        else:
            raise SystemExit(f"fold {fold}: training failed, see {fold_dir / 'train_stdout.log'}")
    print_status(work, args.epochs)


def print_status(work: Path, epochs: int) -> None:
    print("\n=== Status ===")
    for fold in range(5):
        fold_dir = work / f"fold{fold}"
        hist = fold_dir / "model" / "accuracy_history.csv"
        last = best = None
        if hist.exists():
            # mỗi lần train tiếp, template ghi thêm một dòng tiêu đề
            rows = [r.split("\t") for r in hist.read_text().splitlines() if r[:1].isdigit()]
            if rows:
                last = int(rows[-1][0])
                best = max(float(r[1]) for r in rows)
        state = "done" if (fold_dir / "DONE.json").exists() else ("in progress" if last is not None else "not started")
        detail = f"last epoch {last}/{epochs}, best val Dice {best:.4f}" if last is not None else ""
        print(f"fold {fold}: {state} {detail}")


# ---------------------------------------------------------------- lệnh evaluate
def dice(a, b) -> float:
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


def cmd_evaluate(args) -> None:
    import nibabel as nib
    import numpy as np

    work = Path(args.work_dir)
    if args.resume_from:
        copy_previous_output(Path(args.resume_from), work)
    datalist_path = work / "datalist.json"
    data_root = find_data_root(args.data_root)
    datalist = build_datalist(data_root, datalist_path, args.smoke)
    rows = []
    for fold in args.folds:
        fold_dir = work / f"fold{fold}"
        ckpt = next((fold_dir / "model" / n for n in ("model.pt", "model_final.pt") if (fold_dir / "model" / n).exists()), None)
        if ckpt is None:
            print(f"fold {fold}: no checkpoint, skipping")
            continue
        if not (fold_dir / "DONE.json").exists():
            print(f"fold {fold}: WARNING training not finished, evaluating {ckpt.name} anyway")
        pred_dir = fold_dir / "prediction_validation"
        shutil.rmtree(pred_dir, ignore_errors=True)
        cmd = [sys.executable, str(TEMPLATE / "scripts" / "validate.py"), "run", f"--config_file={CONFIG}"]
        cmd += base_overrides(args, fold, fold_dir, datalist_path, data_root)
        cmd += ["--validate#save_mask=True", f"--validate#output_path={pred_dir}", f"--validate#ckpt_name={ckpt}"]
        if run_logged(cmd, fold_dir / "validate_stdout.log", 6 * 3600) != "ok":
            raise SystemExit(f"fold {fold}: validation failed, see {fold_dir / 'validate_stdout.log'}")
        for case in [c for c in datalist["training"] if c["fold"] == fold]:
            stem = Path(case["image"][0]).name.split(".nii")[0]
            hits = [h for h in pred_dir.rglob(stem + ".nii*") if h.is_file() and case["case_id"] in h.parts]
            if len(hits) != 1:
                raise FileNotFoundError(f"{case['case_id']}: expected 1 prediction named {stem}.nii*, found {hits}")
            gt_img = nib.load(case["label"])
            gt = np.asarray(gt_img.dataobj) > 0
            pred = np.squeeze(np.asarray(nib.load(str(hits[0])).dataobj)) > 0
            if pred.shape != gt.shape:
                raise ValueError(f"{case['case_id']}: prediction {pred.shape} vs label {gt.shape}")
            voxel_ml = float(np.prod(gt_img.header.get_zooms()[:3])) / 1000
            rows.append({"fold": fold, "case_id": case["case_id"], "dice": dice(pred, gt),
                         "gt_ml": gt.sum() * voxel_ml, "pred_ml": pred.sum() * voxel_ml})
    if not rows:
        raise SystemExit("No folds evaluated.")
    write_report(work, rows)


def write_report(work: Path, rows: list) -> None:
    with open(work / "cv_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def summary(ds):
        return f"{statistics.mean(ds):.4f} | {statistics.median(ds):.4f} | {len(ds)}"

    lines = ["# NVAUTO reproduction: cross-validation Dice (original resolution)", "",
             "| Fold | Mean | Median | Cases |", "|---|---|---|---|"]
    for fold in sorted({r["fold"] for r in rows}):
        lines.append(f"| {fold} | " + summary([r["dice"] for r in rows if r["fold"] == fold]) + " |")
    all_d = [r["dice"] for r in rows]
    lines.append("| **All** | " + summary(all_d) + " |")
    for name, lo, hi in [("Small (< 5 mL)", 0, 5), ("Medium (5-30 mL)", 5, 30), ("Large (> 30 mL)", 30, 1e9)]:
        ds = [r["dice"] for r in rows if lo <= r["gt_ml"] < hi]
        if ds:
            lines.append(f"| {name} | " + summary(ds) + " |")
    lines += ["", f"Reference: paper 22 5-fold CV mean = {PAPER_CV_MEAN}; team Dual-Encoder V3 5-fold CV mean = {TEAM_CV_MEAN}.",
              "Folds are the team's split (src/dataset/parser.get_kfold_splits, seed 42), so fold k is comparable with the team's fold k."]
    text = "\n".join(lines) + "\n"
    (work / "cv_summary.md").write_text(text, encoding="utf-8")
    print(text)


# ---------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["train", "evaluate", "status"])
    p.add_argument("--data-root", default="auto", help="folder containing sub-strokecase*/ and derivatives/")
    p.add_argument("--work-dir", default="/kaggle/working/nvauto")
    p.add_argument("--resume-from", default=None, help="output of a previous Kaggle run (read-only input) to continue from")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=1000, help="paper: 1000")
    p.add_argument("--hours", type=float, default=11.3, help="stop cleanly before this many hours (Kaggle limit: 12)")
    p.add_argument("--margin-min", type=float, default=10, help="minutes kept free at the end of the session")
    p.add_argument("--min-session-min", type=float, default=15, help="do not start a fold with less time than this")
    p.add_argument("--roi", type=int, nargs=3, default=None, help="override patch size (paper: 192 192 128)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="2 cases per fold, to test the pipeline")
    args = p.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        print_status(Path(args.work_dir), args.epochs)


if __name__ == "__main__":
    main()
