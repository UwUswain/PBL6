"""Chạy một thuật toán DeepISLES (SEALS hoặc Factorizer) trên một ca, trong venv riêng external/env_nnunet.

Web gọi script này qua `arch: external` (xem web/config.yaml). Code của DeepISLES giữ nguyên (external/deepisles_src);
script chỉ xếp file vào đúng thư mục họ đọc, chạy các bước như src/SEALS/nnunet_launcher.sh, rồi chép mask ra.

    python deepisles.py seals --dwi DWI --adc ADC --out mask.nii.gz
    python deepisles.py factorizer --dwi DWI --adc ADC --flair FLAIR --out mask.nii.gz
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import SimpleITK as sitk

EXTERNAL = Path(__file__).resolve().parents[2] / "external"
SRC = EXTERNAL / "deepisles_src" / "src"
WEIGHTS = EXTERNAL / "weights"


def run(cmd, cwd, env):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"Step failed: {' '.join(map(str, cmd))}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")


def seals(case_dir: Path, env: dict) -> Path:
    cwd = SRC / "SEALS"
    for d in ("data/nnUNet_raw_data_base", "test_result", "test_result_recover", "test_ensemble"):
        shutil.rmtree(cwd / d, ignore_errors=True)  # kết quả trung gian của ca trước
    env = {**env, "nnUNet_raw_data_base": "data/nnUNet_raw_data_base", "nnUNet_preprocessed": "data/nnUNet_preprocessed",
           "RESULTS_FOLDER": str(WEIGHTS / "SEALS" / "nnUNet_trained_models"), "nnUNet_n_proc_DA": "2"}
    py = sys.executable
    run([py, "nnunet/dataset_conversion/Task500_Ischemic_Stroke_Test.py", "--raw_data_dir", case_dir], cwd, env)
    images = "data/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/"
    for f in range(5):
        run([py, "-m", "nnunet.inference.predict_simple", "-i", images, "-o", f"test_result/preliminary_phase/fold{f}",
             "-t", "12", "-tr", "nnUNetTrainerV2_DDP", "-m", "3d_fullres", "-f", f, "-z",
             "--overwrite_existing", "--disable_postprocessing"], cwd, env)
        run([py, "recover_softmax.py", "-i", "test_result", "-o", f"test_result_recover/preliminary_phase/fold{f}",
             "-m", "preliminary_phase", "-f", f"fold{f}", "--raw_data_dir", case_dir], cwd, env)
    run([py, "-m", "ensemble_predictions", "-f", *[f"test_result_recover/preliminary_phase/fold{f}" for f in range(5)],
         "-o", "test_ensemble/", "--npz"], cwd, env)
    run([py, "threshold_redirect.py", "-i", "test_ensemble/", "-o", case_dir / "output" / "seals", "--raw_data_dir", case_dir], cwd, env)
    return case_dir / "output" / "seals" / "lesion_msk.nii.gz"


def factorizer(case_dir: Path, env: dict) -> Path:
    env = {**env, "PYTHONPATH": str(SRC / "FACTORIZER" / "model" / "factorizer")}  # bản factorizer DeepISLES đi kèm
    run([sys.executable, "process.py", "--input_path", case_dir, "--model_path", WEIGHTS], SRC / "FACTORIZER", env)
    return case_dir / "output" / "factorizer" / "lesion_msk.nii.gz"


TEAMS = {"seals": seals, "factorizer": factorizer}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("team", choices=TEAMS)
    p.add_argument("--dwi", required=True)
    p.add_argument("--adc", required=True)
    p.add_argument("--flair")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.team == "factorizer" and not a.flair:
        sys.exit("Factorizer needs FLAIR.")

    with tempfile.TemporaryDirectory(prefix="deepisles_") as tmp:
        case_dir = Path(tmp)
        # Cấu trúc DeepISLES đọc: <case>/dwi/dwi.nii.gz, <case>/adc/adc.nii.gz, <case>/flair/flair.nii.gz
        for name, path in (("dwi", a.dwi), ("adc", a.adc), ("flair", a.flair)):
            if path:
                (case_dir / name).mkdir()
                sitk.WriteImage(sitk.ReadImage(path), str(case_dir / name / f"{name}.nii.gz"))
        out = TEAMS[a.team](case_dir, dict(os.environ))
        if not out.is_file():
            sys.exit(f"{a.team} produced no mask at {out}")
        shutil.copyfile(out, a.out)


if __name__ == "__main__":
    main()
