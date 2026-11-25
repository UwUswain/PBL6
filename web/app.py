"""FastAPI backend cho web demo PBL6.

Chạy từ thư mục gốc dự án:  uvicorn web.app:app --port 8000
"""

from functools import lru_cache
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Dict, Optional
import uuid
import zlib

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import nibabel as nib
import numpy as np

from src.dataset.parser import parse_isles_dataset
from src.metrics.volume import compute_lesion_volume
from web.inference import ROOT, checkpoint_paths, get_model_cfg, lesion_stats, load_config, predict, required_inputs
from web.regions import region_stats, safe_region_map

CONFIG = load_config()
RESULTS_DIR = Path(tempfile.mkdtemp(prefix="pbl6_web_"))
MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # FLAIR ISLES chưa nén ~140 MB
MODALITIES = ("dwi", "adc", "flair")

app = FastAPI(title="PBL6 Stroke Lesion Segmentation Demo")


@lru_cache(maxsize=1)
def _cases() -> Dict[str, dict]:
    data_dir = ROOT / CONFIG["data_dir"]
    if not data_dir.exists():
        return {}
    return {c["case_id"]: c for c in parse_isles_dataset(data_dir, require_mask=False)}


def _save_upload(upload: UploadFile, dest_dir: Path, name: str) -> Path:
    filename = (upload.filename or "").lower()
    suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii" if filename.endswith(".nii") else None
    if suffix is None:
        raise HTTPException(400, f"{name.upper()}: only .nii or .nii.gz files are accepted.")
    dest = dest_dir / f"{name}{suffix}"  # không dùng tên file của người dùng làm đường dẫn
    written = 0
    with open(dest, "wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{name.upper()}: file is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
            out.write(chunk)
    return dest


@app.get("/api/models")
def list_models():
    return {
        "default": CONFIG["default_model"],
        "models": [
            {
                "name": name,
                "label": m.get("label", name),
                "requires": required_inputs(m),
                "arch": m["arch"],
                "modalities": m["modalities"],
                "n_checkpoints": len(checkpoint_paths(m)),
                "available": bool(checkpoint_paths(m)) and all(p.exists() for p in checkpoint_paths(m)),
                "metrics": m.get("metrics"),
                "note": m.get("note"),  # {vi, en}: lưu ý hiện dưới ô chọn model
            }
            for name, m in CONFIG["models"].items()
        ],
    }


def _nifti_name(kind: str, path: str) -> str:
    # NiiVue nhận dạng định dạng qua đuôi trong URL
    return kind + (".nii.gz" if path.lower().endswith(".nii.gz") else ".nii")


def _case_files(c: dict) -> Dict[str, str]:
    # parser lấy DWI thế chỗ khi thiếu FLAIR; ở đây coi như không có FLAIR để không chạy model với đầu vào sai
    return {k: c[k] for k in MODALITIES if c[k] and not (k == "flair" and c[k] == c["dwi"])}


@app.get("/api/cases")
def list_cases():
    featured = CONFIG.get("featured_cases") or {}
    order = {cid: i for i, cid in enumerate(featured)}
    return [
        {
            "featured_order": order.get(cid),  # thứ tự trong config.yaml
            "case_id": cid,
            "featured": featured.get(cid),  # {vi, en} mô tả ngắn, hoặc None
            "files": {k: f"/api/cases/{cid}/{_nifti_name(k, p)}" for k, p in _case_files(c).items()},
            "ground_truth": f"/api/cases/{cid}/{_nifti_name('mask', c['mask'])}" if c["mask"] else None,
        }
        for cid, c in _cases().items()
    ]


@app.get("/api/cases/{case_id}/{filename}")
def get_case_file(case_id: str, filename: str):
    case = _cases().get(case_id)
    kind = filename.split(".", 1)[0]
    if case is None or kind not in (*MODALITIES, "mask") or not case[kind] or filename != _nifti_name(kind, case[kind]):
        raise HTTPException(404, "Case or file not found.")
    return FileResponse(case[kind], media_type="application/octet-stream", filename=f"{case_id}_{filename}")


def _compare_with_ground_truth(mask_img: nib.Nifti1Image, gt_path: str) -> Optional[dict]:
    """Dice và thể tích nhãn chuyên gia của ca mẫu, để demo thấy model đúng/sai tới đâu."""
    gt_img = nib.load(gt_path)
    gt = np.asarray(gt_img.dataobj) > 0
    pred = np.asarray(mask_img.dataobj) > 0
    if gt.shape != pred.shape:
        return None
    total = int(gt.sum() + pred.sum())
    return {
        "dice": 1.0 if total == 0 else float(2 * (gt & pred).sum() / total),
        "volume_ml": compute_lesion_volume(gt, gt_img.header.get_zooms()[:3]),
    }


@app.post("/api/predict")
def run_predict(
    case_id: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    dwi: Optional[UploadFile] = File(None),
    adc: Optional[UploadFile] = File(None),
    flair: Optional[UploadFile] = File(None),
):
    try:
        model_cfg = get_model_cfg(CONFIG, model)
    except ValueError as e:
        raise HTTPException(400, str(e))

    job_id = uuid.uuid4().hex
    job_dir = RESULTS_DIR / job_id
    upload_dir = job_dir / "uploads"
    upload_dir.mkdir(parents=True)
    try:
        if case_id:
            case = _cases().get(case_id)
            if case is None:
                raise HTTPException(404, f"Unknown sample case '{case_id}'.")
            paths = {m: Path(p) for m, p in _case_files(case).items()}
        else:
            uploads = {"dwi": dwi, "adc": adc, "flair": flair}
            paths = {m: _save_upload(u, upload_dir, m) for m, u in uploads.items() if u is not None}

        start = time.time()
        try:
            mask_img, stats = predict(paths, model_cfg)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as e:  # model hỏng/thiếu: báo rõ lên giao diện
            raise HTTPException(500, f"{type(e).__name__}: {e}")
        nib.save(mask_img, str(job_dir / "predicted_mask.nii.gz"))
        gt_path = case["mask"] if case_id and case["mask"] else None
        ground_truth = _compare_with_ground_truth(mask_img, gt_path) if gt_path else None
        atlas = safe_region_map(paths["dwi"])
        if atlas:  # giữ bản đồ vùng để tính lại khi bác sĩ sửa mask
            nib.save(nib.Nifti1Image(atlas[0], mask_img.affine), str(job_dir / "regions.nii.gz"))
        # Đủ thông tin để tính lại chỉ số khi bác sĩ sửa mask, không cần giữ ảnh gốc của người dùng
        (job_dir / "meta.json").write_text(json.dumps({
            "affine": mask_img.affine.tolist(),
            "shape": list(mask_img.shape),
            "zooms": stats["voxel_spacing_mm"],
            "midline_x": stats["midline_x"],
            "gt_path": gt_path,
        }))
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)  # ảnh y khoa: không giữ lại sau khi xử lý

    return {
        "job_id": job_id,
        "model": model or CONFIG["default_model"],
        "seconds": round(time.time() - start, 2),
        "ground_truth": ground_truth,
        "mask_url": f"/api/results/{job_id}/predicted_mask.nii.gz",
        "atlas_fit": atlas[1] if atlas else None,
        "regions": region_stats(mask_img.dataobj, atlas[0], stats["voxel_spacing_mm"]) if atlas else None,
        **stats,
    }


MASK_FILES = ("predicted_mask.nii.gz", "edited_mask.nii.gz")


def _job_dir(job_id: str) -> Path:
    job_dir = RESULTS_DIR / job_id
    if not re.fullmatch(r"[0-9a-f]{32}", job_id) or not (job_dir / "meta.json").is_file():
        raise HTTPException(404, "Result not found.")
    return job_dir


@app.get("/api/results/{job_id}/{name}")
def get_mask(job_id: str, name: str):
    path = _job_dir(job_id) / name
    if name not in MASK_FILES or not path.is_file():
        raise HTTPException(404, "Result not found.")
    return FileResponse(path, media_type="application/gzip", filename=name)


def _read_nifti_upload(upload: UploadFile, max_bytes: int) -> nib.Nifti1Image:
    raw = upload.file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(413, "Edited mask is too large.")
    if raw[:2] == bytes((0x1F, 0x8B)):
        inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
        raw = inflater.decompress(raw, max_bytes)  # giới hạn kích thước sau giải nén
        if inflater.unconsumed_tail:
            raise HTTPException(413, "Edited mask is too large.")
    try:
        return nib.Nifti1Image.from_bytes(raw)
    except Exception:
        raise HTTPException(400, "Edited mask is not a valid NIfTI file.")


@app.post("/api/results/{job_id}/edited")
def save_edited_mask(job_id: str, mask: UploadFile = File(...)):
    """Nhận mask bác sĩ đã sửa trên viewer, tính lại chỉ số bằng cùng hàm với mask của AI."""
    job_dir = _job_dir(job_id)
    meta = json.loads((job_dir / "meta.json").read_text())
    affine = np.array(meta["affine"])
    img = _read_nifti_upload(mask, int(np.prod(meta["shape"])) * 8 + 2**20)
    data = np.asarray(img.dataobj)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if list(data.shape) != meta["shape"] or not np.allclose(img.affine, affine, atol=1e-2):
        raise HTTPException(400, "Edited mask does not match the original image grid.")

    edited = nib.Nifti1Image((data > 0).astype(np.uint8), affine)
    nib.save(edited, str(job_dir / "edited_mask.nii.gz"))
    regions_path = job_dir / "regions.nii.gz"
    return {
        **lesion_stats(edited.dataobj, affine, meta["zooms"], meta["midline_x"]),
        "regions": region_stats(edited.dataobj, nib.load(str(regions_path)).dataobj, meta["zooms"]) if regions_path.is_file() else None,
        "ground_truth": _compare_with_ground_truth(edited, meta["gt_path"]) if meta["gt_path"] else None,
        "mask_url": f"/api/results/{job_id}/edited_mask.nii.gz",
    }


# Đăng ký sau cùng để không che các route /api
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
