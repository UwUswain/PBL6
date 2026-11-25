"""Suy luận một ca cho web demo: file NIfTI -> mask trong không gian DWI gốc + chỉ số."""

from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
import monai.transforms as mt
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
import yaml

from src.dataset.transforms import GenerateEnhancedDWId, _is_positive
from src.metrics.volume import compute_lesion_volume
from src.models.builder import get_model
from src.utils.postprocess import remove_small_lesion_islands

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GPU_LOCK = threading.Lock()  # 8 GB VRAM: chạy từng ca một
_MODEL_CACHE: Dict[tuple, List[torch.nn.Module]] = {}

# Phải khớp các bước 6-9 của get_val_transforms (bỏ mask) để model thấy đúng dữ liệu như lúc train.
_PRE = mt.Compose(
    [
        mt.Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        mt.CropForegroundd(keys="image", source_key="image", select_fn=_is_positive, margin=4),
        mt.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)
_INVERT = mt.Invertd(keys="prob", transform=_PRE, orig_keys="image", nearest_interp=False)


def load_config(path: Path = ROOT / "web" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def required_inputs(model_cfg: dict) -> List[str]:
    """File người dùng phải cung cấp (edwi được sinh ra từ dwi)."""
    return sorted({"dwi"} | {m for m in model_cfg["modalities"] if m != "edwi"})


def checkpoint_paths(model_cfg: dict) -> List[Path]:
    """Đường dẫn checkpoint; mục có dấu * được mở rộng theo glob (vd. 15 model của NVAUTO)."""
    paths = []
    for ckpt in model_cfg["checkpoints"]:
        if "*" in ckpt:
            paths += sorted(ROOT.glob(ckpt), key=lambda p: (len(p.name), p.name))  # model2 trước model10
        else:
            paths.append(Path(ckpt) if Path(ckpt).is_absolute() else ROOT / ckpt)
    return paths


def _load_models(model_cfg: dict) -> List[torch.nn.Module]:
    key = (model_cfg["arch"], tuple(model_cfg["checkpoints"]))
    if key not in _MODEL_CACHE:
        paths = checkpoint_paths(model_cfg)
        if not paths:
            raise FileNotFoundError(f"No checkpoints found for: {model_cfg['checkpoints']}")
        nets = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            if model_cfg["arch"] == "torchscript":  # model ngoài đóng gói sẵn kiến trúc (vd. NVAUTO)
                nets.append(torch.jit.load(str(path), map_location=DEVICE).eval())
                continue
            net = get_model(
                model_name=model_cfg["arch"],
                in_channels=len(model_cfg["modalities"]),
                out_channels=1,
                **(model_cfg.get("kwargs") or {}),
            )
            # weights_only=False: checkpoint của nhóm tự train, chứa cả optimizer state
            state = torch.load(path, map_location="cpu", weights_only=False)
            net.load_state_dict(state.get("model_state_dict", state))
            nets.append(net.to(DEVICE).eval())
        _MODEL_CACHE[key] = nets
    return _MODEL_CACHE[key]


def _load_volume(path: Path, name: str) -> MetaTensor:
    try:
        img = mt.LoadImage(image_only=True, ensure_channel_first=True)(str(path))
    except Exception as e:
        raise ValueError(f"Cannot read {name.upper()} as NIfTI: {e}") from e
    if img.ndim != 4 or img.shape[0] != 1:
        raise ValueError(f"{name.upper()} must be a single 3D volume, got shape {tuple(img.shape)}.")
    return img


def midline_x(brain: np.ndarray, affine: np.ndarray) -> float:
    """Tọa độ x (mm) của mặt phẳng giữa não, lấy bằng trung bình các voxel não (xấp xỉ, giả định đầu không nghiêng nhiều)."""
    return float((np.argwhere(brain) @ affine[0, :3]).mean() + affine[0, 3])


def lesion_stats(mask: np.ndarray, affine: np.ndarray, zooms: Tuple[float, ...], mid_x: float) -> dict:
    """Chỉ số của một mask nhị phân; dùng chung cho mask của AI và mask bác sĩ đã sửa."""
    mask = np.asarray(mask) > 0
    labels, lesion_count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    voxel_ml = float(np.prod(zooms)) / 1000.0
    focus_mm = None  # tâm ổ lớn nhất, để viewer nhảy tới
    left = right = 0.0
    side = "none"
    if lesion_count:
        largest = int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
        center = np.argwhere(labels == largest).mean(axis=0)
        focus_mm = [float(v) for v in affine[:3, :3] @ center + affine[:3, 3]]
        lesion_x = np.argwhere(mask) @ affine[0, :3] + affine[0, 3]
        left = float((lesion_x < mid_x).sum() * voxel_ml)  # NIfTI world: +x là bên phải bệnh nhân
        right = float((lesion_x >= mid_x).sum() * voxel_ml)
        share = left / (left + right)
        side = "left" if share >= 0.9 else "right" if share <= 0.1 else "both"
    return {
        "volume_ml": compute_lesion_volume(mask, zooms),
        "lesion_count": int(lesion_count),
        "left_ml": left,
        "right_ml": right,
        "hemisphere": side,
        "focus_mm": focus_mm,
    }


def predict(paths: Dict[str, Path], model_cfg: dict) -> Tuple[nib.Nifti1Image, dict]:
    """Chạy model trên một ca.

    Args:
        paths: {"dwi": ..., "adc": ..., "flair": ...} — chỉ cần các kênh model yêu cầu.
        model_cfg: một mục trong `models` của config.yaml.

    Returns:
        (mask NIfTI uint8 cùng lưới với DWI gốc, dict chỉ số).

    Raises:
        ValueError: đầu vào thiếu hoặc không đọc được (lỗi của người dùng).
    """
    missing = [m for m in required_inputs(model_cfg) if not paths.get(m)]
    if missing:
        raise ValueError(f"This model needs: {', '.join(m.upper() for m in missing)}.")
    if model_cfg["arch"] == "external":
        return _predict_external(paths, model_cfg)

    dwi = _load_volume(paths["dwi"], "dwi")
    if float(dwi.max()) <= 0:
        raise ValueError("DWI volume is empty (no positive voxels).")

    channels = []
    for mod in model_cfg["modalities"]:
        if mod == "dwi":
            img = dwi
        elif mod == "edwi":
            img = GenerateEnhancedDWId(keys=["dwi"])({"dwi": dwi})["edwi"]
        else:
            img = _load_volume(paths[mod], mod)
            # FLAIR của ISLES khác lưới với DWI; đưa mọi kênh về lưới DWI trước khi ghép
            if img.shape != dwi.shape or not np.allclose(img.affine, dwi.affine, atol=1e-3):
                img = mt.ResampleToMatch(mode="bilinear")(img, dwi)
        channels.append(torch.as_tensor(img).float())
    image = MetaTensor(torch.cat(channels, dim=0), meta=dwi.meta)

    data = _PRE({"image": image})
    x = data["image"].as_tensor().unsqueeze(0).to(DEVICE)
    roi = tuple(model_cfg["roi_size"])

    with _GPU_LOCK:
        nets = _load_models(model_cfg)
        prob = torch.zeros(x.shape[2:], dtype=torch.float32, device=DEVICE)
        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            for net in nets:
                logits = sliding_window_inference(x, roi, 1, net, overlap=float(model_cfg.get("overlap", 0.5)), mode="gaussian")
                if model_cfg.get("output") == "softmax":  # nền + tổn thương, lấy xác suất lớp 1
                    prob += torch.softmax(logits[0].float(), dim=0)[1]
                else:
                    prob += torch.sigmoid(logits[0, 0].float())
        prob = (prob / len(nets)).cpu()

    # Đảo Spacing + Crop để xác suất về đúng lưới DWI gốc, rồi mới lấy ngưỡng
    data["prob"] = MetaTensor(
        prob.unsqueeze(0), meta=data["image"].meta, applied_operations=data["image"].applied_operations
    )
    prob_orig = torch.as_tensor(_INVERT(data)["prob"])[0].numpy()

    return _finish((prob_orig >= float(model_cfg.get("threshold", 0.5))).astype(np.uint8), paths["dwi"], model_cfg)


def _predict_external(paths: Dict[str, Path], model_cfg: dict) -> Tuple[nib.Nifti1Image, dict]:
    """Model chạy ở tiến trình/môi trường riêng (vd. SEALS, Factorizer của DeepISLES): lệnh ghi mask ra {out}."""
    with tempfile.TemporaryDirectory(prefix="pbl6_ext_") as tmp:
        out = Path(tmp) / "mask.nii.gz"
        values = {m: str(p) for m, p in paths.items() if p} | {"out": str(out)}
        cmd = [part.format(**values) for part in model_cfg["command"]]
        exe = ROOT / cmd[0]
        if exe.is_file():  # đường dẫn python của venv viết tương đối theo thư mục dự án
            cmd[0] = str(exe)
        with _GPU_LOCK:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=int(model_cfg.get("timeout_s", 1800)))
        if r.returncode or not out.is_file():
            raise RuntimeError(f"External model failed ({r.returncode}): {(r.stderr or r.stdout)[-800:]}")
        mask = (np.asarray(nib.load(str(out)).dataobj) > 0).astype(np.uint8)
    return _finish(mask, paths["dwi"], model_cfg)


def _finish(mask: np.ndarray, dwi_path: Path, model_cfg: dict) -> Tuple[nib.Nifti1Image, dict]:
    """Mask nhị phân trên lưới DWI gốc -> NIfTI + chỉ số; dùng chung cho mọi loại model."""
    ref = nib.load(str(dwi_path))
    if mask.shape != ref.shape[:3]:
        raise RuntimeError(f"Mask shape {mask.shape} != DWI shape {ref.shape[:3]}")
    zooms = tuple(float(z) for z in ref.header.get_zooms()[:3])
    if float(model_cfg.get("min_lesion_ml", 0)) > 0:
        mask = remove_small_lesion_islands(mask, min_size_ml=float(model_cfg["min_lesion_ml"]), voxel_spacing=zooms)

    mid_x = midline_x(np.asarray(ref.dataobj) > 0, ref.affine)
    stats = {
        **lesion_stats(mask, ref.affine, zooms, mid_x),
        "midline_x": mid_x,
        "voxel_spacing_mm": zooms,
        "shape": list(mask.shape),
        "device": DEVICE.type,
    }
    return nib.Nifti1Image(mask, ref.affine), stats


def get_model_cfg(config: dict, name: Optional[str] = None) -> dict:
    name = name or config["default_model"]
    if name not in config["models"]:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(config['models'])}.")
    return config["models"][name]
