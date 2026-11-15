"""Diagnostic engine for investigating late-stage training degradation and collapse.

Strict Medical AI & MLOps Standards:
- Explicit checkpoint and manifest verification (Epoch 70 Peak vs Epoch 150 Degraded).
- Cross-validation with ISLES-2022 dataset parser and ground truth split hashes.
- Environment and GPU hardware sanity audit against pinned specifications.
- NaN / Inf numerical gating prior to sigmoid and threshold evaluation.
- Dual-precision comparative inference: AMP FP16 vs Full Precision FP32.
- Deep supervision target downsampling lesion preservation analysis (Paper 22 / NVAUTO).
- Purely empirical observational reporting: NO pre-baked causal claims without ablation proof.
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import monai.data as md
from monai.inferers import sliding_window_inference
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

from src.dataset.parser import parse_isles_dataset, get_kfold_splits, compute_split_fingerprint
from src.dataset.transforms import get_val_transforms
from src.metrics.dice import compute_dice
from src.models.builder import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("diagnose_collapse")


def fmt(val: Any, fmt_str: str = ".4f", na_str: str = "N/A") -> str:
    """Safely formats numerical values without invalid format specifier runtime errors."""
    if val is None:
        return na_str
    if isinstance(val, (int, float, np.floating, np.integer)):
        if not np.isfinite(val):
            return na_str
        try:
            return f"{val:{fmt_str}}"
        except (ValueError, TypeError):
            return str(val)
    return str(val)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose model training collapse with strict manifest verification and AMP/FP32 parity."
    )
    parser.add_argument("--ckpt-best", type=str, required=True, help="Explicit path to best peak checkpoint (Epoch 70).")
    parser.add_argument("--ckpt-last", type=str, required=True, help="Explicit path to last checkpoint (Epoch 150).")
    parser.add_argument("--expected-best-epoch", type=int, default=70, help="Expected epoch for peak checkpoint.")
    parser.add_argument("--expected-last-epoch", type=int, default=150, help="Expected epoch for last checkpoint.")
    parser.add_argument("--data-dir", type=str, default="data/raw/ISLES-2022", help="ISLES-2022 dataset root.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate (default: 0).")
    parser.add_argument("--n-splits", type=int, default=5, help="Total folds (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="K-Fold split seed (default: 42).")
    parser.add_argument("--sw-roi-size", nargs=3, type=int, default=[96, 96, 64], help="Sliding window ROI.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Sliding window overlap.")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of validation cases to diagnose in detail.")
    parser.add_argument("--output-dir", type=str, default="experiments/diagnostics", help="Output directory for reports.")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference (e.g. cuda or cpu).")
    parser.add_argument("--skip-fp32-comparison", action="store_true", help="Skip FP32 comparison pass (runs AMP only).")
    parser.add_argument("--expected-cuda", type=str, default=None, help="Expected CUDA version (e.g. 13.0). Mismatch causes fatal error.")
    return parser.parse_args()


def audit_environment(device: torch.device, expected_cuda: Optional[str] = None) -> Dict[str, Any]:
    """Audits local Python, PyTorch, and CUDA runtime environment."""
    torch_cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else None
    env_info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch_cuda_version,
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "total_memory_mb": (
            float(torch.cuda.get_device_properties(device).total_memory / (1024 ** 2))
            if device.type == "cuda"
            else 0.0
        ),
    }
    logger.info("Environment Audit: Python %s | PyTorch %s (CUDA: %s) | Device: %s (%s, %.1f MB VRAM)",
                env_info["python_version"], env_info["torch_version"], env_info["torch_cuda_version"],
                env_info["device_type"], env_info["device_name"], env_info["total_memory_mb"])

    if expected_cuda is not None and torch_cuda_version != expected_cuda:
        raise RuntimeError(
            f"CUDA version mismatch! Expected {expected_cuda}, got {torch_cuda_version}"
        )

    return env_info


def verify_checkpoint_integrity(
    ckpt_path: Path,
    expected_epoch: Optional[int],
    expected_fold: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Strictly loads and verifies checkpoint manifest and epoch headers."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")

    logger.info("Verifying checkpoint integrity: %s...", ckpt_path)
    ckpt_data = torch.load(ckpt_path, map_location="cpu")

    actual_epoch = ckpt_data.get("epoch")
    if expected_epoch is not None and actual_epoch != expected_epoch:
        raise ValueError(
            f"Epoch verification FAILED for {ckpt_path.name}: expected {expected_epoch}, got {actual_epoch}"
        )

    manifest = ckpt_data.get("run_manifest")
    if not manifest or not isinstance(manifest, dict):
        raise ValueError(
            f"Checkpoint {ckpt_path.name} is missing 'run_manifest'! Strict MLOps requires embedded manifest."
        )

    cfg = manifest.get("config", {}).get("resolved", {})
    split_info = manifest.get("split", {})
    dataset_info = manifest.get("dataset", {})

    actual_fold = split_info.get("fold")
    if actual_fold is not None and actual_fold != expected_fold:
        raise ValueError(
            f"Fold verification FAILED for {ckpt_path.name}: expected Fold {expected_fold}, got Fold {actual_fold}"
        )

    model_name = cfg.get("model_name")
    if model_name != "dual_segresnet":
        raise ValueError(
            f"Architecture mismatch in {ckpt_path.name}: expected 'dual_segresnet', got '{model_name}'"
        )

    roi_size = cfg.get("roi_size")
    if roi_size != [96, 96, 80]:
        raise ValueError(
            f"Train ROI size mismatch in {ckpt_path.name}: expected [96, 96, 80], got {roi_size}"
        )

    val_roi_size = cfg.get("val_roi_size")
    if val_roi_size != [96, 96, 64]:
        raise ValueError(
            f"Validation ROI size mismatch in {ckpt_path.name}: expected [96, 96, 64], got {val_roi_size}"
        )

    # Read current_dice, best_dice, or val_dice
    val_score = ckpt_data.get("best_dice")
    if val_score is None:
        val_score = ckpt_data.get("current_dice")
    if val_score is None:
        val_score = ckpt_data.get("val_dice")

    cfg_info = manifest.get("config", {})
    code_info = manifest.get("code_version", {})

    manifest_meta = {
        "epoch": actual_epoch,
        "stated_score": val_score,
        "manifest_present": True,
        "model_name": model_name,
        "roi_size": roi_size,
        "val_roi_size": val_roi_size,
        "loss_type": cfg.get("loss_type"),
        "deep_supervision": cfg.get("deep_supervision"),
        "fold": actual_fold,
        "train_sha256": split_info.get("train_sha256"),
        "val_sha256": split_info.get("val_sha256"),
        "dataset_ids_sha256": dataset_info.get("dataset_ids_sha256"),
        "total_cases": dataset_info.get("total_cases"),
        "train_ids": split_info.get("train_ids", []),
        "val_ids": split_info.get("val_ids", []),
        "git_commit": code_info.get("git_commit"),
        "config_file": cfg_info.get("config_file"),
        "config_sha256": cfg_info.get("config_sha256"),
        "run_created_at": manifest.get("created_at"),
    }

    logger.info("  -> Manifest verified: Epoch=%s | Stored Score=%s | Loss=%s | SW-ROI=%s | Val SHA=%s",
                actual_epoch, fmt(val_score, ".4f"),
                manifest_meta["loss_type"], manifest_meta["val_roi_size"],
                str(manifest_meta["val_sha256"])[:12])

    return ckpt_data, manifest_meta


def cross_check_split_with_dataset(
    all_cases: List[Dict[str, Any]],
    splits: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    fold: int,
    meta_best: Dict[str, Any],
    meta_last: Dict[str, Any],
) -> None:
    """Cross-checks parser-generated cases and splits with both checkpoint manifests,
    and strictly validates code/config/run parity between best and last checkpoints."""
    logger.info("Cross-checking dataset and split manifests with filesystem parser...")
    if len(all_cases) != 250:
        raise ValueError(f"Expected 250 cases in dataset, found {len(all_cases)}")

    train_cases, val_cases = splits[fold]
    fingerprint = compute_split_fingerprint(train_cases, val_cases)
    actual_train_sha256 = fingerprint["train_sha256"]
    actual_val_sha256 = fingerprint["val_sha256"]

    # Check match with Best Checkpoint
    if meta_best.get("val_sha256") != actual_val_sha256:
        raise ValueError(
            f"Val SHA256 mismatch in Best Checkpoint! Manifest={meta_best.get('val_sha256')}, Parser={actual_val_sha256}"
        )
    if meta_best.get("train_sha256") != actual_train_sha256:
        raise ValueError(
            f"Train SHA256 mismatch in Best Checkpoint! Manifest={meta_best.get('train_sha256')}, Parser={actual_train_sha256}"
        )

    # Check match with Last Checkpoint
    if meta_last.get("val_sha256") != actual_val_sha256:
        raise ValueError(
            f"Val SHA256 mismatch in Last Checkpoint! Manifest={meta_last.get('val_sha256')}, Parser={actual_val_sha256}"
        )
    if meta_last.get("train_sha256") != actual_train_sha256:
        raise ValueError(
            f"Train SHA256 mismatch in Last Checkpoint! Manifest={meta_last.get('train_sha256')}, Parser={actual_train_sha256}"
        )

    # Check run identity & config/code parity between Best and Last Checkpoints
    b_created = meta_best.get("run_created_at")
    l_created = meta_last.get("run_created_at")
    if b_created and l_created and b_created != l_created:
        raise ValueError(
            f"Run created_at mismatch between Best ({b_created}) and Last ({l_created})! Checkpoints are from different runs."
        )

    b_commit = meta_best.get("git_commit")
    l_commit = meta_last.get("git_commit")
    if b_commit and l_commit and b_commit != l_commit:
        raise ValueError(
            f"Git commit mismatch between Best ({b_commit}) and Last ({l_commit})!"
        )

    b_cfg_sha = meta_best.get("config_sha256")
    l_cfg_sha = meta_last.get("config_sha256")
    if b_cfg_sha and l_cfg_sha and b_cfg_sha != l_cfg_sha:
        raise ValueError(
            f"Config SHA256 mismatch between Best ({b_cfg_sha}) and Last ({l_cfg_sha})!"
        )

    # Check both checkpoints come from the exact same split and configuration
    if meta_best.get("train_sha256") != meta_last.get("train_sha256"):
        raise ValueError("Checkpoints come from DIFFERENT train splits!")
    if meta_best.get("val_sha256") != meta_last.get("val_sha256"):
        raise ValueError("Checkpoints come from DIFFERENT validation splits!")
    if meta_best.get("roi_size") != meta_last.get("roi_size"):
        raise ValueError("Checkpoints have mismatched training ROI sizes!")
    if meta_best.get("val_roi_size") != meta_last.get("val_roi_size"):
        raise ValueError("Checkpoints have mismatched validation ROI sizes!")
    if meta_best.get("model_name") != meta_last.get("model_name"):
        raise ValueError("Checkpoints have mismatched model architectures!")
    if meta_best.get("loss_type") != meta_last.get("loss_type"):
        raise ValueError("Checkpoints have mismatched loss functions!")
    if meta_best.get("deep_supervision") != meta_last.get("deep_supervision"):
        raise ValueError("Checkpoints have mismatched deep supervision settings!")

    logger.info("  -> Dataset & Split Parity Verified: 250 total cases (Train: %d, Val: %d)",
                len(train_cases), len(val_cases))
    logger.info("  -> Train SHA256: %s | Val SHA256: %s", actual_train_sha256, actual_val_sha256)
    logger.info("  -> Run Identity Verified: Run created_at=%s | Git commit=%s | Config SHA=%s",
                b_created, str(b_commit)[:12], str(b_cfg_sha)[:12])


def audit_weights(
    sd_best: Dict[str, torch.Tensor],
    sd_last: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Compares layer parameters between best and collapsed checkpoints for NaN/Inf and norm drift."""
    results = {
        "best_has_nan": False,
        "best_has_inf": False,
        "last_has_nan": False,
        "last_has_inf": False,
        "best_total_l2": 0.0,
        "last_total_l2": 0.0,
        "l2_distance": 0.0,
        "layer_drift": [],
    }

    best_sq_sum = 0.0
    last_sq_sum = 0.0
    diff_sq_sum = 0.0

    for k, w_b in sd_best.items():
        if k not in sd_last:
            continue
        w_l = sd_last[k]

        if torch.isnan(w_b).any().item():
            results["best_has_nan"] = True
        if torch.isinf(w_b).any().item():
            results["best_has_inf"] = True
        if torch.isnan(w_l).any().item():
            results["last_has_nan"] = True
        if torch.isinf(w_l).any().item():
            results["last_has_inf"] = True

        wb_float = w_b.float()
        wl_float = w_l.float()

        b_norm = torch.sum(wb_float ** 2).item()
        l_norm = torch.sum(wl_float ** 2).item()
        d_norm = torch.sum((wl_float - wb_float) ** 2).item()

        best_sq_sum += b_norm
        last_sq_sum += l_norm
        diff_sq_sum += d_norm

        rel_drift = np.sqrt(d_norm) / (np.sqrt(b_norm) + 1e-8)
        results["layer_drift"].append({
            "name": k,
            "best_norm": float(np.sqrt(b_norm)),
            "last_norm": float(np.sqrt(l_norm)),
            "diff_norm": float(np.sqrt(d_norm)),
            "rel_drift": float(rel_drift),
        })

    results["best_total_l2"] = float(np.sqrt(best_sq_sum))
    results["last_total_l2"] = float(np.sqrt(last_sq_sum))
    results["l2_distance"] = float(np.sqrt(diff_sq_sum))
    results["layer_drift"].sort(key=lambda x: x["rel_drift"], reverse=True)
    return results


def analyze_deep_supervision_distortion(
    mask_tensor: torch.Tensor,
) -> Dict[str, Any]:
    """Measures lesion voxel and connected component retention under whole-volume downsampling.

    Note: This is an illustrative spatial representation check on the whole volume;
    it is NOT an exact measurement of the training patch auxiliary loss during SGD.
    """
    arr = (mask_tensor[0, 0].cpu().numpy() > 0).astype(np.int32)
    labeled_orig, num_comp = ndi.label(arr, structure=ndi.generate_binary_structure(3, 3))
    orig_voxels = int(arr.sum())

    res = {
        "orig_voxels": orig_voxels,
        "orig_components": int(num_comp),
        "stage1_voxels": 0,
        "stage1_lost_components": 0,
        "stage0_voxels": 0,
        "stage0_lost_components": 0,
    }

    if num_comp == 0:
        return res

    h, w, d = arr.shape
    t_stage1 = F.interpolate(mask_tensor, size=(h // 2, w // 2, d // 2), mode="nearest")
    arr_stage1 = (t_stage1[0, 0].cpu().numpy() > 0).astype(np.int32)
    res["stage1_voxels"] = int(arr_stage1.sum())

    t_stage0 = F.interpolate(mask_tensor, size=(h // 4, w // 4, d // 4), mode="nearest")
    arr_stage0 = (t_stage0[0, 0].cpu().numpy() > 0).astype(np.int32)
    res["stage0_voxels"] = int(arr_stage0.sum())

    lost_stage1 = 0
    lost_stage0 = 0
    for comp_idx in range(1, num_comp + 1):
        comp_mask = (labeled_orig == comp_idx)
        comp_tensor = torch.from_numpy(comp_mask).float().unsqueeze(0).unsqueeze(0)
        c_s1 = F.interpolate(comp_tensor, size=(h // 2, w // 2, d // 2), mode="nearest")
        c_s0 = F.interpolate(comp_tensor, size=(h // 4, w // 4, d // 4), mode="nearest")
        if c_s1.sum().item() == 0:
            lost_stage1 += 1
        if c_s0.sum().item() == 0:
            lost_stage0 += 1

    res["stage1_lost_components"] = lost_stage1
    res["stage0_lost_components"] = lost_stage0
    return res


def run_sliding_window_pass(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    sw_roi_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
    use_amp: bool,
) -> Tuple[torch.Tensor, float]:
    """Executes a single sliding window inference pass with explicit AMP control."""
    if not torch.isfinite(image_tensor).all():
        raise ValueError("Input image_tensor contains non-finite values (NaN/Inf)!")

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            out = sliding_window_inference(
                inputs=image_tensor.to(device),
                roi_size=sw_roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
    elapsed = time.time() - t0
    return out.cpu().float(), elapsed


def compute_case_diagnostics(
    logits: Optional[torch.Tensor],
    gt_mask: torch.Tensor,
    thresholds: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
    status: str = "ok",
) -> Dict[str, Any]:
    """Computes logit/probability distributions with strict NaN/Inf pre-threshold gating."""
    stats = {
        "status": status,
        "has_nan": False,
        "has_inf": False,
        "nan_count": 0,
        "inf_count": 0,
        "logit_min": None,
        "logit_max": None,
        "logit_mean": None,
        "fg_logit_mean": None,
        "fg_logit_max": None,
        "bg_logit_mean": None,
        "prob_max": None,
        "prob_p50": None,
        "prob_p95": None,
        "prob_p99": None,
        "fg_prob_mean": None,
        "fg_prob_median": None,
        "fg_prob_max": None,
        "bg_prob_mean": None,
        "bg_prob_p99": None,
        "threshold_dices": {f"{th:.2f}": None for th in thresholds},
        "threshold_pred_voxels": {f"{th:.2f}": None for th in thresholds},
    }

    if logits is None or status != "ok":
        return stats

    if not torch.isfinite(gt_mask).all():
        stats["status"] = "error_non_finite_gt"
        return stats

    logits_np = logits[0, 0].numpy()
    gt = (gt_mask[0, 0].cpu().numpy() > 0).astype(np.int32)

    has_nan = bool(np.isnan(logits_np).any())
    has_inf = bool(np.isinf(logits_np).any())
    nan_count = int(np.isnan(logits_np).sum())
    inf_count = int(np.isinf(logits_np).sum())

    stats["has_nan"] = has_nan
    stats["has_inf"] = has_inf
    stats["nan_count"] = nan_count
    stats["inf_count"] = inf_count

    # Strict gating: Do NOT compute sigmoid or threshold metrics on NaNs/Infs
    if has_nan or has_inf:
        logger.error("NON-FINITE LOGITS ENCOUNTERED: nan_count=%d, inf_count=%d! Bypassing threshold calculation.",
                     nan_count, inf_count)
        stats["status"] = "error_non_finite_logits"
        return stats

    # Finite execution path
    probs = torch.sigmoid(logits)[0, 0].numpy()
    fg_mask = (gt == 1)
    bg_mask = (gt == 0)

    stats["logit_min"] = float(logits_np.min())
    stats["logit_max"] = float(logits_np.max())
    stats["logit_mean"] = float(logits_np.mean())

    stats["prob_max"] = float(probs.max())
    stats["prob_p50"] = float(np.percentile(probs, 50))
    stats["prob_p95"] = float(np.percentile(probs, 95))
    stats["prob_p99"] = float(np.percentile(probs, 99))

    # Foreground stats: retain None if fg_mask is empty
    if fg_mask.any():
        fg_probs = probs[fg_mask]
        fg_logits = logits_np[fg_mask]
        stats["fg_prob_mean"] = float(fg_probs.mean())
        stats["fg_prob_median"] = float(np.median(fg_probs))
        stats["fg_prob_max"] = float(fg_probs.max())
        stats["fg_logit_mean"] = float(fg_logits.mean())
        stats["fg_logit_max"] = float(fg_logits.max())

    # Background stats: retain None if bg_mask is empty
    if bg_mask.any():
        bg_probs = probs[bg_mask]
        bg_logits = logits_np[bg_mask]
        stats["bg_prob_mean"] = float(bg_probs.mean())
        stats["bg_prob_p99"] = float(np.percentile(bg_probs, 99))
        stats["bg_logit_mean"] = float(bg_logits.mean())

    for th in thresholds:
        pred_bin = (probs > th).astype(np.int32)
        dice = compute_dice(pred_bin, gt)
        stats["threshold_dices"][f"{th:.2f}"] = float(dice)
        stats["threshold_pred_voxels"][f"{th:.2f}"] = int(pred_bin.sum())

    return stats


def compare_amp_vs_fp32(
    logits_amp: Optional[torch.Tensor],
    logits_fp32: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Compares numerical outputs between AMP FP16 and pure FP32 with strict non-finite gating."""
    if logits_amp is None or logits_fp32 is None:
        return {
            "status": "not_run",
            "max_logit_abs_diff": None,
            "mean_logit_abs_diff": None,
            "max_prob_abs_diff": None,
            "mean_prob_abs_diff": None,
        }

    is_amp_finite = torch.isfinite(logits_amp).all().item()
    is_fp32_finite = torch.isfinite(logits_fp32).all().item()

    if not is_amp_finite or not is_fp32_finite:
        return {
            "status": "error_non_finite",
            "max_logit_abs_diff": None,
            "mean_logit_abs_diff": None,
            "max_prob_abs_diff": None,
            "mean_prob_abs_diff": None,
        }

    diff = torch.abs(logits_amp - logits_fp32)
    max_diff = float(torch.max(diff).item())
    mean_diff = float(torch.mean(diff).item())

    probs_amp = torch.sigmoid(logits_amp)
    probs_fp32 = torch.sigmoid(logits_fp32)
    prob_diff = torch.abs(probs_amp - probs_fp32)

    return {
        "status": "computed",
        "max_logit_abs_diff": max_diff,
        "mean_logit_abs_diff": mean_diff,
        "max_prob_abs_diff": float(torch.max(prob_diff).item()),
        "mean_prob_abs_diff": float(torch.mean(prob_diff).item()),
    }


def build_markdown_report(
    env_info: Dict[str, Any],
    meta_b: Dict[str, Any],
    meta_l: Dict[str, Any],
    weight_audit: Dict[str, Any],
    case_diagnostics: List[Dict[str, Any]],
    deep_sup_distortions: List[Dict[str, Any]],
    ckpt_best_path: Path,
    ckpt_last_path: Path,
    sw_roi: Tuple[int, int, int],
    overlap: float,
) -> str:
    """Builds clean, objective empirical observation markdown report without pre-baked causal claims."""
    lines = [
        "# BÁO CÁO ĐO ĐẠC THỰC NGHIỆM CHẨN ĐOÁN MÔ HÌNH V3 (EPOCH 70 VS EPOCH 150)",
        "",
        f"- **Thời điểm thực thi:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Môi trường:** Python `{env_info['python_version']}` | PyTorch `{env_info['torch_version']}` (CUDA: `{env_info.get('torch_cuda_version', 'N/A')}`) | Thiết bị: `{env_info['device_name']}` ({fmt(env_info.get('total_memory_mb'), '.1f')} MB VRAM)",
        f"- **Best Checkpoint:** Epoch `{meta_b['epoch']}` (Stated Score: `{fmt(meta_b.get('stated_score'), '.4f')}`) — `{ckpt_best_path.name}`",
        f"- **Last Checkpoint:** Epoch `{meta_l['epoch']}` (Stated Score: `{fmt(meta_l.get('stated_score'), '.4f')}`) — `{ckpt_last_path.name}`",
        f"- **Thiết lập suy luận:** Sliding Window ROI `{sw_roi}` | Overlap `{overlap}`",
        "",
        "---",
        "",
        "## 1. ĐỐI CHIẾU TRỌNG SỐ VÀ KIỂM TRA SỐ HỌC (NUMERICAL HEALTH)",
        "",
        "| Chỉ số đo đạc | Best Checkpoint (Ep 70) | Last Checkpoint (Ep 150) | Trạng thái kỹ thuật |",
        "|---|---|---|---|",
        f"| Chứa giá trị NaN | `{'CÓ' if weight_audit['best_has_nan'] else 'KHÔNG'}` | `{'CÓ' if weight_audit['last_has_nan'] else 'KHÔNG'}` | {'Phát hiện phân kỳ vô định trong trọng số' if (weight_audit['best_has_nan'] or weight_audit['last_has_nan']) else 'Không có giá trị NaN trong trọng số'} |",
        f"| Chứa giá trị Inf | `{'CÓ' if weight_audit['best_has_inf'] else 'KHÔNG'}` | `{'CÓ' if weight_audit['last_has_inf'] else 'KHÔNG'}` | {'Phát hiện tràn số thực trong trọng số' if (weight_audit['best_has_inf'] or weight_audit['last_has_inf']) else 'Không có giá trị Inf trong trọng số'} |",
        f"| Tổng chuẩn L2 weights | `{fmt(weight_audit['best_total_l2'], '.4f')}` | `{fmt(weight_audit['last_total_l2'], '.4f')}` | Chênh lệch `{fmt(((weight_audit['last_total_l2'] / (weight_audit['best_total_l2'] + 1e-8)) - 1) * 100, '.2f')}%` |",
        f"| Khoảng cách L2 (Best vs Last) | — | `{fmt(weight_audit['l2_distance'], '.4f')}` | Khoảng cách dịch chuyển tham số trong không gian trọng số |",
        "",
        "### Top 5 Layer có mức độ dịch chuyển (Relative Drift) lớn nhất:",
    ]
    for it in weight_audit["layer_drift"][:5]:
        lines.append(
            f"- **`{it['name']}`**: Relative Drift = `{fmt(it['rel_drift'], '.4f')}` "
            f"(Norm Best: `{fmt(it['best_norm'], '.4f')}` $\\rightarrow$ Norm Last: `{fmt(it['last_norm'], '.4f')}`)"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. ĐỐI CHIẾU ĐỘ CHÍNH XÁC: AMP FP16 VS FULL PRECISION FP32",
        "",
        "| Case ID | Best AMP vs FP32 Diff | Best FP32 Max Prob | Last AMP vs FP32 Diff | Last FP32 Max Prob | Last AMP Dice@0.5 | Last FP32 Dice@0.5 |",
        "|---|---|---|---|---|---|---|",
    ])
    for cd in case_diagnostics:
        b_pdiff = cd["best_prec_diff"]
        l_pdiff = cd["last_prec_diff"]

        b_diff_str = fmt(b_pdiff.get("max_logit_abs_diff"), ".2e") if b_pdiff.get("status") == "computed" else b_pdiff.get("status", "N/A")
        l_diff_str = fmt(l_pdiff.get("max_logit_abs_diff"), ".2e") if l_pdiff.get("status") == "computed" else l_pdiff.get("status", "N/A")

        b_fp32_p = fmt(cd["best_fp32"].get("prob_max"), ".4f")
        l_fp32_p = fmt(cd["last_fp32"].get("prob_max"), ".4f")

        l_d_amp = fmt(cd["last_amp"]["threshold_dices"].get("0.50"), ".4f")
        l_d_fp32 = fmt(cd["last_fp32"]["threshold_dices"].get("0.50"), ".4f")

        lines.append(
            f"| `{cd['case_id']}` | {b_diff_str} | {b_fp32_p} | {l_diff_str} | {l_fp32_p} | {l_d_amp} | {l_d_fp32} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. PHÂN BỐ XÁC SUẤT VÀ QUÉT ĐỘ NHẠY NGƯỠNG (THRESHOLD SWEEP - AMP)",
        "",
        "| Case ID | GT Voxels | Best Max Prob | Best Dice@0.50 | Last Max Prob | Last Dice@0.50 | Last Dice@0.20 | Last Dice@0.10 | Last Pred Voxels (th=0.5 / 0.1) |",
        "|---|---|---|---|---|---|---|---|---|",
    ])
    for cd in case_diagnostics:
        b = cd["best_amp"]
        l = cd["last_amp"]

        b_p_max = fmt(b.get("prob_max"), ".4f")
        b_d_050 = fmt(b["threshold_dices"].get("0.50"), ".4f")

        l_p_max = fmt(l.get("prob_max"), ".4f")
        l_d_050 = fmt(l["threshold_dices"].get("0.50"), ".4f")
        l_d_020 = fmt(l["threshold_dices"].get("0.20"), ".4f")
        l_d_010 = fmt(l["threshold_dices"].get("0.10"), ".4f")

        v_050 = fmt(l["threshold_pred_voxels"].get("0.50"), "d")
        v_010 = fmt(l["threshold_pred_voxels"].get("0.10"), "d")

        lines.append(
            f"| `{cd['case_id']}` | {cd['gt_voxels']} | {b_p_max} | **{b_d_050}** | {l_p_max} | "
            f"{l_d_050} | {l_d_020} | {l_d_010} | {v_050} / {v_010} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. TỶ LỆ BẢO TOÀN TỔN THƯƠNG KHI DOWNSAMPLE DEEP SUPERVISION (PAPER 22 / NVAUTO)",
        "",
        "> [!NOTE]",
        "> Phân tích downsample trên toàn bộ volume ở đây là phép đo minh họa hình thái học không gian (spatial representation);",
        "> đây **KHÔNG** phải là phép đo auxiliary loss trực tiếp trên các patch crop ngẫu nhiên trong quá trình huấn luyện SGD.",
        "",
        "| Case ID | GT Voxels | GT Components | Stage 1 (2x) Mất ổ | Stage 0 (4x) Mất ổ | Tỷ lệ ổ tổn thương bị xóa sổ ở Aux Head 0 |",
        "|---|---|---|---|---|---|",
    ])
    for dist in deep_sup_distortions:
        tot_c = max(dist["orig_components"], 1)
        lost_pct = (dist["stage0_lost_components"] / tot_c) * 100.0 if dist["orig_components"] > 0 else 0.0
        lines.append(
            f"| `{dist['case_id']}` | {dist['orig_voxels']} | {dist['orig_components']} | "
            f"{dist['stage1_lost_components']}/{dist['orig_components']} | "
            f"{dist['stage0_lost_components']}/{dist['orig_components']} | **{fmt(lost_pct, '.1f')}%** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. TỔNG HỢP SỐ LIỆU ĐO ĐẠC KHÁCH QUAN (EMPIRICAL OBSERVATIONS SUMMARY)",
        "",
        "> [!IMPORTANT]",
        "> Báo cáo này hoàn toàn chỉ ghi nhận dữ liệu đo đạc định lượng khách quan từ các tensor đầu ra.",
        "> Tuyệt đối không suy diễn kết luận nhân quả khi chưa thực hiện các thực nghiệm cô lập biến số (ablation study).",
        "",
        f"1. **Tình trạng số học (Numerical Sanity):** "
        f"{'Phát hiện giá trị không hữu hạn (NaN/Inf) trong trọng số mô hình.' if (weight_audit['best_has_nan'] or weight_audit['last_has_nan']) else 'Toàn bộ tensor trọng số của cả hai checkpoint đều hữu hạn (không có NaN hoặc Inf).'}",
        "",
        "2. **So sánh suy diễn AMP (FP16) vs FP32:** "
        "Số liệu chênh lệch giữa suy luận AMP FP16 và full precision FP32 được ghi nhận tại Bảng 2. "
        "Lưu ý: Phép thử suy luận này kiểm tra tính ổn định số học của mô hình tĩnh ở bước forward pass; "
        "phép thử này KHÔNG suy diễn hoặc loại trừ khả năng underflow gradient trong quá trình lan truyền ngược (backward propagation) khi huấn luyện.",
        "",
        "3. **Phân bố xác suất và ngưỡng binarization:** "
        "Số liệu xác suất cực đại và độ nhạy ngưỡng (threshold sensitivity) từ 0.50 xuống 0.05 được đo đạc định lượng tại Bảng 3.",
        "",
        "4. **Hình thái tổn thương dưới phép chiếu thu nhỏ:** "
        "Tỷ lệ các thành phần liên thông bị triệt tiêu khi nội suy nearest về kích thước 1/2 và 1/4 của mask tổn thương toàn volume được ghi nhận tại Bảng 4.",
    ])

    return "\n".join(lines)


def main() -> None:
    """Main diagnostic execution routine."""
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_best_path = Path(args.ckpt_best).resolve()
    ckpt_last_path = Path(args.ckpt_last).resolve()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("=" * 76)
    logger.info("ISLES-2022 MODEL COLLAPSE & DEGRADATION DIAGNOSTIC ENGINE (v2.1)")
    logger.info("=" * 76)

    # 1. Audit Environment & Verify Checkpoint Integrity
    env_info = audit_environment(device, expected_cuda=args.expected_cuda)
    ckpt_b, meta_b = verify_checkpoint_integrity(ckpt_best_path, args.expected_best_epoch, args.fold)
    ckpt_l, meta_l = verify_checkpoint_integrity(ckpt_last_path, args.expected_last_epoch, args.fold)

    # 2. Cross-check Split with Dataset Parser
    all_cases = parse_isles_dataset(data_dir=args.data_dir, strict_isles250=True)
    splits = get_kfold_splits(all_cases, n_splits=args.n_splits, seed=args.seed)
    cross_check_split_with_dataset(all_cases, splits, args.fold, meta_b, meta_l)

    sd_b = ckpt_b.get("model_state_dict", ckpt_b.get("state_dict", ckpt_b))
    sd_l = ckpt_l.get("model_state_dict", ckpt_l.get("state_dict", ckpt_l))

    # 3. Audit Weights Health & Parameter Norm Drift
    logger.info("\n>>> [1/4] Auditing Weights Health & Layer-by-Layer Drift...")
    weight_audit = audit_weights(sd_b, sd_l)
    logger.info("Weight Audit: NaN in Best=%s | NaN in Last=%s | Inf in Best=%s | Inf in Last=%s",
                weight_audit["best_has_nan"], weight_audit["last_has_nan"],
                weight_audit["best_has_inf"], weight_audit["last_has_inf"])
    logger.info("Total L2 Norm: Best=%.4f | Last=%.4f | L2 Param Distance=%.4f",
                weight_audit["best_total_l2"], weight_audit["last_total_l2"], weight_audit["l2_distance"])

    for item in weight_audit["layer_drift"][:5]:
        logger.info("  - Layer %s: rel_drift=%.4f (best_norm=%.4f, last_norm=%.4f)",
                    item["name"], item["rel_drift"], item["best_norm"], item["last_norm"])

    # 4. Instantiate Models
    logger.info("\n>>> [2/4] Instantiating Dual-Encoder SegResNet Models...")
    model_best = get_model(
        model_name="dual_segresnet",
        in_channels=2,
        out_channels=1,
        init_filters=32,
        fusion_mode="bidirectional",
        deep_supervision=True,
    ).to(device)
    model_best.load_state_dict(sd_b, strict=True)
    model_best.eval()

    model_last = get_model(
        model_name="dual_segresnet",
        in_channels=2,
        out_channels=1,
        init_filters=32,
        fusion_mode="bidirectional",
        deep_supervision=True,
    ).to(device)
    model_last.load_state_dict(sd_l, strict=True)
    model_last.eval()

    # 5. Diagnostic Validation Loop
    logger.info("\n>>> [3/4] Running Dual-Precision Diagnostic Evaluation on Validation Cases...")
    _, val_cases = splits[args.fold]
    val_transforms = get_val_transforms(modalities=["dwi", "adc"])
    val_ds = md.Dataset(data=val_cases, transform=val_transforms)

    step_case = max(len(val_ds) // args.num_cases, 1)
    diagnose_indices = list(range(0, len(val_ds), step_case))[:args.num_cases]
    sw_roi = tuple(args.sw_roi_size)

    case_diagnostics = []
    deep_sup_distortions = []

    for idx in diagnose_indices:
        item = val_ds[idx]
        case_id = val_cases[idx].get("case_id", f"case_{idx:04d}")
        img = item["image"].unsqueeze(0)  # [1, 2, H, W, D]
        mask = item["mask"].unsqueeze(0)  # [1, 1, H, W, D]

        logger.info("\nDiagnosing Case [%s] (shape: %s)...", case_id, tuple(img.shape[2:]))

        distortion = analyze_deep_supervision_distortion(mask)
        distortion["case_id"] = case_id
        deep_sup_distortions.append(distortion)

        # Run Best Model: AMP Pass
        logits_b_amp, _ = run_sliding_window_pass(model_best, img, sw_roi, args.overlap, device, use_amp=True)
        diag_b_amp = compute_case_diagnostics(logits_b_amp, mask)

        # Run Best Model: FP32 Pass
        if not args.skip_fp32_comparison and device.type == "cuda":
            logits_b_fp32, _ = run_sliding_window_pass(model_best, img, sw_roi, args.overlap, device, use_amp=False)
            diag_b_fp32 = compute_case_diagnostics(logits_b_fp32, mask)
            prec_diff_b = compare_amp_vs_fp32(logits_b_amp, logits_b_fp32)
        else:
            diag_b_fp32 = compute_case_diagnostics(None, mask, status="not_run")
            prec_diff_b = compare_amp_vs_fp32(None, None)

        # Run Last Model: AMP Pass
        logits_l_amp, _ = run_sliding_window_pass(model_last, img, sw_roi, args.overlap, device, use_amp=True)
        diag_l_amp = compute_case_diagnostics(logits_l_amp, mask)

        # Run Last Model: FP32 Pass
        if not args.skip_fp32_comparison and device.type == "cuda":
            logits_l_fp32, _ = run_sliding_window_pass(model_last, img, sw_roi, args.overlap, device, use_amp=False)
            diag_l_fp32 = compute_case_diagnostics(logits_l_fp32, mask)
            prec_diff_l = compare_amp_vs_fp32(logits_l_amp, logits_l_fp32)
        else:
            diag_l_fp32 = compute_case_diagnostics(None, mask, status="not_run")
            prec_diff_l = compare_amp_vs_fp32(None, None)

        logger.info("  [BEST Ep %s AMP] Dice@0.50=%s | Max Prob=%s",
                    meta_b["epoch"], fmt(diag_b_amp["threshold_dices"].get("0.50")), fmt(diag_b_amp.get("prob_max")))
        logger.info("  [LAST Ep %s AMP] Dice@0.50=%s | Max Prob=%s | Sweep: D@0.20=%s, D@0.10=%s",
                    meta_l["epoch"], fmt(diag_l_amp["threshold_dices"].get("0.50")), fmt(diag_l_amp.get("prob_max")),
                    fmt(diag_l_amp["threshold_dices"].get("0.20")), fmt(diag_l_amp["threshold_dices"].get("0.10")))
        logger.info("  Precision Parity Status: Best=%s (Diff=%s) | Last=%s (Diff=%s)",
                    prec_diff_b["status"], fmt(prec_diff_b.get("max_logit_abs_diff"), ".2e"),
                    prec_diff_l["status"], fmt(prec_diff_l.get("max_logit_abs_diff"), ".2e"))

        case_diagnostics.append({
            "case_id": case_id,
            "shape": list(img.shape[2:]),
            "gt_voxels": distortion["orig_voxels"],
            "gt_components": distortion["orig_components"],
            "best_amp": diag_b_amp,
            "best_fp32": diag_b_fp32,
            "best_prec_diff": prec_diff_b,
            "last_amp": diag_l_amp,
            "last_fp32": diag_l_fp32,
            "last_prec_diff": prec_diff_l,
        })

    # 6. Generate Empirical Markdown Report & JSON Artifacts
    logger.info("\n>>> [4/4] Writing Empirical Observation Report...")
    report_file = output_dir / "collapse_diagnosis_report.md"
    json_file = output_dir / "diagnosis_metrics.json"

    json_payload = {
        "timestamp": datetime.now().isoformat(),
        "environment": env_info,
        "ckpt_best": str(ckpt_best_path),
        "ckpt_last": str(ckpt_last_path),
        "manifest_best": meta_b,
        "manifest_last": meta_l,
        "weight_audit": weight_audit,
        "case_diagnostics": case_diagnostics,
        "deep_supervision_distortions": deep_sup_distortions,
    }
    json_file.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    report_content = build_markdown_report(
        env_info=env_info,
        meta_b=meta_b,
        meta_l=meta_l,
        weight_audit=weight_audit,
        case_diagnostics=case_diagnostics,
        deep_sup_distortions=deep_sup_distortions,
        ckpt_best_path=ckpt_best_path,
        ckpt_last_path=ckpt_last_path,
        sw_roi=sw_roi,
        overlap=args.overlap,
    )
    report_file.write_text(report_content, encoding="utf-8")
    logger.info("Report successfully written to: %s", report_file)
    logger.info("JSON metrics successfully written to: %s", json_file)
    print("\n" + report_content)


if __name__ == "__main__":
    main()
