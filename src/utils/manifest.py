"""Reproducibility manifest generation and verification helpers for ISLES-2022 runs."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.dataset.parser import ISLES_2022_EXPECTED_IDS, hash_id_list

PARITY_ENV_PACKAGES: Tuple[str, ...] = (
    "torch",
    "monai",
    "nibabel",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "PyYAML",
    "connected-components-3d",
)

CORE_CODE_RELATIVE_PATHS: Tuple[str, ...] = (
    "scripts/train.py",
    "src/dataset/parser.py",
    "src/dataset/dataloader.py",
    "src/dataset/transforms.py",
    "src/models/builder.py",
    "src/models/dual_segresnet.py",
    "src/losses/builder.py",
    "src/metrics/lesion.py",
    "src/utils/postprocess.py",
    "src/utils/manifest.py",
)

PAIRED_PARITY_CONTROL_KEYS: Tuple[str, ...] = (
    "model_name",
    "in_channels",
    "modalities",
    "use_edwi",
    "init_filters",
    "fusion_mode",
    "deep_supervision",
    "roi_size",
    "batch_size",
    "pos_ratio",
    "neg_ratio",
    "num_samples_per_volume",
    "loss_type",
    "loss_gamma",
    "loss_alpha",
    "learning_rate",
    "weight_decay",
    "max_epochs",
    "val_interval",
    "postprocess_min_size",
    "seed",
    "amp",
)


def compute_file_sha256(file_path: Path, normalize_newlines: bool = True) -> str:
    """Computes full SHA-256 hex digest of a file (normalizing CRLF->LF for cross-OS parity)."""
    raw = file_path.read_bytes()
    if normalize_newlines:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _run_git_cmd(repo_root: Path, args: Sequence[str]) -> Optional[str]:
    """Runs a git command safely, returning stripped stdout or None if unavailable."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None


def get_code_version_info(
    repo_root: Path,
    tracked_rel_paths: Sequence[str] = CORE_CODE_RELATIVE_PATHS,
) -> Dict[str, Any]:
    """Captures git commit/dirty metadata plus SHA-256 digests of core training files."""
    git_commit = _run_git_cmd(repo_root, ["rev-parse", "HEAD"])
    git_branch = _run_git_cmd(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    git_status = _run_git_cmd(repo_root, ["status", "--short"])

    dirty_files: List[str] = []
    if git_status:
        dirty_files = [line.strip() for line in git_status.splitlines() if line.strip()][:50]

    core_hashes: Dict[str, Optional[str]] = {}
    for rel_path in tracked_rel_paths:
        target = repo_root / rel_path
        core_hashes[rel_path] = (
            compute_file_sha256(target, normalize_newlines=True) if target.is_file() else None
        )

    return {
        "git_available": git_commit is not None,
        "git_commit": git_commit or "unknown",
        "git_branch": git_branch or "unknown",
        "git_dirty": bool(dirty_files),
        "git_dirty_files": dirty_files,
        "core_files_sha256": core_hashes,
    }


def _normalize_parity_value(value: Any) -> Any:
    """Normalizes Python sequences/primitives for JSON-compatible parity comparisons."""
    if isinstance(value, tuple):
        return [_normalize_parity_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_parity_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_parity_value(v) for k, v in value.items()}
    return value


def get_runtime_environment_info(
    package_names: Sequence[str] = PARITY_ENV_PACKAGES,
) -> Dict[str, Any]:
    """Captures Python, CUDA, and installed dependency versions for paired run reproducibility."""
    packages: Dict[str, Optional[str]] = {}
    for pkg in package_names:
        try:
            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = None

    torch_cuda: Optional[str] = None
    try:
        import torch  # local import avoids hard torch dependency during lightweight checks

        torch_cuda = getattr(torch.version, "cuda", None)
    except ImportError:
        torch_cuda = None

    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.platform(),
        "torch_cuda": torch_cuda,
        "packages": packages,
    }


def build_run_manifest(
    repo_root: Path,
    config_path: Path,
    raw_cfg: Dict[str, Any],
    resolved_cfg: Dict[str, Any],
    split_manifest: Dict[str, Any],
    data_dir: str,
    strict_isles250: bool,
    expected_num_cases: Optional[int],
    init_sum: float,
    init_l2: float,
) -> Dict[str, Any]:
    """Builds a self-contained reproducibility manifest dictionary for a training run."""
    return {
        "manifest_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": get_code_version_info(repo_root),
        "environment": get_runtime_environment_info(),
        "dataset": {
            "data_dir": str(data_dir),
            "strict_isles250": bool(strict_isles250),
            "expected_num_cases": int(expected_num_cases) if expected_num_cases is not None else None,
            "total_cases": int(split_manifest["total_cases"]),
            "dataset_ids_sha256": str(split_manifest["dataset_ids_sha256"]),
        },
        "split": {
            "fold": int(split_manifest["fold"]),
            "n_splits": int(split_manifest["n_splits"]),
            "split_seed": int(split_manifest["split_seed"]),
            "train_seed": int(split_manifest["train_seed"]),
            "train_count": int(split_manifest["train_count"]),
            "val_count": int(split_manifest["val_count"]),
            "train_sha256": str(split_manifest["train_sha256"]),
            "val_sha256": str(split_manifest["val_sha256"]),
            "train_ids": list(split_manifest["train_ids"]),
            "val_ids": list(split_manifest["val_ids"]),
        },
        "config": {
            "config_file": config_path.name,
            "config_sha256": compute_file_sha256(config_path, normalize_newlines=True),
            "raw_yaml": _normalize_parity_value(raw_cfg),
            "resolved": _normalize_parity_value(resolved_cfg),
        },
        "model_init_checksum": {
            "sum": float(init_sum),
            "l2_norm": float(init_l2),
        },
    }


def write_run_manifest(manifest: Dict[str, Any], output_path: Path) -> Path:
    """Serializes the run manifest cleanly as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def extract_checkpoint_manifest(checkpoint: Dict[str, Any], ckpt_path: Path) -> Dict[str, Any]:
    """Loads and cross-validates run_manifest from checkpoint payload or sibling run_manifest.json."""
    embedded = checkpoint.get("run_manifest") if isinstance(checkpoint, dict) else None
    json_path = ckpt_path.parent / "run_manifest.json"
    sidecar = json.loads(json_path.read_text(encoding="utf-8")) if json_path.is_file() else None

    if embedded is None and sidecar is None:
        raise ValueError(
            f"Checkpoint '{ckpt_path}' does not contain 'run_manifest' and no sibling '{json_path.name}' exists."
        )

    if embedded is not None and sidecar is not None:
        for section, key in (
            ("split", "train_sha256"),
            ("split", "val_sha256"),
            ("config", "config_sha256"),
        ):
            if embedded.get(section, {}).get(key) != sidecar.get(section, {}).get(key):
                raise ValueError(
                    f"Manifest mismatch between checkpoint '{ckpt_path}' and sibling '{json_path}' at {section}.{key}."
                )

    return embedded if embedded is not None else sidecar


def _validate_single_manifest_against_val(
    label: str,
    manifest: Dict[str, Any],
    expected_val_ids: List[str],
    expected_val_sha256: str,
    expected_fold: int,
    expected_n_splits: int,
    expected_split_seed: int,
    strict_isles250: bool,
) -> None:
    """Validates one run manifest for leakage-free splits and exact fold/dataset alignment."""
    split = manifest.get("split", {})
    dataset = manifest.get("dataset", {})

    train_ids = list(split.get("train_ids", []))
    val_ids = list(split.get("val_ids", []))
    if not train_ids or not val_ids:
        raise ValueError(f"[{label}] run_manifest is missing non-empty split.train_ids / split.val_ids.")

    overlap = sorted(set(train_ids).intersection(set(val_ids)))
    if overlap:
        raise ValueError(f"[{label}] Train/Val leakage detected in manifest: {overlap[:5]}")

    computed_train_sha = hash_id_list(train_ids, sort_ids=False)
    computed_val_sha = hash_id_list(val_ids, sort_ids=False)
    if split.get("train_sha256") != computed_train_sha or split.get("val_sha256") != computed_val_sha:
        raise ValueError(f"[{label}] Manifest ID list does not match recorded train_sha256 / val_sha256.")

    if int(split.get("fold", -1)) != int(expected_fold):
        raise ValueError(f"[{label}] Fold mismatch: checkpoint={split.get('fold')} vs eval={expected_fold}.")
    if int(split.get("n_splits", -1)) != int(expected_n_splits):
        raise ValueError(
            f"[{label}] n_splits mismatch: checkpoint={split.get('n_splits')} vs eval={expected_n_splits}."
        )
    if int(split.get("split_seed", -1)) != int(expected_split_seed):
        raise ValueError(
            f"[{label}] split_seed mismatch: checkpoint={split.get('split_seed')} vs eval={expected_split_seed}."
        )
    if val_ids != expected_val_ids or split.get("val_sha256") != expected_val_sha256:
        raise ValueError(
            f"[{label}] Validation IDs mismatch! ckpt val_sha256={split.get('val_sha256')} "
            f"vs eval val_sha256={expected_val_sha256}."
        )

    if strict_isles250:
        if int(dataset.get("total_cases", -1)) != 250 or len(train_ids) != 200 or len(val_ids) != 50:
            raise ValueError(
                f"[{label}] Strict ISLES-250 mismatch: total={dataset.get('total_cases')}, "
                f"train={len(train_ids)}, val={len(val_ids)}."
            )
        combined = set(train_ids).union(set(val_ids))
        if combined != set(ISLES_2022_EXPECTED_IDS):
            raise ValueError(f"[{label}] Combined train+val IDs do not match canonical ISLES-2022 0001..0250.")
        expected_ds_sha = hash_id_list(ISLES_2022_EXPECTED_IDS, sort_ids=True)
        if dataset.get("dataset_ids_sha256") != expected_ds_sha:
            raise ValueError(
                f"[{label}] dataset_ids_sha256 mismatch: {dataset.get('dataset_ids_sha256')} vs {expected_ds_sha}."
            )


def verify_paired_manifests(
    manifest_v3: Dict[str, Any],
    manifest_v4: Dict[str, Any],
    expected_val_cases: Sequence[Dict[str, str]],
    expected_fold: int,
    expected_n_splits: int = 5,
    expected_split_seed: int = 42,
    strict_isles250: bool = False,
    allow_code_mismatch: bool = False,
    allow_dry_run: bool = False,
    allow_env_mismatch: bool = False,
    allow_param_mismatch: bool = False,
) -> Dict[str, Any]:
    """Verifies that V3 and V4-Fixed checkpoints satisfy full paired experimental parity."""
    expected_val_ids = [c["case_id"] for c in expected_val_cases]
    expected_val_sha256 = hash_id_list(expected_val_ids, sort_ids=False)

    _validate_single_manifest_against_val(
        "Model_A",
        manifest_v3,
        expected_val_ids,
        expected_val_sha256,
        expected_fold,
        expected_n_splits,
        expected_split_seed,
        strict_isles250,
    )
    _validate_single_manifest_against_val(
        "Model_B",
        manifest_v4,
        expected_val_ids,
        expected_val_sha256,
        expected_fold,
        expected_n_splits,
        expected_split_seed,
        strict_isles250,
    )

    s3, s4 = manifest_v3["split"], manifest_v4["split"]
    if s3["train_ids"] != s4["train_ids"] or s3["train_sha256"] != s4["train_sha256"]:
        raise ValueError(
            f"Paired train split mismatch: Model_A train_sha256={s3['train_sha256']} vs Model_B train_sha256={s4['train_sha256']}."
        )
    if s3["val_ids"] != s4["val_ids"] or s3["val_sha256"] != s4["val_sha256"]:
        raise ValueError(
            f"Paired val split mismatch: Model_A val_sha256={s3['val_sha256']} vs Model_B val_sha256={s4['val_sha256']}."
        )
    if manifest_v3["dataset"]["dataset_ids_sha256"] != manifest_v4["dataset"]["dataset_ids_sha256"]:
        raise ValueError("Paired dataset_ids_sha256 mismatch between checkpoints.")

    c3 = manifest_v3.get("config", {}).get("resolved", {})
    c4 = manifest_v4.get("config", {}).get("resolved", {})

    mode_v3 = c3.get("sampling_mode")
    mode_v4 = c4.get("sampling_mode")
    warnings: List[str] = []
    if mode_v3 != "standard" or mode_v4 != "component_aware":
        if not allow_param_mismatch:
            raise ValueError(
                f"Sampling mode pairing error: expected Model_A='standard' & Model_B='component_aware', "
                f"got Model_A='{mode_v3}' & Model_B='{mode_v4}'. Use --allow-param-mismatch for custom variant comparisons."
            )
        warnings.append(
            f"Custom variant comparison: sampling_modes=(Model_A='{mode_v3}', Model_B='{mode_v4}')."
        )

    dry_v3 = bool(c3.get("is_dry_run", False))
    dry_v4 = bool(c4.get("is_dry_run", False))
    if dry_v3 != dry_v4:
        raise ValueError(f"Cannot mix dry-run and full-run checkpoints (Model_A dry={dry_v3}, Model_B dry={dry_v4}).")
    if (dry_v3 or dry_v4) and not allow_dry_run:
        raise ValueError(
            "Checkpoint(s) come from --dry-run smoke training. Pass --allow-dry-run only for smoke evaluation."
        )

    mismatched_params: List[str] = []
    for key in PAIRED_PARITY_CONTROL_KEYS:
        v3_val = _normalize_parity_value(c3.get(key))
        v4_val = _normalize_parity_value(c4.get(key))
        if v3_val != v4_val:
            mismatched_params.append(f"{key} (Model_A={v3_val!r} vs Model_B={v4_val!r})")
    if mismatched_params:
        if not allow_param_mismatch:
            raise ValueError("Paired hyperparameter parity violation: " + "; ".join(mismatched_params))
        warnings.append("Hyperparameter differences bypassed via --allow-param-mismatch: " + "; ".join(mismatched_params))

    cv3 = manifest_v3.get("code_version", {})
    cv4 = manifest_v4.get("code_version", {})
    code_matched = True
    commit3 = cv3.get("git_commit", "unknown")
    commit4 = cv4.get("git_commit", "unknown")
    hashes3 = cv3.get("core_files_sha256", {})
    hashes4 = cv4.get("core_files_sha256", {})

    code_diffs: List[str] = []
    if commit3 != "unknown" and commit4 != "unknown" and commit3 != commit4:
        code_diffs.append(f"git_commit V3={commit3[:12]} vs V4={commit4[:12]}")
    if hashes3 != hashes4:
        changed_files = [
            k for k in sorted(set(hashes3).union(hashes4)) if hashes3.get(k) != hashes4.get(k)
        ]
        code_diffs.append(f"core_files_sha256 differ in {changed_files}")

    if code_diffs:
        code_matched = False
        if not allow_code_mismatch:
            raise ValueError("Code version mismatch between V3 and V4 checkpoints: " + "; ".join(code_diffs))

    warnings: List[str] = []
    if cv3.get("git_dirty") or cv4.get("git_dirty"):
        warnings.append(
            f"Dirty git tree noted during training (V3 dirty={cv3.get('git_dirty')}, V4 dirty={cv4.get('git_dirty')})."
        )
    if not code_matched:
        warnings.append("Code mismatch bypassed via --allow-code-mismatch: " + "; ".join(code_diffs))

    env3 = manifest_v3.get("environment")
    env4 = manifest_v4.get("environment")
    env_matched = True
    env_diffs: List[str] = []
    if env3 is not None and env4 is not None:
        if env3.get("python_major_minor") != env4.get("python_major_minor"):
            env_diffs.append(
                f"python V3={env3.get('python_major_minor')} vs V4={env4.get('python_major_minor')}"
            )
        if env3.get("torch_cuda") != env4.get("torch_cuda"):
            env_diffs.append(f"torch_cuda V3={env3.get('torch_cuda')} vs V4={env4.get('torch_cuda')}")
        pkgs3 = env3.get("packages", {})
        pkgs4 = env4.get("packages", {})
        for pkg in sorted(set(pkgs3).union(pkgs4)):
            if pkgs3.get(pkg) != pkgs4.get(pkg):
                env_diffs.append(f"{pkg} V3={pkgs3.get(pkg)!r} vs V4={pkgs4.get(pkg)!r}")
        if env_diffs:
            env_matched = False
            if not allow_env_mismatch:
                raise ValueError(
                    "Runtime dependency/environment mismatch between V3 and V4 checkpoints: "
                    + "; ".join(env_diffs)
                )
            warnings.append("Environment mismatch bypassed via --allow-env-mismatch: " + "; ".join(env_diffs))

    return {
        "fold": int(expected_fold),
        "train_count": int(s3["train_count"]),
        "val_count": int(s3["val_count"]),
        "dataset_ids_sha256": str(manifest_v3["dataset"]["dataset_ids_sha256"]),
        "train_sha256": str(s3["train_sha256"]),
        "val_sha256": str(s3["val_sha256"]),
        "v3_sampling_mode": mode_v3,
        "v4_sampling_mode": mode_v4,
        "v3_git_commit": commit3,
        "v4_git_commit": commit4,
        "code_matched": code_matched,
        "env_matched": env_matched,
        "environment_v3": env3 or {},
        "environment_v4": env4 or {},
        "is_dry_run": dry_v3,
        "resolved_v3": c3,
        "resolved_v4": c4,
        "init_checksum_v3": manifest_v3.get("model_init_checksum", {}),
        "init_checksum_v4": manifest_v4.get("model_init_checksum", {}),
        "warnings": warnings,
    }

