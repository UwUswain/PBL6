"""Unit and integration tests for run_manifest parity verification and dry-run save/reload."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import nibabel as nib
import numpy as np
import yaml

from src.dataset.parser import ISLES_2022_EXPECTED_IDS, get_kfold_splits, hash_id_list
from src.utils.manifest import (
    build_run_manifest,
    extract_checkpoint_manifest,
    verify_paired_manifests,
    write_run_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestManifestAndDryRun(unittest.TestCase):
    """Tests manifest completeness, strict paired verification, and full dry-run save/reload."""

    def _make_manifest(
        self,
        tmp_dir: Path,
        sampling_mode: str = "standard",
        is_dry_run: bool = False,
        max_epochs: int = 100,
    ) -> dict:
        cfg_path = tmp_dir / f"cfg_{sampling_mode}.yaml"
        raw_cfg = {
            "experiment": {
                "model_name": "dual_segresnet",
                "in_channels": 2,
                "init_filters": 32,
                "fusion_mode": "bidirectional",
                "deep_supervision": True,
                "sampling_mode": sampling_mode,
                "max_epochs": max_epochs,
                "learning_rate": 0.0002,
                "weight_decay": 0.01,
                "loss": "dice_focal",
                "loss_gamma": 2.0,
                "val_interval": 5,
                "postprocess_min_size": 0,
                "seed": 42,
            },
            "data": {
                "modalities": ["dwi", "adc"],
                "roi_size": [96, 96, 64],
                "batch_size": 2,
                "pos_ratio": 1.0,
                "neg_ratio": 1.0,
                "num_samples_per_volume": 2,
                "sampling_mode": sampling_mode,
            },
        }
        cfg_path.write_text(yaml.safe_dump(raw_cfg), encoding="utf-8")

        cases = [{"case_id": cid} for cid in ISLES_2022_EXPECTED_IDS]
        train_cases, val_cases = get_kfold_splits(cases, n_splits=5, seed=42)[0]
        train_ids = [c["case_id"] for c in train_cases]
        val_ids = [c["case_id"] for c in val_cases]

        split_manifest = {
            "fold": 0,
            "n_splits": 5,
            "split_seed": 42,
            "train_seed": 42,
            "total_cases": 250,
            "dataset_ids_sha256": hash_id_list(ISLES_2022_EXPECTED_IDS, sort_ids=True),
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "train_sha256": hash_id_list(train_ids, sort_ids=False),
            "val_sha256": hash_id_list(val_ids, sort_ids=False),
            "train_ids": train_ids,
            "val_ids": val_ids,
        }
        resolved_cfg = {
            "model_name": "dual_segresnet",
            "in_channels": 2,
            "modalities": ["dwi", "adc"],
            "use_edwi": False,
            "init_filters": 32,
            "fusion_mode": "bidirectional",
            "deep_supervision": True,
            "sampling_mode": sampling_mode,
            "pos_ratio": 1.0,
            "neg_ratio": 1.0,
            "num_samples_per_volume": 2,
            "roi_size": [96, 96, 64],
            "batch_size": 2,
            "loss_type": "dice_focal",
            "loss_gamma": 2.0,
            "loss_alpha": None,
            "learning_rate": 0.0002,
            "weight_decay": 0.01,
            "max_epochs": max_epochs,
            "total_epochs": 1 if is_dry_run else max_epochs,
            "val_interval": 5,
            "postprocess_min_size": 0,
            "amp": True,
            "seed": 42,
            "is_dry_run": is_dry_run,
        }
        return build_run_manifest(
            repo_root=PROJECT_ROOT,
            config_path=cfg_path,
            raw_cfg=raw_cfg,
            resolved_cfg=resolved_cfg,
            split_manifest=split_manifest,
            data_dir="data/raw/ISLES-2022",
            strict_isles250=True,
            expected_num_cases=250,
            init_sum=100.0,
            init_l2=10.0,
        )

    def test_verify_paired_manifests_checks_strict_parity(self) -> None:
        """Verifies valid parity passes and catches split, hyperparam, code, and dry-run mismatches."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            m_v3 = self._make_manifest(tmp_dir, sampling_mode="standard")
            m_v4 = self._make_manifest(tmp_dir, sampling_mode="component_aware")

            cases = [{"case_id": cid} for cid in ISLES_2022_EXPECTED_IDS]
            val_cases = get_kfold_splits(cases, n_splits=5, seed=42)[0][1]

            summary = verify_paired_manifests(
                m_v3,
                m_v4,
                expected_val_cases=val_cases,
                expected_fold=0,
                expected_n_splits=5,
                expected_split_seed=42,
                strict_isles250=True,
            )
            self.assertEqual(summary["val_count"], 50)
            self.assertTrue(summary["code_matched"])
            self.assertTrue(summary["env_matched"])
            self.assertIn("monai", m_v3.get("environment", {}).get("packages", {}))

            # 1. Sidecar fallback and mismatch detection
            ckpt_file = tmp_dir / "best_metric_model.pth"
            write_run_manifest(m_v3, tmp_dir / "run_manifest.json")
            self.assertEqual(
                extract_checkpoint_manifest({"epoch": 1}, ckpt_file)["split"]["val_sha256"],
                m_v3["split"]["val_sha256"],
            )
            with self.assertRaises(ValueError):
                extract_checkpoint_manifest({}, tmp_dir / "other" / "best_metric_model.pth")

            # 2. Hyperparameter mismatch
            bad_lr = copy.deepcopy(m_v4)
            bad_lr["config"]["resolved"]["learning_rate"] = 0.001
            with self.assertRaises(ValueError):
                verify_paired_manifests(m_v3, bad_lr, val_cases, expected_fold=0, strict_isles250=True)

            # 3. Split mismatch / leakage
            bad_val = copy.deepcopy(m_v4)
            bad_val["split"]["val_ids"] = bad_val["split"]["val_ids"][::-1]
            with self.assertRaises(ValueError):
                verify_paired_manifests(m_v3, bad_val, val_cases, expected_fold=0, strict_isles250=True)

            # 4. Code mismatch
            bad_code = copy.deepcopy(m_v4)
            bad_code["code_version"]["core_files_sha256"]["src/dataset/transforms.py"] = "0" * 64
            with self.assertRaises(ValueError):
                verify_paired_manifests(m_v3, bad_code, val_cases, expected_fold=0, strict_isles250=True)

            # 5. Runtime dependency mismatch
            bad_env = copy.deepcopy(m_v4)
            bad_env["environment"]["packages"]["monai"] = "0.0.0-mismatch"
            with self.assertRaises(ValueError):
                verify_paired_manifests(m_v3, bad_env, val_cases, expected_fold=0, strict_isles250=True)
            bypassed_env = verify_paired_manifests(
                m_v3, bad_env, val_cases, expected_fold=0, strict_isles250=True, allow_env_mismatch=True
            )
            self.assertFalse(bypassed_env["env_matched"])

            # 6. Dry-run checkpoint blocked unless allow_dry_run=True
            d3 = self._make_manifest(tmp_dir, sampling_mode="standard", is_dry_run=True)
            d4 = self._make_manifest(tmp_dir, sampling_mode="component_aware", is_dry_run=True)
            with self.assertRaises(ValueError):
                verify_paired_manifests(d3, d4, val_cases, expected_fold=0, allow_dry_run=False)
            verify_paired_manifests(d3, d4, val_cases, expected_fold=0, allow_dry_run=True)

    def test_end_to_end_synthetic_dry_run_and_paired_eval(self) -> None:
        """Runs tiny synthetic train --dry-run for V3 & V4-Fixed and evaluates both with strict manifest check."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = root / "isles_synth"
            for idx in range(1, 7):
                cid = f"sub-strokecase{idx:04d}"
                img_dir = raw_root / cid / "ses-0001" / "dwi"
                msk_dir = raw_root / "derivatives" / cid / "ses-0001"
                img_dir.mkdir(parents=True, exist_ok=True)
                msk_dir.mkdir(parents=True, exist_ok=True)

                vol = np.random.RandomState(idx).rand(36, 36, 24).astype(np.float32) + 0.1
                msk = np.zeros((36, 36, 24), dtype=np.uint8)
                msk[10:22, 10:22, 6:16] = 1
                affine = np.eye(4, dtype=np.float32)
                for mod in ("dwi", "adc"):
                    nib.save(nib.Nifti1Image(vol, affine), img_dir / f"{cid}_ses-0001_{mod}.nii.gz")
                nib.save(nib.Nifti1Image(msk, affine), msk_dir / f"{cid}_ses-0001_msk.nii.gz")

            cfg_paths = {}
            run_dirs = {}
            for tag, mode in (("v3", "standard"), ("v4", "component_aware")):
                cfg = {
                    "experiment": {
                        "model_name": "dual_segresnet",
                        "in_channels": 2,
                        "init_filters": 8,
                        "fusion_mode": "bidirectional",
                        "deep_supervision": True,
                        "sampling_mode": mode,
                        "max_epochs": 5,
                        "learning_rate": 0.0002,
                        "weight_decay": 0.01,
                        "loss": "dice_focal",
                        "loss_gamma": 2.0,
                        "val_interval": 1,
                        "postprocess_min_size": 0,
                        "seed": 42,
                    },
                    "data": {
                        "modalities": ["dwi", "adc"],
                        "roi_size": [32, 32, 16],
                        "batch_size": 1,
                        "num_workers": 0,
                        "use_cache": False,
                        "pos_ratio": 1.0,
                        "neg_ratio": 1.0,
                        "num_samples_per_volume": 2,
                        "sampling_mode": mode,
                    },
                }
                cfg_p = root / f"{tag}.yaml"
                cfg_p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                cfg_paths[tag] = cfg_p

                save_root = root / f"runs_{tag}"
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "train.py"),
                        "--config",
                        str(cfg_p),
                        "--data-dir",
                        str(raw_root),
                        "--fold",
                        "0",
                        "--n-splits",
                        "3",
                        "--save-dir",
                        str(save_root),
                        "--device",
                        "cpu",
                        "--dry-run",
                        "--dry-run-steps",
                        "2",
                        "--dry-run-val-cases",
                        "2",
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=f"{tag} dry-run failed:\n{proc.stdout}\n{proc.stderr}")
                run_dirs[tag] = next(save_root.glob("dry_run_*"))
                self.assertTrue((run_dirs[tag] / "run_manifest.json").is_file())
                self.assertTrue((run_dirs[tag] / "dry_run_checkpoint.pth").is_file())

            eval_out = root / "eval_out"
            eval_proc = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluate_paired_fold.py"),
                    "--ckpt-v3",
                    str(run_dirs["v3"] / "dry_run_checkpoint.pth"),
                    "--ckpt-v4",
                    str(run_dirs["v4"] / "dry_run_checkpoint.pth"),
                    "--data-dir",
                    str(raw_root),
                    "--fold",
                    "0",
                    "--n-splits",
                    "3",
                    "--postprocess-modes",
                    "0.008ml",
                    "raw",
                    "--allow-dry-run",
                    "--output-dir",
                    str(eval_out),
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(
                eval_proc.returncode,
                0,
                msg=f"evaluate_paired_fold failed:\n{eval_proc.stdout}\n{eval_proc.stderr}",
            )
            report_text = (eval_out / "paired_fold0_comparison_report.md").read_text(encoding="utf-8")
            self.assertLess(
                report_text.index("## Kết Quả Chính: `raw`"),
                report_text.index("## Phân Tích Bổ Sung: `0.008ml`"),
            )
            self.assertIn("Train Split SHA-256", report_text)


if __name__ == "__main__":
    unittest.main()
