"""Unit tests for diagnostic report building under synthetic edge cases on CPU.

Covers:
1. All metrics finite (normal case).
2. AMP has NaN, FP32 finite.
3. FP32 has NaN/Inf.
4. FP32 not run (skipped / CPU environment).
Verifies:
- JSON and Markdown export cleanly.
- N/A is preserved (never converted to 0).
- No false equivalence claims when FP32 is skipped or non-finite.
- Strict NaN/Inf gating prevents calculation crashes.
"""

import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_collapse import (
    build_markdown_report,
    compare_amp_vs_fp32,
    compute_case_diagnostics,
    cross_check_split_with_dataset,
    fmt,
)
from src.dataset.parser import compute_split_fingerprint


class TestDiagnosticsReport(unittest.TestCase):
    """Test suite for report building and diagnostic functions."""

    def setUp(self):
        self.env_info = {
            "python_version": "3.10.11",
            "torch_version": "2.1.2",
            "cuda_available": False,
            "torch_cuda_version": None,
            "device_type": "cpu",
            "device_name": "CPU",
            "total_memory_mb": 0.0,
        }
        self.meta_b = {
            "epoch": 70,
            "stated_score": 0.6816,
            "manifest_present": True,
            "model_name": "dual_segresnet",
            "roi_size": [96, 96, 80],
            "val_roi_size": [96, 96, 64],
            "loss_type": "dice_focal",
            "deep_supervision": True,
            "fold": 0,
            "val_sha256": "e505d2eb32670af7",
        }
        self.meta_l = {
            "epoch": 150,
            "stated_score": 0.0000,
            "manifest_present": True,
            "model_name": "dual_segresnet",
            "roi_size": [96, 96, 80],
            "val_roi_size": [96, 96, 64],
            "loss_type": "dice_focal",
            "deep_supervision": True,
            "fold": 0,
            "val_sha256": "e505d2eb32670af7",
        }
        self.weight_audit = {
            "best_has_nan": False,
            "best_has_inf": False,
            "last_has_nan": False,
            "last_has_inf": False,
            "best_total_l2": 119.5179,
            "last_total_l2": 124.3988,
            "l2_distance": 23.0615,
            "layer_drift": [
                {"name": "conv_final.weight", "best_norm": 1.2, "last_norm": 1.1, "diff_norm": 0.3, "rel_drift": 0.25},
                {"name": "aux_head0.weight", "best_norm": 0.8, "last_norm": 0.7, "diff_norm": 0.2, "rel_drift": 0.20},
            ],
        }
        self.deep_sup = [{
            "case_id": "case_test_01",
            "orig_voxels": 500,
            "orig_components": 3,
            "stage1_voxels": 120,
            "stage1_lost_components": 1,
            "stage0_voxels": 20,
            "stage0_lost_components": 2,
        }]

    def test_case1_all_metrics_finite(self):
        """Case 1: All metrics finite."""
        gt_mask = torch.zeros(1, 1, 16, 16, 16)
        gt_mask[0, 0, 4:8, 4:8, 4:8] = 1.0  # 64 voxels

        logits_amp = torch.full((1, 1, 16, 16, 16), -2.0)
        logits_fp32 = torch.full((1, 1, 16, 16, 16), -2.01)

        diag_amp = compute_case_diagnostics(logits_amp, gt_mask)
        diag_fp32 = compute_case_diagnostics(logits_fp32, gt_mask)
        prec_diff = compare_amp_vs_fp32(logits_amp, logits_fp32)

        self.assertEqual(prec_diff["status"], "computed")
        self.assertAlmostEqual(prec_diff["max_logit_abs_diff"], 0.01, places=3)
        self.assertFalse(diag_amp["has_nan"])
        self.assertIsNotNone(diag_amp["threshold_dices"]["0.50"])

        case_diag = [{
            "case_id": "case_test_01",
            "shape": [16, 16, 16],
            "gt_voxels": 64,
            "gt_components": 1,
            "best_amp": diag_amp,
            "best_fp32": diag_fp32,
            "best_prec_diff": prec_diff,
            "last_amp": diag_amp,
            "last_fp32": diag_fp32,
            "last_prec_diff": prec_diff,
        }]

        md = build_markdown_report(
            self.env_info, self.meta_b, self.meta_l, self.weight_audit,
            case_diag, self.deep_sup, Path("best.pth"), Path("last.pth"),
            (96, 96, 64), 0.25,
        )
        self.assertIn("BÁO CÁO ĐO ĐẠC THỰC NGHIỆM", md)
        self.assertIn("case_test_01", md)
        # Verify JSON serializability
        json_str = json.dumps(case_diag, indent=2)
        self.assertIsInstance(json_str, str)

    def test_case2_amp_has_nan_fp32_finite(self):
        """Case 2: AMP has NaN, FP32 is finite."""
        gt_mask = torch.zeros(1, 1, 16, 16, 16)
        gt_mask[0, 0, 4:8, 4:8, 4:8] = 1.0

        logits_amp = torch.full((1, 1, 16, 16, 16), -2.0)
        logits_amp[0, 0, 0, 0, 0] = float("nan")  # Inject NaN
        logits_fp32 = torch.full((1, 1, 16, 16, 16), -2.0)

        diag_amp = compute_case_diagnostics(logits_amp, gt_mask)
        diag_fp32 = compute_case_diagnostics(logits_fp32, gt_mask)
        prec_diff = compare_amp_vs_fp32(logits_amp, logits_fp32)

        # Strict checks
        self.assertTrue(diag_amp["has_nan"])
        self.assertEqual(diag_amp["nan_count"], 1)
        self.assertEqual(diag_amp["status"], "error_non_finite_logits")
        self.assertIsNone(diag_amp["prob_max"])
        self.assertIsNone(diag_amp["threshold_dices"]["0.50"])
        # Check compare_amp_vs_fp32 blocked non-finite computation
        self.assertEqual(prec_diff["status"], "error_non_finite")
        self.assertIsNone(prec_diff["max_logit_abs_diff"])

        case_diag = [{
            "case_id": "case_test_nan",
            "shape": [16, 16, 16],
            "gt_voxels": 64,
            "gt_components": 1,
            "best_amp": diag_amp,
            "best_fp32": diag_fp32,
            "best_prec_diff": prec_diff,
            "last_amp": diag_amp,
            "last_fp32": diag_fp32,
            "last_prec_diff": prec_diff,
        }]

        md = build_markdown_report(
            self.env_info, self.meta_b, self.meta_l, self.weight_audit,
            case_diag, self.deep_sup, Path("best.pth"), Path("last.pth"),
            (96, 96, 64), 0.25,
        )
        self.assertIn("error_non_finite", md)
        self.assertIn("N/A", md)
        # Ensure N/A was not converted to 0.0000
        self.assertNotIn("0.0000 | 0.0000 | error_non_finite", md)

    def test_case3_fp32_has_inf(self):
        """Case 3: FP32 has Inf, AMP is finite."""
        gt_mask = torch.zeros(1, 1, 16, 16, 16)
        gt_mask[0, 0, 4:8, 4:8, 4:8] = 1.0

        logits_amp = torch.full((1, 1, 16, 16, 16), -2.0)
        logits_fp32 = torch.full((1, 1, 16, 16, 16), -2.0)
        logits_fp32[0, 0, 0, 0, 0] = float("inf")  # Inject Inf

        diag_amp = compute_case_diagnostics(logits_amp, gt_mask)
        diag_fp32 = compute_case_diagnostics(logits_fp32, gt_mask)
        prec_diff = compare_amp_vs_fp32(logits_amp, logits_fp32)

        self.assertTrue(diag_fp32["has_inf"])
        self.assertEqual(diag_fp32["inf_count"], 1)
        self.assertEqual(prec_diff["status"], "error_non_finite")
        self.assertIsNone(prec_diff["max_logit_abs_diff"])

        case_diag = [{
            "case_id": "case_test_inf",
            "shape": [16, 16, 16],
            "gt_voxels": 64,
            "gt_components": 1,
            "best_amp": diag_amp,
            "best_fp32": diag_fp32,
            "best_prec_diff": prec_diff,
            "last_amp": diag_amp,
            "last_fp32": diag_fp32,
            "last_prec_diff": prec_diff,
        }]

        md = build_markdown_report(
            self.env_info, self.meta_b, self.meta_l, self.weight_audit,
            case_diag, self.deep_sup, Path("best.pth"), Path("last.pth"),
            (96, 96, 64), 0.25,
        )
        self.assertIn("case_test_inf", md)

    def test_case4_fp32_not_run(self):
        """Case 4: FP32 not run (e.g. --skip-fp32-comparison)."""
        gt_mask = torch.zeros(1, 1, 16, 16, 16)
        gt_mask[0, 0, 4:8, 4:8, 4:8] = 1.0

        logits_amp = torch.full((1, 1, 16, 16, 16), -2.0)

        diag_amp = compute_case_diagnostics(logits_amp, gt_mask)
        diag_fp32 = compute_case_diagnostics(None, gt_mask, status="not_run")
        prec_diff = compare_amp_vs_fp32(None, None)

        self.assertEqual(diag_fp32["status"], "not_run")
        self.assertIsNone(diag_fp32["prob_max"])
        self.assertEqual(prec_diff["status"], "not_run")
        self.assertIsNone(prec_diff["max_logit_abs_diff"])

        case_diag = [{
            "case_id": "case_test_not_run",
            "shape": [16, 16, 16],
            "gt_voxels": 64,
            "gt_components": 1,
            "best_amp": diag_amp,
            "best_fp32": diag_fp32,
            "best_prec_diff": prec_diff,
            "last_amp": diag_amp,
            "last_fp32": diag_fp32,
            "last_prec_diff": prec_diff,
        }]

        md = build_markdown_report(
            self.env_info, self.meta_b, self.meta_l, self.weight_audit,
            case_diag, self.deep_sup, Path("best.pth"), Path("last.pth"),
            (96, 96, 64), 0.25,
        )
        # Must show not_run or N/A, never 0.0000 or identical to AMP
        self.assertIn("not_run", md)
        self.assertIn("N/A", md)

    def test_empty_voxel_group_null_stats(self):
        """Verifies that empty foreground or background groups retain None instead of 0 or dummy values."""
        gt_empty = torch.zeros(1, 1, 16, 16, 16)  # No foreground voxels
        logits = torch.full((1, 1, 16, 16, 16), -1.5)

        diag = compute_case_diagnostics(logits, gt_empty)
        self.assertIsNone(diag["fg_prob_mean"])
        self.assertIsNone(diag["fg_prob_median"])
        self.assertIsNone(diag["fg_prob_max"])
        self.assertIsNone(diag["fg_logit_mean"])
        self.assertIsNone(diag["fg_logit_max"])
        self.assertIsNotNone(diag["bg_prob_mean"])

    def _create_mock_dataset_and_splits(self):
        """Generates synthetic 250 cases and fold 0 split (200 train / 50 val)."""
        cases = [{"case_id": f"sub-strokecase{i:04d}"} for i in range(1, 251)]
        train_cases = cases[:200]
        val_cases = cases[200:]
        splits = [(train_cases, val_cases)]
        return cases, splits

    def test_cross_check_split_success(self):
        """Verifies cross_check_split_with_dataset passes when split hashes and run identity match."""
        cases, splits = self._create_mock_dataset_and_splits()
        train_cases, val_cases = splits[0]
        fp = compute_split_fingerprint(train_cases, val_cases)

        meta_best = {
            "epoch": 70,
            "train_sha256": fp["train_sha256"],
            "val_sha256": fp["val_sha256"],
            "roi_size": [96, 96, 80],
            "val_roi_size": [96, 96, 64],
            "model_name": "dual_segresnet",
            "loss_type": "dice_focal",
            "deep_supervision": True,
            "run_created_at": "2026-11-13T09:21:13.045948+00:00",
            "git_commit": "ffe09856961b92360009b9b526da0b6f2dd54ada",
            "config_sha256": "950df13d331d02988595df2f81938780f7b192d5f90f6fc62006911b0e044593",
        }
        meta_last = dict(meta_best)
        meta_last["epoch"] = 150

        # Should pass with no exception
        cross_check_split_with_dataset(cases, splits, 0, meta_best, meta_last)

    def test_cross_check_split_mismatched_train_hash(self):
        """Verifies that mismatched train split SHA256 raises ValueError."""
        cases, splits = self._create_mock_dataset_and_splits()
        train_cases, val_cases = splits[0]
        fp = compute_split_fingerprint(train_cases, val_cases)

        meta_best = {
            "epoch": 70,
            "train_sha256": "tampered_train_hash",
            "val_sha256": fp["val_sha256"],
            "roi_size": [96, 96, 80],
            "val_roi_size": [96, 96, 64],
            "model_name": "dual_segresnet",
            "loss_type": "dice_focal",
            "deep_supervision": True,
            "run_created_at": "2026-11-13T09:21:13.045948+00:00",
            "git_commit": "ffe09856961b92360009b9b526da0b6f2dd54ada",
            "config_sha256": "950df13d331d02988595df2f81938780f7b192d5f90f6fc62006911b0e044593",
        }
        meta_last = dict(meta_best)
        meta_last["train_sha256"] = fp["train_sha256"]

        with self.assertRaises(ValueError) as ctx:
            cross_check_split_with_dataset(cases, splits, 0, meta_best, meta_last)
        self.assertIn("Train SHA256 mismatch", str(ctx.exception))

    def test_cross_check_split_mismatched_val_hash(self):
        """Verifies that mismatched validation split SHA256 raises ValueError."""
        cases, splits = self._create_mock_dataset_and_splits()
        train_cases, val_cases = splits[0]
        fp = compute_split_fingerprint(train_cases, val_cases)

        meta_best = {
            "epoch": 70,
            "train_sha256": fp["train_sha256"],
            "val_sha256": fp["val_sha256"],
            "roi_size": [96, 96, 80],
            "val_roi_size": [96, 96, 64],
            "model_name": "dual_segresnet",
            "loss_type": "dice_focal",
            "deep_supervision": True,
            "run_created_at": "2026-11-13T09:21:13.045948+00:00",
            "git_commit": "ffe09856961b92360009b9b526da0b6f2dd54ada",
            "config_sha256": "950df13d331d02988595df2f81938780f7b192d5f90f6fc62006911b0e044593",
        }
        meta_last = dict(meta_best)
        meta_last["val_sha256"] = "tampered_val_hash"

        with self.assertRaises(ValueError) as ctx:
            cross_check_split_with_dataset(cases, splits, 0, meta_best, meta_last)
        self.assertIn("Val SHA256 mismatch", str(ctx.exception))

    def test_cross_check_split_mismatched_run_identity(self):
        """Verifies that checkpoints from different runs (mismatched created_at) are blocked."""
        cases, splits = self._create_mock_dataset_and_splits()
        train_cases, val_cases = splits[0]
        fp = compute_split_fingerprint(train_cases, val_cases)

        meta_best = {
            "epoch": 70,
            "train_sha256": fp["train_sha256"],
            "val_sha256": fp["val_sha256"],
            "roi_size": [96, 96, 80],
            "val_roi_size": [96, 96, 64],
            "model_name": "dual_segresnet",
            "loss_type": "dice_focal",
            "deep_supervision": True,
            "run_created_at": "2026-11-13T09:21:13.045948+00:00",
            "git_commit": "ffe09856961b92360009b9b526da0b6f2dd54ada",
            "config_sha256": "950df13d331d02988595df2f81938780f7b192d5f90f6fc62006911b0e044593",
        }
        meta_last = dict(meta_best)
        meta_last["run_created_at"] = "2026-11-13T10:00:00.000000+00:00"  # Different run

        with self.assertRaises(ValueError) as ctx:
            cross_check_split_with_dataset(cases, splits, 0, meta_best, meta_last)
        self.assertIn("Run created_at mismatch", str(ctx.exception))

    def test_cross_check_split_mismatched_config_sha256(self):
        """Verifies that checkpoints with different config hashes are blocked."""
        cases, splits = self._create_mock_dataset_and_splits()
        train_cases, val_cases = splits[0]
        fp = compute_split_fingerprint(train_cases, val_cases)

        meta_best = {
            "epoch": 70,
            "train_sha256": fp["train_sha256"],
            "val_sha256": fp["val_sha256"],
            "roi_size": [96, 96, 80],
            "val_roi_size": [96, 96, 64],
            "model_name": "dual_segresnet",
            "loss_type": "dice_focal",
            "deep_supervision": True,
            "run_created_at": "2026-11-13T09:21:13.045948+00:00",
            "git_commit": "ffe09856961b92360009b9b526da0b6f2dd54ada",
            "config_sha256": "950df13d331d02988595df2f81938780f7b192d5f90f6fc62006911b0e044593",
        }
        meta_last = dict(meta_best)
        meta_last["config_sha256"] = "different_config_sha256"

        with self.assertRaises(ValueError) as ctx:
            cross_check_split_with_dataset(cases, splits, 0, meta_best, meta_last)
        self.assertIn("Config SHA256 mismatch", str(ctx.exception))

    def test_cross_check_split_invalid_case_count(self):
        """Verifies that dataset with count != 250 is blocked."""
        cases, splits = self._create_mock_dataset_and_splits()
        short_cases = cases[:240]

        meta_best = {"train_sha256": "a", "val_sha256": "b"}
        meta_last = {"train_sha256": "a", "val_sha256": "b"}

        with self.assertRaises(ValueError) as ctx:
            cross_check_split_with_dataset(short_cases, splits, 0, meta_best, meta_last)
        self.assertIn("Expected 250 cases", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

