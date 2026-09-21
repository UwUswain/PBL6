"""Unit test suite for ISLES-2022 medical segmentation metrics."""

from pathlib import Path
import unittest
import nibabel as nib
import numpy as np
from scipy import ndimage

from src.metrics.dice import compute_dice
from src.metrics.evaluator import ISLESBatchEvaluator, evaluate_isles_case
from src.metrics.lesion import compute_lesion_metrics
from src.metrics.volume import (
    compute_absolute_volume_difference,
    compute_lesion_volume,
    compute_relative_volume_difference,
)


class TestISLESMetrics(unittest.TestCase):
    """Test cases for ISLES 2022 metrics suite."""

    def setUp(self) -> None:
        self.shape = (64, 64, 32)
        self.spacing = (1.0, 1.0, 2.0)  # voxel vol = 2 mm^3

    def test_dice_scores(self) -> None:
        """Tests DSC calculation under perfect match, disjoint, and empty states."""
        # 1. Perfect match
        mask = np.zeros(self.shape, dtype=np.uint8)
        mask[10:20, 10:20, 5:15] = 1
        self.assertAlmostEqual(compute_dice(mask, mask), 1.0, places=5)

        # 2. Both empty
        empty_pred = np.zeros(self.shape, dtype=np.uint8)
        empty_gt = np.zeros(self.shape, dtype=np.uint8)
        self.assertAlmostEqual(compute_dice(empty_pred, empty_gt), 1.0, places=5)

        # 3. Completely disjoint
        pred = np.zeros(self.shape, dtype=np.uint8)
        gt = np.zeros(self.shape, dtype=np.uint8)
        pred[5:15, 5:15, 5:15] = 1
        gt[30:40, 30:40, 5:15] = 1
        self.assertAlmostEqual(compute_dice(pred, gt), 0.0, places=5)

        # 4. One empty, one non-empty
        self.assertAlmostEqual(compute_dice(empty_pred, gt), 0.0, places=5)
        self.assertAlmostEqual(compute_dice(pred, empty_gt), 0.0, places=5)

        # 5. Partial overlap (50% overlap of identical size)
        # Pred: [0:10, 0:10, 0:10] (1000 voxels)
        # GT:   [5:15, 0:10, 0:10] (1000 voxels)
        # Inter: [5:10, 0:10, 0:10] (500 voxels)
        # DSC = 2 * 500 / (1000 + 1000) = 0.5
        m1 = np.zeros(self.shape, dtype=np.uint8)
        m2 = np.zeros(self.shape, dtype=np.uint8)
        m1[0:10, 0:10, 0:10] = 1
        m2[5:15, 0:10, 0:10] = 1
        self.assertAlmostEqual(compute_dice(m1, m2), 0.5, places=5)

    def test_volume_metrics(self) -> None:
        """Tests volume in mL and Absolute Volume Difference (AVD)."""
        # Spacing: 1mm x 1mm x 1mm -> 1 voxel = 1 mm^3 = 0.001 mL
        spacing_1mm = (1.0, 1.0, 1.0)
        mask_1000 = np.zeros((20, 20, 20), dtype=np.uint8)
        mask_1000[0:10, 0:10, 0:10] = 1  # 1000 voxels
        vol = compute_lesion_volume(mask_1000, spacing_1mm)
        self.assertAlmostEqual(vol, 1.0, places=5)  # 1.0 mL

        mask_1500 = np.zeros((20, 20, 20), dtype=np.uint8)
        mask_1500[0:15, 0:10, 0:10] = 1  # 1500 voxels
        avd = compute_absolute_volume_difference(mask_1500, mask_1000, spacing_1mm)
        self.assertAlmostEqual(avd, 0.5, places=5)  # |1.5 - 1.0| = 0.5 mL

        rvd = compute_relative_volume_difference(mask_1500, mask_1000)
        self.assertAlmostEqual(rvd, 0.5, places=5)  # (1500 - 1000) / 1000 = 0.5

    def test_lesion_metrics_multi_lesion(self) -> None:
        """Tests LCD and Lesion F1 detection under single and multi-lesion scenarios."""
        # 1. Both empty
        empty = np.zeros(self.shape, dtype=np.uint8)
        res_empty = compute_lesion_metrics(empty, empty)
        self.assertEqual(res_empty["lcd"], 0)
        self.assertAlmostEqual(res_empty["lesion_f1"], 1.0, places=5)

        # 2. Perfect single lesion match
        m = np.zeros(self.shape, dtype=np.uint8)
        m[10:20, 10:20, 10:20] = 1
        res_perf = compute_lesion_metrics(m, m)
        self.assertEqual(res_perf["lcd"], 0)
        self.assertAlmostEqual(res_perf["lesion_f1"], 1.0, places=5)
        self.assertEqual(res_perf["pred_lesion_count"], 1)
        self.assertEqual(res_perf["gt_lesion_count"], 1)

        # 3. Multi-lesion: GT has 3 lesions (1 large, 2 satellite), Pred only captures large
        gt = np.zeros(self.shape, dtype=np.uint8)
        gt[5:20, 5:20, 5:15] = 1      # Lesion 1 (large)
        gt[40:45, 40:45, 5:10] = 1    # Lesion 2 (satellite)
        gt[50:55, 10:15, 20:25] = 1   # Lesion 3 (satellite)

        pred = np.zeros(self.shape, dtype=np.uint8)
        pred[5:20, 5:20, 5:15] = 1    # Matches Lesion 1 perfectly

        res_multi = compute_lesion_metrics(pred, gt, iou_threshold=0.2)
        self.assertEqual(res_multi["gt_lesion_count"], 3)
        self.assertEqual(res_multi["pred_lesion_count"], 1)
        self.assertEqual(res_multi["lcd"], 2)  # |1 - 3| = 2
        # Pred has 1 lesion, matches 1 GT -> Precision = 1.0
        # GT has 3 lesions, 1 detected -> Recall = 1/3 ~ 0.3333
        # F1 = 2 * 1.0 * 0.3333 / (1.0 + 0.3333) = 0.5
        self.assertAlmostEqual(res_multi["lesion_precision"], 1.0, places=5)
        self.assertAlmostEqual(res_multi["lesion_recall"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(res_multi["lesion_f1"], 0.5, places=5)

    def test_connectivity_difference(self) -> None:
        """Tests that diagonal voxels are 1 component in 26-conn and 2 components in 6-conn."""
        diag_mask = np.zeros((10, 10, 10), dtype=np.uint8)
        diag_mask[2, 2, 2] = 1
        diag_mask[3, 3, 3] = 1  # diagonally adjacent

        res_26 = compute_lesion_metrics(diag_mask, diag_mask, connectivity=26)
        res_6 = compute_lesion_metrics(diag_mask, diag_mask, connectivity=6)

        self.assertEqual(res_26["gt_lesion_count"], 1)
        self.assertEqual(res_6["gt_lesion_count"], 2)

    def test_unified_evaluator_and_batch(self) -> None:
        """Tests ISLESMetricsResult and ISLESBatchEvaluator."""
        evaluator = ISLESBatchEvaluator()

        # Case 1: Perfect
        m1 = np.zeros(self.shape, dtype=np.uint8)
        m1[10:20, 10:20, 10:15] = 1
        evaluator.add_case("case_01", m1, m1, self.spacing)

        # Case 2: Partial overlap
        m2 = np.zeros(self.shape, dtype=np.uint8)
        m2[15:25, 10:20, 10:15] = 1
        evaluator.add_case("case_02", m1, m2, self.spacing)

        df = evaluator.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn("dsc", df.columns)
        self.assertIn("avd_ml", df.columns)
        self.assertIn("lcd", df.columns)
        self.assertIn("lesion_f1", df.columns)

        summary = evaluator.compute_summary()
        self.assertIn("dsc", summary)
        self.assertGreater(summary["dsc"]["mean"], 0.0)

    def test_real_isles_nii_sample(self) -> None:
        """Sanity test on real ISLES-2022 dataset case sub-strokecase0001."""
        sample_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "raw"
            / "ISLES-2022"
            / "derivatives"
            / "sub-strokecase0001"
            / "ses-0001"
            / "sub-strokecase0001_ses-0001_msk.nii.gz"
        )

        if not sample_path.exists():
            self.skipTest(f"Sample file not found: {sample_path}")

        img = nib.load(sample_path)
        gt_mask = img.get_fdata().astype(np.uint8)
        spacing = img.header.get_zooms()[:3]

        # 1. Evaluating GT against GT should be perfect
        res_perfect = evaluate_isles_case(gt_mask, gt_mask, spacing)
        self.assertAlmostEqual(res_perfect.dsc, 1.0, places=5)
        self.assertAlmostEqual(res_perfect.avd_ml, 0.0, places=5)
        self.assertEqual(res_perfect.lcd, 0)
        self.assertAlmostEqual(res_perfect.lesion_f1, 1.0, places=5)

        # 2. Perturb GT by 3D dilation to simulate prediction error
        struct = ndimage.generate_binary_structure(3, 1)
        dilated_pred = ndimage.binary_dilation(gt_mask, structure=struct, iterations=1)

        res_dilated = evaluate_isles_case(dilated_pred, gt_mask, spacing)
        # Dice should be high but < 1.0
        self.assertGreater(res_dilated.dsc, 0.5)
        self.assertLess(res_dilated.dsc, 1.0)
        # Volume of dilated should be strictly greater than GT
        self.assertGreater(res_dilated.pred_volume_ml, res_dilated.gt_volume_ml)
        self.assertGreater(res_dilated.avd_ml, 0.0)


if __name__ == "__main__":
    unittest.main()
