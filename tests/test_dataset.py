"""Unit test suite for ISLES-2022 dataset parser and transforms."""

from pathlib import Path
import unittest
import numpy as np

from src.dataset.parser import get_kfold_splits, parse_isles_dataset

DATASET_ROOT = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "ISLES-2022"
)


class TestISLESDataset(unittest.TestCase):
    """Test suite for ISLES-2022 data parsing and K-Fold splitting."""

    def test_parse_isles_dataset(self) -> None:
        """Verifies that all 250 ISLES-2022 cases are discovered with valid paths."""
        if not DATASET_ROOT.exists():
            self.skipTest(f"Dataset root not found: {DATASET_ROOT}")

        cases = parse_isles_dataset(DATASET_ROOT, require_mask=True)
        self.assertEqual(
            len(cases),
            250,
            f"Expected exactly 250 training cases, found {len(cases)}",
        )

        first_case = cases[0]
        self.assertIn("case_id", first_case)
        self.assertIn("dwi", first_case)
        self.assertIn("adc", first_case)
        self.assertIn("flair", first_case)
        self.assertIn("mask", first_case)

        # Verify all 4 files actually exist on disk
        self.assertTrue(Path(first_case["dwi"]).exists(), "DWI file missing")
        self.assertTrue(Path(first_case["adc"]).exists(), "ADC file missing")
        self.assertTrue(Path(first_case["flair"]).exists(), "FLAIR file missing")
        self.assertTrue(Path(first_case["mask"]).exists(), "Mask file missing")

    def test_kfold_splits_disjoint(self) -> None:
        """Verifies that K-Fold splits have zero overlap and cover the entire dataset."""
        dummy_cases = [{"case_id": f"sub-{i:04d}"} for i in range(100)]
        n_splits = 5
        splits = get_kfold_splits(dummy_cases, n_splits=n_splits, seed=42)

        self.assertEqual(len(splits), n_splits)
        all_val_ids = []

        for fold_idx, (train_split, val_split) in enumerate(splits):
            self.assertEqual(len(train_split), 80)
            self.assertEqual(len(val_split), 20)

            train_ids = {c["case_id"] for c in train_split}
            val_ids = {c["case_id"] for c in val_split}

            # Check zero data leakage
            overlap = train_ids.intersection(val_ids)
            self.assertEqual(
                len(overlap),
                0,
                f"Data leakage in fold {fold_idx}: {overlap}",
            )
            all_val_ids.extend(val_ids)

        # Check all cases were validated exactly once
        self.assertEqual(len(all_val_ids), 100)
        self.assertEqual(len(set(all_val_ids)), 100)

    def test_kfold_splits_real_data(self) -> None:
        """Tests K-Fold splits on the real 250 ISLES cases."""
        if not DATASET_ROOT.exists():
            self.skipTest("Dataset root not found")

        cases = parse_isles_dataset(DATASET_ROOT, require_mask=True)
        splits = get_kfold_splits(cases, n_splits=5, seed=42)

        for fold_idx, (train_cases, val_cases) in enumerate(splits):
            self.assertEqual(len(train_cases), 200)
            self.assertEqual(len(val_cases), 50)


if __name__ == "__main__":
    unittest.main()
