"""Unit test suite for ISLES-2022 dataset parser and transforms."""

from pathlib import Path
import unittest
import monai.data as md
import monai.transforms as mt
import monai.utils as mu
import numpy as np
import torch

from src.dataset.parser import (
    compute_split_fingerprint,
    get_kfold_splits,
    parse_isles_dataset,
    verify_isles_dataset,
)
from src.dataset.transforms import RandCropByConnectedComponentsd

DATASET_ROOT = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "ISLES-2022"
)


def _identity_dict(x):
    return x


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

    def test_verify_isles250_guard(self) -> None:
        """Verifies strict ISLES-250 guard accepts real 250 cases and rejects 144/duplicate subsets."""
        if not DATASET_ROOT.exists():
            self.skipTest("Dataset root not found")

        cases = parse_isles_dataset(DATASET_ROOT, require_mask=True)
        summary = verify_isles_dataset(
            cases,
            expected_num_cases=250,
            require_exact_isles250_ids=True,
            required_modalities=("dwi", "adc"),
        )
        self.assertEqual(summary["num_cases"], 250)

        # Reject truncated 144-case dataset
        with self.assertRaises(RuntimeError):
            verify_isles_dataset(cases[:144], expected_num_cases=250, require_exact_isles250_ids=True)

        # Reject duplicate case_id
        with self.assertRaises(RuntimeError):
            verify_isles_dataset(cases[:-1] + [cases[0]], expected_num_cases=250, require_exact_isles250_ids=True)

        splits = get_kfold_splits(cases, n_splits=5, seed=42)
        fp = compute_split_fingerprint(splits[0][0], splits[0][1])
        self.assertEqual(fp["train_count"], 200)
        self.assertEqual(fp["val_count"], 50)

    def test_component_cropper_randomizable_cache_alignment_and_workers(self) -> None:
        """Verifies Randomizable trait, CacheDataset non-frozen cropping, negative brain constraint, and worker determinism."""
        vol_shape = (64, 64, 48)
        img = np.zeros((2, *vol_shape), dtype=np.float32)
        msk = np.zeros((1, *vol_shape), dtype=np.uint8)
        img[:, 10:55, 10:55, 8:42] = 0.5  # valid brain content (> 0)
        # Large component (10x10x10 = 1000 voxels) vs micro component (2 voxels)
        msk[0, 12:22, 12:22, 10:20] = 1
        img[0, 12:22, 12:22, 10:20] = 10.0
        msk[0, 48, 48, 34:36] = 1
        img[0, 48, 48, 34:36] = 20.0

        cropper = RandCropByConnectedComponentsd(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(24, 24, 24),
            pos=1.0,
            neg=1.0,
            num_samples=2,
            image_key="image",
            image_threshold=0.0,
        )
        self.assertIsInstance(cropper, mt.Randomizable)
        self.assertIsInstance(cropper, mt.RandomizableTrait)
        self.assertIsInstance(cropper, mt.MultiSampleTrait)

        pipe = mt.Compose([mt.EnsureTyped(keys=["image", "mask"], track_meta=False), cropper])
        items = [{"image": torch.from_numpy(img.copy()), "mask": torch.from_numpy(msk.copy())} for _ in range(4)]
        ds = md.CacheDataset(data=items, transform=pipe, cache_rate=1.0, num_workers=0)

        # Must vary across repeated CacheDataset reads (NOT frozen!)
        reads = [float(ds[0][0]["image"].sum().item()) for _ in range(8)]
        self.assertGreater(len(set(reads)), 1)

        # Negative samples strictly satisfy label==0 and image>0
        neg_only = RandCropByConnectedComponentsd(
            keys=["image", "mask"],
            spatial_size=(24, 24, 24),
            pos=0.0,
            neg=1.0,
            num_samples=40,
            image_key="image",
            image_threshold=0.0,
        )
        neg_only.set_random_state(seed=17)
        neg_only({"image": torch.from_numpy(img), "mask": torch.from_numpy(msk)})
        for c in neg_only.centers:
            self.assertEqual(int(msk[0, c[0], c[1], c[2]]), 0)
            self.assertTrue(bool((img[:, c[0], c[1], c[2]] > 0).any()))

        # Spatial alignment + component oversampling parity
        pos_only = RandCropByConnectedComponentsd(
            keys=["image", "mask"],
            spatial_size=(24, 24, 24),
            pos=1.0,
            neg=0.0,
            num_samples=100,
            image_key="image",
            image_threshold=0.0,
        )
        pos_only.set_random_state(seed=42)
        pos_samples = pos_only({"image": torch.from_numpy(img), "mask": torch.from_numpy(msk)})
        micro_hits = 0
        for s in pos_samples:
            self.assertEqual(tuple(s["image"].shape), (2, 24, 24, 24))
            self.assertEqual(tuple(s["mask"].shape), (1, 24, 24, 24))
            self.assertTrue(
                torch.equal(s["mask"][0] > 0, (s["image"][0] == 10.0) | (s["image"][0] == 20.0))
            )
            if (s["image"][0] == 20.0).any():
                micro_hits += 1
        self.assertGreater(micro_hits, 25)  # ~50/100 expected vs <1/100 for voxel sampling

        # Seeded reproducibility across multi-worker DataLoader (num_workers=2)
        def run_worker_epoch() -> list:
            mu.set_determinism(seed=42)
            c_w = RandCropByConnectedComponentsd(
                keys=["image", "mask"],
                spatial_size=(24, 24, 24),
                pos=1.0,
                neg=1.0,
                num_samples=2,
                image_key="image",
                image_threshold=0.0,
            )
            tf_w = mt.Compose([mt.Lambdad(keys=["image", "mask"], func=_identity_dict), c_w])
            ds_w = md.CacheDataset(data=items, transform=tf_w, cache_rate=1.0, num_workers=0)
            gen = torch.Generator().manual_seed(42)
            dl = md.DataLoader(
                ds_w,
                batch_size=2,
                shuffle=True,
                num_workers=2,
                collate_fn=md.list_data_collate,
                generator=gen,
            )
            return [round(float(b["image"].sum().item()), 4) for b in dl]

        run_a = run_worker_epoch()
        run_b = run_worker_epoch()
        self.assertEqual(run_a, run_b)


if __name__ == "__main__":
    unittest.main()
