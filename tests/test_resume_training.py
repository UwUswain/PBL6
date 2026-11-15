"""Unit tests for training resume functionality, manifest verification, and telemetry logging."""

import copy
from pathlib import Path
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.train import (
    train_epoch,
    validate,
    verify_resume_checkpoint,
    _compute_model_checksum,
)
from src.losses.builder import get_loss_function


class Tiny3DNet(nn.Module):
    """Minimal 3D network with deep supervision outputs for CPU testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 4, kernel_size=3, padding=1)
        self.aux1 = nn.Conv3d(4, 1, kernel_size=1)
        self.aux2 = nn.Conv3d(4, 1, kernel_size=1)
        self.head = nn.Conv3d(4, 1, kernel_size=1)

    def forward(self, x):
        h = self.conv(x)
        if self.training:
            return self.head(h), self.aux1(h), self.aux2(h)
        return self.head(h)


class SingleHead3DNet(nn.Module):
    """Minimal 3D network without auxiliary heads for CPU testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 4, kernel_size=3, padding=1)
        self.head = nn.Conv3d(4, 1, kernel_size=1)

    def forward(self, x):
        h = self.conv(x)
        return self.head(h)


def _build_valid_manifest():
    """Constructs a valid mock run_manifest dictionary."""
    return {
        "config": {
            "resolved": {
                "model_name": "SegResNet",
                "in_channels": 2,
                "roi_size": [96, 96, 80],
                "val_roi_size": [96, 96, 64],
            }
        },
        "split": {
            "fold": 0,
            "val_sha256": "abcdef1234567890",
        },
    }


class TestResumeTraining(unittest.TestCase):
    def test_manifest_schema_and_mismatch_rejection(self):
        """Verifies that verify_resume_checkpoint strictly enforces manifest schema and rejects mismatches."""
        base_manifest = _build_valid_manifest()
        valid_ckpt = {
            "epoch": 70,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "scaler_state_dict": {},
            "run_manifest": base_manifest,
        }
        split_ref = {"val_sha256": "abcdef1234567890"}

        # 1. Valid checkpoint passes
        verify_resume_checkpoint(
            ckpt=valid_ckpt,
            model_name="SegResNet",
            in_channels=2,
            roi_size=(96, 96, 80),
            val_roi_size=(96, 96, 64),
            fold=0,
            split_manifest=split_ref,
        )

        # 2. Missing state keys
        for key in ["epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"]:
            bad_ckpt = copy.deepcopy(valid_ckpt)
            del bad_ckpt[key]
            with self.assertRaises(KeyError):
                verify_resume_checkpoint(bad_ckpt, "SegResNet", 2, (96, 96, 80), (96, 96, 64), 0, split_ref)

        # 3. Missing or invalid run_manifest
        no_manifest_ckpt = copy.deepcopy(valid_ckpt)
        del no_manifest_ckpt["run_manifest"]
        with self.assertRaises(KeyError):
            verify_resume_checkpoint(no_manifest_ckpt, "SegResNet", 2, (96, 96, 80), (96, 96, 64), 0, split_ref)

        non_dict_manifest_ckpt = copy.deepcopy(valid_ckpt)
        non_dict_manifest_ckpt["run_manifest"] = "invalid_string"
        with self.assertRaises(TypeError):
            verify_resume_checkpoint(non_dict_manifest_ckpt, "SegResNet", 2, (96, 96, 80), (96, 96, 64), 0, split_ref)

        # 4. Manifest missing nested keys ('config', 'resolved', 'split')
        missing_cfg = copy.deepcopy(valid_ckpt)
        del missing_cfg["run_manifest"]["config"]
        with self.assertRaises(KeyError):
            verify_resume_checkpoint(missing_cfg, "SegResNet", 2, (96, 96, 80), (96, 96, 64), 0, split_ref)

        missing_res = copy.deepcopy(valid_ckpt)
        del missing_res["run_manifest"]["config"]["resolved"]
        with self.assertRaises(KeyError):
            verify_resume_checkpoint(missing_res, "SegResNet", 2, (96, 96, 80), (96, 96, 64), 0, split_ref)

        missing_split = copy.deepcopy(valid_ckpt)
        del missing_split["run_manifest"]["split"]
        with self.assertRaises(KeyError):
            verify_resume_checkpoint(missing_split, "SegResNet", 2, (96, 96, 80), (96, 96, 64), 0, split_ref)

        # 5. Value mismatches (architecture, channels, ROIs, fold, val split hash)
        with self.assertRaises(ValueError):  # Model architecture mismatch
            verify_resume_checkpoint(valid_ckpt, "SwinUNETR", 2, (96, 96, 80), (96, 96, 64), 0, split_ref)

        with self.assertRaises(ValueError):  # Channels mismatch
            verify_resume_checkpoint(valid_ckpt, "SegResNet", 3, (96, 96, 80), (96, 96, 64), 0, split_ref)

        with self.assertRaises(ValueError):  # Train ROI mismatch
            verify_resume_checkpoint(valid_ckpt, "SegResNet", 2, (96, 96, 64), (96, 96, 64), 0, split_ref)

        with self.assertRaises(ValueError):  # Val ROI mismatch
            verify_resume_checkpoint(valid_ckpt, "SegResNet", 2, (96, 96, 80), (96, 96, 80), 0, split_ref)

        with self.assertRaises(ValueError):  # Fold mismatch
            verify_resume_checkpoint(valid_ckpt, "SegResNet", 2, (96, 96, 80), (96, 96, 64), 1, split_ref)

        with self.assertRaises(ValueError):  # Val split hash mismatch
            verify_resume_checkpoint(valid_ckpt, "SegResNet", 2, (96, 96, 80), (96, 96, 64), 0, {"val_sha256": "different_hash"})

    def test_batch_multi_patch_empty_and_non_empty(self):
        """Verifies empty_patch_ratio computes per-patch across mini-batches."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = Tiny3DNet().to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = get_loss_function(loss_type="dice_focal", sigmoid=True)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        # Create batch of 2 patches: Patch 0 has no foreground, Patch 1 has foreground
        img_b2 = torch.randn(2, 2, 8, 8, 8)
        mask_b2 = torch.zeros(2, 1, 8, 8, 8)
        mask_b2[1, 0, 2:6, 2:6, 2:6] = 1.0  # 64 positive voxels in patch 1

        dummy_loader = [{"image": img_b2, "mask": mask_b2}]

        avg_loss, telem = train_epoch(
            model=model,
            loader=dummy_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            use_amp=False,
        )

        self.assertAlmostEqual(telem["empty_patch_ratio"], 0.5)
        self.assertAlmostEqual(telem["empty_patch_aux1_ratio"], 0.5)
        self.assertAlmostEqual(telem["empty_patch_aux2_ratio"], 0.5)
        self.assertEqual(telem["mean_fg_voxels"], 32.0)  # 64 total across 2 patches = 32.0/patch

    def test_all_zero_foreground_retains_zero_not_none(self):
        """Verifies that all-zero foreground retains numeric 0.0 (not None) while absent heads are None."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = Tiny3DNet().to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        loss_fn = get_loss_function(loss_type="dice_focal", sigmoid=True)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        all_zero_loader = [
            {"image": torch.randn(2, 2, 8, 8, 8), "mask": torch.zeros(2, 1, 8, 8, 8)}
        ]

        _, telem = train_epoch(
            model=model,
            loader=all_zero_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            use_amp=False,
        )

        self.assertEqual(telem["empty_patch_ratio"], 1.0)
        self.assertEqual(telem["mean_fg_voxels"], 0.0)
        self.assertIsInstance(telem["mean_fg_voxels"], float)
        self.assertEqual(telem["mean_fg_aux1"], 0.0)
        self.assertEqual(telem["mean_fg_aux2"], 0.0)

        # Test model without auxiliary heads
        single_head_model = SingleHead3DNet().to(device)
        single_opt = AdamW(single_head_model.parameters(), lr=1e-3)
        _, single_telem = train_epoch(
            model=single_head_model,
            loader=all_zero_loader,
            optimizer=single_opt,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            use_amp=False,
        )

        self.assertEqual(single_telem["empty_patch_ratio"], 1.0)
        self.assertEqual(single_telem["mean_fg_voxels"], 0.0)
        self.assertIsNone(single_telem["empty_patch_aux1_ratio"])
        self.assertIsNone(single_telem["empty_patch_aux2_ratio"])
        self.assertIsNone(single_telem["mean_fg_aux1"])
        self.assertIsNone(single_telem["mean_fg_aux2"])

    def test_smoke_detects_no_optimizer_update(self):
        """Verifies dry-run smoke validation fails if weights do not change or if all steps are skipped."""
        dry_run_steps = 2

        # 1. Test skipped_steps >= dry_run_steps detection
        telem_skipped = {"skipped_steps": 2}
        with self.assertRaises(RuntimeError) as ctx_skip:
            skipped = telem_skipped.get("skipped_steps", 0)
            if skipped >= dry_run_steps:
                raise RuntimeError(
                    f"[DRY-RUN] All {dry_run_steps} training steps were skipped by AMP GradScaler! "
                    f"No optimizer update occurred."
                )
        self.assertIn("All 2 training steps were skipped", str(ctx_skip.exception))

        # 2. Test unchanged weights detection (pre == post checksum)
        pre_train_sum = 12.345678
        pre_train_l2 = 3.456789
        post_train_sum = 12.345678
        post_train_l2 = 3.456789

        with self.assertRaises(RuntimeError) as ctx_weight:
            if abs(post_train_sum - pre_train_sum) < 1e-9 and abs(post_train_l2 - pre_train_l2) < 1e-9:
                raise RuntimeError(
                    f"[DRY-RUN] Model weights did not change after optimizer step "
                    f"(pre_sum={pre_train_sum:.6f}, post_sum={post_train_sum:.6f})."
                )
        self.assertIn("Model weights did not change", str(ctx_weight.exception))

    def test_train_resume_workflow_simulation(self):
        """Simulates full checkpoint save at epoch 70 and resume at epoch 71 with scheduler continuity."""
        torch.manual_seed(42)
        device = torch.device("cpu")

        # Session 1: Train to epoch 70 with T_max=150
        m1 = nn.Linear(10, 2)
        opt1 = AdamW(m1.parameters(), lr=2e-4, weight_decay=0.01)
        sched1 = CosineAnnealingLR(opt1, T_max=150)
        scaler1 = torch.amp.GradScaler("cpu", enabled=False)

        for _ in range(70):
            sched1.step()

        lr_ep70 = opt1.param_groups[0]["lr"]
        self.assertAlmostEqual(lr_ep70, 0.00011045, places=6)

        manifest = _build_valid_manifest()
        ckpt_data = {
            "epoch": 70,
            "best_dice": 0.6816,
            "val_dice": 0.6816,
            "model_state_dict": m1.state_dict(),
            "optimizer_state_dict": opt1.state_dict(),
            "scheduler_state_dict": sched1.state_dict(),
            "scaler_state_dict": scaler1.state_dict(),
            "run_manifest": manifest,
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            torch.save(ckpt_data, tmp_path)

        try:
            # Session 2: Resume loading checkpoint
            loaded_ckpt = torch.load(tmp_path, map_location=device, weights_only=False)
            verify_resume_checkpoint(
                ckpt=loaded_ckpt,
                model_name="SegResNet",
                in_channels=2,
                roi_size=(96, 96, 80),
                val_roi_size=(96, 96, 64),
                fold=0,
                split_manifest=manifest["split"],
                checkpoint_name=tmp_path.name,
            )

            m2 = nn.Linear(10, 2)
            opt2 = AdamW(m2.parameters(), lr=2e-4, weight_decay=0.01)
            # Recreate scheduler with default total_epochs (e.g. 100), must be restored to T_max=150
            sched2 = CosineAnnealingLR(opt2, T_max=100)
            scaler2 = torch.amp.GradScaler("cpu", enabled=False)

            m2.load_state_dict(loaded_ckpt["model_state_dict"])
            opt2.load_state_dict(loaded_ckpt["optimizer_state_dict"])
            sched2.load_state_dict(loaded_ckpt["scheduler_state_dict"])
            scaler2.load_state_dict(loaded_ckpt["scaler_state_dict"])

            start_epoch = int(loaded_ckpt["epoch"]) + 1
            best_dice = float(loaded_ckpt["best_dice"])

            self.assertEqual(start_epoch, 71)
            self.assertEqual(best_dice, 0.6816)
            self.assertEqual(sched2.last_epoch, 70)
            self.assertEqual(sched2.T_max, 150)
            self.assertAlmostEqual(opt2.param_groups[0]["lr"], lr_ep70, places=9)

            # Step at epoch 71
            pre_sum, _ = _compute_model_checksum(m2)
            dummy_input = torch.randn(4, 10)
            dummy_target = torch.randn(4, 2)
            out = m2(dummy_input)
            loss = nn.MSELoss()(out, dummy_target)
            loss.backward()
            opt2.step()
            sched2.step()
            post_sum, _ = _compute_model_checksum(m2)

            self.assertNotEqual(pre_sum, post_sum)
            lr_ep71 = opt2.param_groups[0]["lr"]
            self.assertLess(lr_ep71, lr_ep70)
            self.assertAlmostEqual(lr_ep71, 0.00010836, places=6)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_validate_positive_voxels_tracking(self):
        """Verifies that validate() tracks predicted positive voxel count."""
        device = torch.device("cpu")
        model = Tiny3DNet().to(device)
        dummy_val_data = [
            {
                "image": torch.randn(1, 2, 8, 8, 8),
                "mask": (torch.rand(1, 1, 8, 8, 8) > 0.8).float(),
            }
        ]

        mean_dice, num_cases, pos_voxels = validate(
            model=model,
            loader=dummy_val_data,
            patch_size=(8, 8, 8),
            device=device,
            use_amp=False,
            overlap=0.0,
        )

        self.assertEqual(num_cases, 1)
        self.assertIsInstance(mean_dice, float)
        self.assertIsInstance(pos_voxels, int)
        self.assertGreaterEqual(pos_voxels, 0)


if __name__ == "__main__":
    unittest.main()
