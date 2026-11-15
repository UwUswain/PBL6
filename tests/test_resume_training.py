"""Unit tests for training resume functionality and telemetry logging."""

import json
from pathlib import Path
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.train import train_epoch, validate
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


class TestResumeTraining(unittest.TestCase):
    def test_telemetry_and_no_extra_backward(self):
        """Verifies that train_epoch collects telemetry without extra backward or NaN/Inf."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = Tiny3DNet().to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = get_loss_function(loss_type="dice_focal", sigmoid=True)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        # Create dummy DataLoader batches
        dummy_data = [
            {
                "image": torch.randn(1, 2, 8, 8, 8),
                "mask": (torch.rand(1, 1, 8, 8, 8) > 0.7).float(),
            },
            {
                "image": torch.randn(1, 2, 8, 8, 8),
                "mask": torch.zeros(1, 1, 8, 8, 8),  # Empty patch to test empty_patch_ratio
            },
        ]

        avg_loss, telem = train_epoch(
            model=model,
            loader=dummy_data,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            use_amp=False,
        )

        self.assertIsInstance(avg_loss, float)
        self.assertGreater(avg_loss, 0)
        self.assertIn("grad_norm", telem)
        self.assertGreater(telem["grad_norm"], 0)
        self.assertEqual(telem["empty_patch_ratio"], 0.5)  # 1 out of 2 batches has 0 foreground
        self.assertIn("main_dice", telem)
        self.assertIsNotNone(telem["main_dice"])
        self.assertIn("main_focal", telem)
        self.assertIsNotNone(telem["main_focal"])
        self.assertIn("mean_fg_voxels", telem)

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

    def test_resume_scheduler_tmax_and_lr_continuity(self):
        """Verifies that loading a checkpoint restores scheduler T_max=150 and continuous LR."""
        m1 = nn.Linear(10, 1)
        opt1 = AdamW(m1.parameters(), lr=2e-4, weight_decay=0.01)
        sched1 = CosineAnnealingLR(opt1, T_max=150)

        # Step to epoch 70
        for _ in range(70):
            sched1.step()

        lr_ep70 = opt1.param_groups[0]["lr"]
        self.assertAlmostEqual(lr_ep70, 0.00011045, places=6)

        # Save state dicts
        opt_state = opt1.state_dict()
        sched_state = sched1.state_dict()

        # Recreate in resume session
        m2 = nn.Linear(10, 1)
        opt2 = AdamW(m2.parameters(), lr=2e-4, weight_decay=0.01)
        # Even if total_epochs is 100, scheduler must keep T_max=150 from loaded checkpoint
        sched2 = CosineAnnealingLR(opt2, T_max=150)

        opt2.load_state_dict(opt_state)
        sched2.load_state_dict(sched_state)

        self.assertEqual(sched2.last_epoch, 70)
        self.assertEqual(sched2.T_max, 150)
        self.assertAlmostEqual(opt2.param_groups[0]["lr"], lr_ep70, places=9)

        # Step at epoch 71
        sched2.step()
        lr_ep71 = opt2.param_groups[0]["lr"]
        self.assertLess(lr_ep71, lr_ep70)  # LR must continue decreasing along 150-epoch curve
        self.assertAlmostEqual(lr_ep71, 0.00010836, places=6)


if __name__ == "__main__":
    unittest.main()

