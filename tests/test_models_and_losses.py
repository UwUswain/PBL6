"""Unit tests for loss functions and model builders."""

import unittest
import torch

from src.losses.builder import get_loss_function
from src.models.builder import build_model, get_model


class TestLossesAndModels(unittest.TestCase):
    """Test suite for loss functions and neural network builders."""

    def test_loss_function_initialization(self) -> None:
        """Verifies DiceJaccardLoss initialization with expected attributes."""
        loss_fn = get_loss_function()
        self.assertFalse(loss_fn.dice_loss.include_background)
        self.assertTrue(loss_fn.dice_loss.sigmoid)
        self.assertTrue(loss_fn.dice_loss.squared_pred)
        self.assertEqual(loss_fn.dice_weight, 0.5)
        self.assertEqual(loss_fn.jaccard_weight, 0.5)

    def test_segresnet_forward_backward(self) -> None:
        """Verifies 3D SegResNet instantiation and gradient flow."""
        model = build_model("segresnet", in_channels=3, out_channels=1)
        loss_fn = get_loss_function()

        dummy_x = torch.randn(1, 3, 32, 32, 16)
        dummy_y = torch.randint(0, 2, (1, 1, 32, 32, 16)).float()

        pred = model(dummy_x)
        self.assertEqual(pred.shape, (1, 1, 32, 32, 16))

        loss = loss_fn(pred, dummy_y)
        loss.backward()
        self.assertFalse(torch.isnan(loss).item())

    def test_mednext_initialization(self) -> None:
        """Verifies MedNeXt Large Kernel instantiation."""
        model = get_model("mednext", in_channels=3, out_channels=1, init_filters=16)
        self.assertEqual(model.__class__.__name__, "MedNeXt")

    def test_swin_unetr_initialization(self) -> None:
        """Verifies SwinUNETR instantiation."""
        model = get_model("swin_unetr", in_channels=3, out_channels=1)
        self.assertEqual(model.__class__.__name__, "SwinUNETR")

    def test_unsupported_model(self) -> None:
        """Verifies ValueError for unknown model names."""
        with self.assertRaises(ValueError):
            get_model("unsupported_model_xyz")


if __name__ == "__main__":
    unittest.main()
