"""Custom loss functions for ISLES-2022 stroke lesion segmentation.

Implements SOTA compound losses according to recent benchmark literature (arXiv:2501.02287).
"""

import logging
import torch
import torch.nn as nn
from monai.losses import DiceLoss

logger = logging.getLogger(__name__)


class JaccardLoss(DiceLoss):
    """MONAI-based Jaccard (IoU) Loss."""

    def __init__(self, **kwargs):
        kwargs["jaccard"] = True
        super().__init__(**kwargs)


class DiceJaccardLoss(nn.Module):
    """Compound loss combining DiceLoss and JaccardLoss (IoU Loss) with equal weighting (0.5 / 0.5).

    Formula:
        Loss = 0.5 * DiceLoss + 0.5 * JaccardLoss
    """

    def __init__(
        self,
        include_background: bool = False,
        sigmoid: bool = True,
        squared_pred: bool = True,
        dice_weight: float = 0.5,
        jaccard_weight: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.jaccard_weight = jaccard_weight

        self.dice_loss = DiceLoss(
            include_background=include_background,
            sigmoid=sigmoid,
            squared_pred=squared_pred,
            jaccard=False,
            **kwargs,
        )
        self.jaccard_loss = JaccardLoss(
            include_background=include_background,
            sigmoid=sigmoid,
            squared_pred=squared_pred,
            **kwargs,
        )

        logger.info(
            "Initialized DiceJaccardLoss: %.1f * DiceLoss + %.1f * JaccardLoss "
            "(include_background=%s, sigmoid=%s, squared_pred=%s)",
            self.dice_weight,
            self.jaccard_weight,
            include_background,
            sigmoid,
            squared_pred,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes combined 0.5 * Dice + 0.5 * Jaccard loss."""
        dice = self.dice_loss(pred, target)
        jaccard = self.jaccard_loss(pred, target)
        return self.dice_weight * dice + self.jaccard_weight * jaccard
