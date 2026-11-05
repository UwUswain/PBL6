"""Loss function builder module for ISLES-2022 segmentation.

Supports:
1. Paper 32 standard: DiceCELoss (Soft Dice Loss + Cross-Entropy Loss, Eq. 16).
2. Compound DiceJaccardLoss (0.5 * Dice + 0.5 * Jaccard).
"""

import logging
from typing import Union
import torch.nn as nn
from monai.losses import DiceCELoss
from src.losses.custom_loss import DiceJaccardLoss

logger = logging.getLogger(__name__)


def get_loss_function(
    loss_type: str = "dice_ce",
    include_background: bool = False,
    sigmoid: bool = True,
    squared_pred: bool = False,
    dice_weight: float = 0.5,
    jaccard_weight: float = 0.5,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    **kwargs,
) -> nn.Module:
    """Builds and returns the configured loss function.

    Args:
        loss_type: Type of loss function ('dice_ce' for Paper 32 or 'dice_jaccard').
        include_background: Whether to include background in loss calculation.
        sigmoid: Whether to apply sigmoid activation to raw logits.
        squared_pred: Whether to square denominator predictions in Dice loss.
        dice_weight: Weight assigned to Dice loss component (for DiceJaccardLoss).
        jaccard_weight: Weight assigned to Jaccard loss component (for DiceJaccardLoss).
        smooth_nr: Numerator smoothing constant (Paper 32 Eq. 17 epsilon = 1e-5).
        smooth_dr: Denominator smoothing constant (Paper 32 Eq. 17 epsilon = 1e-5).
        **kwargs: Optional additional keyword arguments passed to the loss module.

    Returns:
        Configured PyTorch loss module.
    """
    normalized_type = loss_type.lower().replace("-", "_")

    if normalized_type in ("dice_ce", "diceceloss", "paper32"):
        logger.info(
            "Initializing Paper 32 DiceCELoss (sigmoid=%s, squared_pred=%s, smooth=%.1e)",
            sigmoid,
            squared_pred,
            smooth_nr,
        )
        return DiceCELoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            **kwargs,
        )

    if normalized_type in ("dice_jaccard", "dicejaccardloss"):
        logger.info(
            "Initializing DiceJaccardLoss (include_background=%s, sigmoid=%s, squared_pred=%s)",
            include_background,
            sigmoid,
            squared_pred,
        )
        return DiceJaccardLoss(
            include_background=include_background,
            sigmoid=sigmoid,
            squared_pred=squared_pred,
            dice_weight=dice_weight,
            jaccard_weight=jaccard_weight,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported loss_type: '{loss_type}'. Expected 'dice_ce' or 'dice_jaccard'."
    )
