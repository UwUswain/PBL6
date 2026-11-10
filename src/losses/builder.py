"""Loss function builder module for ISLES-2022 segmentation."""

import logging
from src.losses.custom_loss import DiceJaccardLoss

logger = logging.getLogger(__name__)


def get_loss_function(
    include_background: bool = False,
    sigmoid: bool = True,
    squared_pred: bool = True,
    dice_weight: float = 0.5,
    jaccard_weight: float = 0.5,
    **kwargs,
) -> DiceJaccardLoss:
    """Builds and returns the SOTA DiceJaccardLoss function (0.5 * Dice + 0.5 * Jaccard).

    Args:
        include_background: Whether to include background in loss calculation.
            Default is False to focus exclusively on lesion foreground.
        sigmoid: Whether to apply sigmoid activation to raw logits.
            Default is True since network outputs raw logits.
        squared_pred: Whether to square denominator predictions in Dice/Jaccard loss.
            Default is True to ensure smoother gradients during backpropagation.
        dice_weight: Weight assigned to Dice loss component (default: 0.5).
        jaccard_weight: Weight assigned to Jaccard loss component (default: 0.5).
        **kwargs: Optional additional keyword arguments passed to DiceLoss/JaccardLoss.

    Returns:
        DiceJaccardLoss: Configured compound loss function.
    """
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
