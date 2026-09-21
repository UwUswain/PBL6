"""Loss function builder module for ISLES-2022 segmentation."""

import logging
from typing import Optional
from monai.losses import DiceFocalLoss

logger = logging.getLogger(__name__)


def get_loss_function(
    include_background: bool = False,
    sigmoid: bool = True,
    squared_pred: bool = True,
    **kwargs,
) -> DiceFocalLoss:
    """Builds and configures the MONAI DiceFocalLoss function.

    Args:
        include_background: Whether to include background in loss calculation.
            Default is False to focus exclusively on lesion foreground.
        sigmoid: Whether to apply sigmoid activation to raw logits.
            Default is True since network outputs raw logits.
        squared_pred: Whether to square denominator predictions in Dice loss.
            Default is True to ensure smoother gradients during backpropagation.
        **kwargs: Optional additional keyword arguments passed to DiceFocalLoss.

    Returns:
        DiceFocalLoss: Configured MONAI loss function.
    """
    logger.info(
        "Initializing DiceFocalLoss (include_background=%s, sigmoid=%s, squared_pred=%s)",
        include_background,
        sigmoid,
        squared_pred,
    )
    return DiceFocalLoss(
        include_background=include_background,
        sigmoid=sigmoid,
        squared_pred=squared_pred,
        **kwargs,
    )
