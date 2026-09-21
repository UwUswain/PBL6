"""Model architecture builder for ISLES-2022 stroke lesion segmentation."""

import logging
import torch.nn as nn
from monai.networks.nets import SegResNet

logger = logging.getLogger(__name__)


def build_model(
    model_name: str = "segresnet",
    in_channels: int = 3,
    out_channels: int = 1,
    **kwargs,
) -> nn.Module:
    """Builds and returns the requested 3D segmentation network.

    Args:
        model_name: Name of the architecture ('segresnet' or 'mednext').
        in_channels: Number of input MRI modalities (default: 3 for DWI, ADC, FLAIR).
        out_channels: Number of segmentation output channels (default: 1 for binary lesion mask).
        **kwargs: Additional hyperparameters passed to the model constructor.

    Returns:
        nn.Module: Instantiated PyTorch/MONAI model.

    Raises:
        NotImplementedError: If model_name is 'mednext' (pending custom integration).
        ValueError: If model_name is not supported.
    """
    normalized_name = model_name.lower().strip()

    if normalized_name == "segresnet":
        logger.info(
            "Building SegResNet 3D (in_channels=%d, out_channels=%d)",
            in_channels,
            out_channels,
        )
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )

    if normalized_name == "mednext":
        message = "MedNeXt sẽ được tích hợp custom module ở chặng sau"
        logger.warning(message)
        raise NotImplementedError(message)

    raise ValueError(
        f"Unsupported model_name: '{model_name}'. Available options: 'segresnet', 'mednext'."
    )
