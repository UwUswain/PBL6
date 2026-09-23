"""Model architecture builder for ISLES-2022 stroke lesion segmentation."""

import logging
import torch.nn as nn
from monai.networks.nets import MedNeXt, SegResNet, SwinUNETR

logger = logging.getLogger(__name__)


def get_model(
    model_name: str = "segresnet",
    in_channels: int = 3,
    out_channels: int = 1,
    **kwargs,
) -> nn.Module:
    """Builds and returns the requested 3D segmentation network.

    Args:
        model_name: Name of the architecture ('segresnet', 'mednext', 'swin_unetr').
        in_channels: Number of input MRI modalities (default: 3 for DWI, ADC, FLAIR).
        out_channels: Number of segmentation output channels (default: 1 for binary lesion mask).
        **kwargs: Additional hyperparameters passed to the model constructor.

    Returns:
        nn.Module: Instantiated PyTorch/MONAI model.

    Raises:
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
    elif normalized_name == "mednext":
        logger.info(
            "Building MedNeXt 3D Large Kernel (in_channels=%d, out_channels=%d)",
            in_channels,
            out_channels,
        )
        kernel_size = kwargs.pop("kernel_size", 7)
        return MedNeXt(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )
    elif normalized_name == "swin_unetr":
        logger.info(
            "Building SwinUNETR 3D (in_channels=%d, out_channels=%d)",
            in_channels,
            out_channels,
        )
        return SwinUNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported model_name: '{model_name}'. Available options: 'segresnet', 'mednext', 'swin_unetr'."
        )


# Backward-compatible alias
build_model = get_model
