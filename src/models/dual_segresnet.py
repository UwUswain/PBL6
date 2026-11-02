"""3D Dual-Encoder SegResNet for Multimodal Ischemic Stroke Lesion Segmentation.

Inspired by the multimodal feature disentanglement principle from:
Usman et al. (Paper 5) - "Dual-Encoder Transformer-Based Multimodal Learning
for Ischemic Stroke Lesion Segmentation Using Diffusion MRI".

Extends the single-encoder SegResNet baseline into a dual-branch 3D volumetric
architecture that independently models:
1. DWI stream: Captures restricted water diffusion hyperintensity.
2. ADC stream: Captures quantitative apparent diffusion coefficient hypointensity.

Modality-specific features are hierarchically fused across all resolution scales
via 1x1x1 convolutional bottleneck projections with residual skip connections.
"""

from typing import Sequence, Tuple
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet


class DualEncoderSegResNet(nn.Module):
    """3D Dual-Branch SegResNet for independent DWI and ADC feature extraction.

    Args:
        in_channels: Total input channels (must be 2: channel 0 = DWI, channel 1 = ADC).
        out_channels: Number of segmentation output channels (default: 1 for binary stroke mask).
        init_filters: Number of base filters (default: 16).
        blocks_down: Number of convolutional blocks per downsampling level (default: (1, 2, 2, 4)).
        blocks_up: Number of convolutional blocks per upsampling level (default: (1, 1, 1)).
        dropout_prob: Dropout probability for regularization (default: 0.2).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        init_filters: int = 16,
        blocks_down: Sequence[int] = (1, 2, 2, 4),
        blocks_up: Sequence[int] = (1, 1, 1),
        dropout_prob: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        if in_channels != 2:
            raise ValueError(f"DualEncoderSegResNet requires exactly 2 input channels [DWI, ADC], got {in_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Branch 1: Dedicated 3D Encoder & Decoder backbone (DWI stream)
        self.seg_dwi = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob,
            **kwargs,
        )

        # Branch 2: Dedicated 3D Encoder (ADC stream)
        self.seg_adc = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob,
            **kwargs,
        )

        # Multimodal fusion projections at each downsampling resolution scale
        # Level i has channels = init_filters * (2 ** i)
        self.fusion_convs = nn.ModuleList()
        for i in range(len(blocks_down)):
            ch = init_filters * (2 ** i)
            # Concatenates DWI (ch) + ADC (ch) -> 1x1x1 Conv -> ch
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv3d(ch * 2, ch, kernel_size=1, bias=False),
                    nn.InstanceNorm3d(ch),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dual encoders, multi-scale fusion, and shared decoder.

        Args:
            x: Input tensor of shape [B, 2, D, H, W] where:
               x[:, 0:1, ...] is DWI
               x[:, 1:2, ...] is ADC

        Returns:
            Output logits tensor of shape [B, 1, D, H, W]
        """
        x_dwi = x[:, 0:1, ...]
        x_adc = x[:, 1:2, ...]

        # 1. Initial 3D convolutions for each modality
        x_d = self.seg_dwi.convInit(x_dwi)
        x_a = self.seg_adc.convInit(x_adc)

        # 2. Hierarchical multi-scale dual encoding with fusion
        down_features = []
        for layer_d, layer_a, fuse in zip(
            self.seg_dwi.down_layers,
            self.seg_adc.down_layers,
            self.fusion_convs,
        ):
            x_d = layer_d(x_d)
            x_a = layer_a(x_a)

            # Fuse complementary DWI and ADC representations
            fused = fuse(torch.cat([x_d, x_a], dim=1))
            # Residual injection into primary feature stream
            x_d = x_d + fused
            down_features.append(x_d)

        # 3. 3D Decoder with fused skip connections
        x_dec = down_features[-1]
        for i, (up_sample, up_layer) in enumerate(
            zip(self.seg_dwi.up_samples, self.seg_dwi.up_layers)
        ):
            x_dec = up_sample(x_dec)
            skip = down_features[-(i + 2)]
            x_dec = x_dec + skip
            x_dec = up_layer(x_dec)

        # 4. Final segmentation head
        out = self.seg_dwi.conv_final(x_dec)
        return out
