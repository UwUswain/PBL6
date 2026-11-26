"""3D Dual-Encoder SegResNet with Bidirectional Cross-Modality Attention Fusion & Deep Supervision (V3).

Architectural Innovations:
1. Multimodal Feature Disentanglement (Paper 5 - Usman et al.):
   - Independent 3D encoders for DWI (restricted diffusion hyperintensity)
     and ADC (quantitative diffusion coefficient hypointensity).
2. Bidirectional Cross-Modality Attention (BiCrossAttention):
   - Modality-specific reciprocal recalibration via concurrent 3D scSE (Roy et al.).
   - DWI is guided by ADC (diffusion coefficient hypointensity suppressing T2 shine-through).
   - ADC is guided by DWI (hyperintensity providing sharp spatial localization).
3. Parameter-Efficient Backbone:
   - Unused ADC decoder structures are cleanly pruned to maximize parameter efficiency.
4. Multi-Scale Deep Supervision (Paper 32 - Factorizer, Eq. 15):
   - Auxiliary segmentation heads at intermediate decoder resolutions (1/2, 1/4)
     to enforce strong gradient flow across deep volumetric pathways.
"""

from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet


class ChannelSpatialSELayer3D(nn.Module):
    """3D Concurrent Spatial and Channel Squeeze & Excitation (scSE) block.

    Reference:
        Roy et al., 'Concurrent Spatial and Channel 'Squeeze & Excitation'
        in Fully Convolutional Networks', IEEE TMI 2018.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_ch = max(1, channels // reduction)

        # Channel Squeeze-and-Excitation (cSE)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.cse = nn.Sequential(
            nn.Conv3d(channels, reduced_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_ch, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Spatial Squeeze-and-Excitation (sSE)
        self.sse = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies concurrent cSE and sSE calibration to the input volume."""
        cse_weight = self.cse(self.avg_pool(x))
        x_cse = x * cse_weight

        sse_weight = self.sse(x)
        x_sse = x * sse_weight

        return x_cse + x_sse


class CrossModalityAttentionFusion(nn.Module):
    """Unidirectional Cross-Modality Attention Fusion block for DWI and ADC (V2 compatibility).

    Fuses complementary modalities via:
    1. Channel-preserving 1x1x1 bottleneck projection with Instance Normalization.
    2. Concurrent 3D scSE attention recalibration.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.conv_proj = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.scse = ChannelSpatialSELayer3D(channels, reduction=reduction)

    def forward(self, dwi_feat: torch.Tensor, adc_feat: torch.Tensor) -> torch.Tensor:
        """Fuses DWI and ADC features into a single recalibrated feature map."""
        concat_feat = torch.cat([dwi_feat, adc_feat], dim=1)
        proj_feat = self.conv_proj(concat_feat)
        attn_feat = self.scse(proj_feat)
        return attn_feat


class BiCrossAttentionFusion3D(nn.Module):
    """3D Bidirectional Cross-Modality Attention Fusion block for DWI and ADC (V3 Innovation).

    Fuses complementary modalities reciprocally:
    1. Joint channel-preserving 1x1x1 bottleneck projection with Instance Normalization.
    2. Concurrent 3D scSE (channel and spatial) attention calibration.
    3. Bidirectional feature recalibration:
       - Modality-specific residual projections update both DWI and ADC pathways.
       - DWI is guided by ADC (diffusion coefficient hypointensity suppressing T2 shine-through).
       - ADC is guided by DWI (hyperintensity providing sharp spatial localization).
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.conv_proj = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.scse = ChannelSpatialSELayer3D(channels, reduction=reduction)

        # Modality-specific projection heads for reciprocal updates
        self.proj_dwi = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
        )
        self.proj_adc = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
        )

    def forward(
        self, dwi_feat: torch.Tensor, adc_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reciprocally fuses DWI and ADC features.

        Args:
            dwi_feat: DWI feature tensor [B, C, D, H, W].
            adc_feat: ADC feature tensor [B, C, D, H, W].

        Returns:
            Tuple of recalibrated (dwi_out, adc_out) tensors, each [B, C, D, H, W].
        """
        concat_feat = torch.cat([dwi_feat, adc_feat], dim=1)
        proj_feat = self.conv_proj(concat_feat)
        calibrated = self.scse(proj_feat)

        dwi_out = dwi_feat + self.proj_dwi(calibrated)
        adc_out = adc_feat + self.proj_adc(calibrated)
        return dwi_out, adc_out


class DualEncoderSegResNet(nn.Module):
    """3D Dual-Branch SegResNet with BiCrossAttention and Deep Supervision (V3).

    Args:
        in_channels: Total input channels (must be 2: channel 0 = DWI, channel 1 = ADC).
        out_channels: Number of output channels (default: 1 for binary stroke mask).
        init_filters: Number of base filters (default: 32 for SOTA capacity, or 16).
        blocks_down: Convolutional blocks per downsampling level (default: (1, 2, 2, 4)).
        blocks_up: Convolutional blocks per upsampling level (default: (1, 1, 1)).
        dropout_prob: Dropout probability for regularization (default: 0.2).
        deep_supervision: Whether to output intermediate auxiliary logits during training (default: True).
        fusion_mode: Fusion mode ('bidirectional' for V3 or 'unidirectional' for V2).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        init_filters: int = 32,
        blocks_down: Sequence[int] = (1, 2, 2, 4),
        blocks_up: Sequence[int] = (1, 1, 1),
        dropout_prob: float = 0.2,
        deep_supervision: bool = True,
        fusion_mode: str = "bidirectional",
        **kwargs,
    ) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError(
                f"DualEncoderSegResNet requires exactly 2 input channels [DWI, ADC], got {in_channels}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision
        self.fusion_mode = fusion_mode.lower().strip()

        # Branch 1: Dedicated 3D DWI backbone (Encoder + Decoder structure)
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

        # Branch 2: Dedicated 3D ADC encoder backbone
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

        # Prune unused ADC decoder structures to optimize parameter count and VRAM
        del self.seg_adc.up_samples
        del self.seg_adc.up_layers
        del self.seg_adc.conv_final
        self.seg_adc.up_samples = nn.ModuleList()
        self.seg_adc.up_layers = nn.ModuleList()
        self.seg_adc.conv_final = nn.Identity()

        # Multi-scale cross-modality attention fusion modules
        self.fusion_modules = nn.ModuleList()
        for i in range(len(blocks_down)):
            ch = init_filters * (2 ** i)
            if self.fusion_mode == "bidirectional":
                self.fusion_modules.append(BiCrossAttentionFusion3D(ch))
            else:
                self.fusion_modules.append(CrossModalityAttentionFusion(ch))

        # Deep supervision auxiliary heads
        if self.deep_supervision:
            # Stage 0: 1/4 resolution (channels = init_filters * 4)
            ch_stage0 = init_filters * 4
            self.aux_head_stage0 = nn.Conv3d(ch_stage0, out_channels, kernel_size=1)

            # Stage 1: 1/2 resolution (channels = init_filters * 2)
            ch_stage1 = init_filters * 2
            self.aux_head_stage1 = nn.Conv3d(ch_stage1, out_channels, kernel_size=1)
        else:
            self.aux_head_stage0 = None
            self.aux_head_stage1 = None

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through dual encoders, BiCrossAttention, and shared decoder.

        Args:
            x: Input tensor [B, 2, D, H, W] where channel 0 is DWI and channel 1 is ADC.

        Returns:
            In training mode (if deep_supervision=True):
                Tuple of (main_logits, aux_stage1_logits, aux_stage0_logits).
            In eval/inference mode:
                Main logits tensor [B, 1, D, H, W].
        """
        x_dwi = x[:, 0:1, ...]
        x_adc = x[:, 1:2, ...]

        # 1. Modality-specific initial 3D convolutions
        x_d = self.seg_dwi.convInit(x_dwi)
        x_a = self.seg_adc.convInit(x_adc)

        # 2. Hierarchical dual encoding with BiCrossAttention fusion
        down_features = []
        for layer_d, layer_a, fuse_mod in zip(
            self.seg_dwi.down_layers,
            self.seg_adc.down_layers,
            self.fusion_modules,
        ):
            x_d = layer_d(x_d)
            x_a = layer_a(x_a)

            if self.fusion_mode == "bidirectional":
                x_d, x_a = fuse_mod(x_d, x_a)
                # Symmetrically balanced representation for decoder skip connections
                down_features.append(0.5 * (x_d + x_a))
            else:
                fused = fuse_mod(x_d, x_a)
                x_d = x_d + fused
                down_features.append(x_d)

        # 3. 3D Decoder with multi-resolution intermediate taps
        x_dec = down_features[-1]
        aux_stage0 = None
        aux_stage1 = None

        for i, (up_sample, up_layer) in enumerate(
            zip(self.seg_dwi.up_samples, self.seg_dwi.up_layers)
        ):
            x_dec = up_sample(x_dec)
            skip = down_features[-(i + 2)]
            x_dec = x_dec + skip
            x_dec = up_layer(x_dec)

            # Collect auxiliary logits for deep supervision
            if self.deep_supervision and self.training:
                if i == 0:  # 1/4 resolution scale
                    aux_stage0 = self.aux_head_stage0(x_dec)
                elif i == 1:  # 1/2 resolution scale
                    aux_stage1 = self.aux_head_stage1(x_dec)

        # 4. Final full-resolution segmentation head
        out = self.seg_dwi.conv_final(x_dec)

        if self.deep_supervision and self.training:
            return out, aux_stage1, aux_stage0

        return out

