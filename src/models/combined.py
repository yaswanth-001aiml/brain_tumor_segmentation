import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ResidualBlock3D,
    MultiScaleBlock3D,
    Downsample3D,
    Upsample3D,
    AttentionGate3D,
    SegmentationHead3D,
)


class CombinedDecoderBlock3D(nn.Module):
    """
    Combined decoder block.

    Combines:
        1. Upsampling
        2. Attention-gated skip connection
        3. Multi-scale feature extraction
        4. Residual refinement
    """

    def __init__(
        self,
        decoder_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        self.up = Upsample3D(
            decoder_channels,
            out_channels
        )

        # ----------------------------------------------------
        # Attention Gate
        # ----------------------------------------------------

        self.attention = AttentionGate3D(
            skip_channels=skip_channels,
            gating_channels=out_channels,
            intermediate_channels=max(out_channels // 2, 1)
        )

        # ----------------------------------------------------
        # Multi-scale feature extraction
        # ----------------------------------------------------

        self.multiscale = MultiScaleBlock3D(
            out_channels + skip_channels,
            out_channels
        )

        # ----------------------------------------------------
        # Residual refinement
        # ----------------------------------------------------

        self.residual = ResidualBlock3D(
            out_channels,
            out_channels
        )

    def forward(self, x, skip):

        # ====================================================
        # Upsampling
        # ====================================================

        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False
            )

        # ====================================================
        # Attention-gated skip connection
        # ====================================================

        attended_skip = self.attention(
            skip,
            x
        )

        # ====================================================
        # Concatenate
        # ====================================================

        x = torch.cat(
            [x, attended_skip],
            dim=1
        )

        # ====================================================
        # Multi-scale feature extraction
        # ====================================================

        x = self.multiscale(x)

        # ====================================================
        # Residual refinement
        # ====================================================

        x = self.residual(x)

        return x


class Combined3DUNet(nn.Module):
    """
    E4: Combined 3D U-Net.

    Combines:

        - Residual Blocks
        - Attention Gates
        - Multi-scale Feature Extraction

    Also supports deep supervision.

    Input:
        [B, 4, D, H, W]

    Normal output:
        [B, 4, D, H, W]

    Deep-supervision output:
        [
            final_output,
            auxiliary_output_2,
            auxiliary_output_3,
            auxiliary_output_4
        ]

    Classes:
        0 = Background
        1 = NCR/NET
        2 = Edema
        3 = Enhancing Tumor
    """

    def __init__(
        self,
        in_channels=4,
        num_classes=4,
        base_channels=32,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # ====================================================
        # Encoder
        # ====================================================

        # Stage 1: Residual
        self.encoder1 = ResidualBlock3D(
            in_channels,
            c1
        )

        self.down1 = Downsample3D(c1)

        # Stage 2: Multi-scale + Residual
        self.encoder2 = MultiScaleBlock3D(
            c1,
            c2
        )

        self.residual2 = ResidualBlock3D(
            c2,
            c2
        )

        self.down2 = Downsample3D(c2)

        # Stage 3: Multi-scale + Residual
        self.encoder3 = MultiScaleBlock3D(
            c2,
            c3
        )

        self.residual3 = ResidualBlock3D(
            c3,
            c3
        )

        self.down3 = Downsample3D(c3)

        # Stage 4: Multi-scale + Residual
        self.encoder4 = MultiScaleBlock3D(
            c3,
            c4
        )

        self.residual4 = ResidualBlock3D(
            c4,
            c4
        )

        self.down4 = Downsample3D(c4)

        # ====================================================
        # Bottleneck
        # ====================================================

        self.bottleneck_multiscale = MultiScaleBlock3D(
            c4,
            c5
        )

        self.bottleneck_residual = ResidualBlock3D(
            c5,
            c5
        )

        # ====================================================
        # Combined Decoder
        # ====================================================

        self.decoder4 = CombinedDecoderBlock3D(
            decoder_channels=c5,
            skip_channels=c4,
            out_channels=c4
        )

        self.decoder3 = CombinedDecoderBlock3D(
            decoder_channels=c4,
            skip_channels=c3,
            out_channels=c3
        )

        self.decoder2 = CombinedDecoderBlock3D(
            decoder_channels=c3,
            skip_channels=c2,
            out_channels=c2
        )

        self.decoder1 = CombinedDecoderBlock3D(
            decoder_channels=c2,
            skip_channels=c1,
            out_channels=c1
        )

        # ====================================================
        # Final Segmentation Head
        # ====================================================

        self.segmentation_head = SegmentationHead3D(
            in_channels=c1,
            num_classes=num_classes
        )

        # ====================================================
        # Deep Supervision Heads
        # ====================================================

        self.ds_head4 = nn.Conv3d(
            c4,
            num_classes,
            kernel_size=1
        )

        self.ds_head3 = nn.Conv3d(
            c3,
            num_classes,
            kernel_size=1
        )

        self.ds_head2 = nn.Conv3d(
            c2,
            num_classes,
            kernel_size=1
        )

    def forward(
        self,
        x,
        deep_supervision=False
    ):

        # ====================================================
        # Encoder
        # ====================================================

        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.down1(e1)
        )

        e2 = self.residual2(e2)

        e3 = self.encoder3(
            self.down2(e2)
        )

        e3 = self.residual3(e3)

        e4 = self.encoder4(
            self.down3(e3)
        )

        e4 = self.residual4(e4)

        # ====================================================
        # Bottleneck
        # ====================================================

        b = self.bottleneck_multiscale(
            self.down4(e4)
        )

        b = self.bottleneck_residual(b)

        # ====================================================
        # Decoder
        # ====================================================

        d4 = self.decoder4(
            b,
            e4
        )

        d3 = self.decoder3(
            d4,
            e3
        )

        d2 = self.decoder2(
            d3,
            e2
        )

        d1 = self.decoder1(
            d2,
            e1
        )

        # ====================================================
        # Final Prediction
        # ====================================================

        final_output = self.segmentation_head(d1)

        # ====================================================
        # Normal inference
        # ====================================================

        if not deep_supervision:
            return final_output

        # ====================================================
        # Auxiliary Predictions
        # ====================================================

        auxiliary_output_2 = self.ds_head2(d2)

        auxiliary_output_3 = self.ds_head3(d3)

        auxiliary_output_4 = self.ds_head4(d4)

        # ====================================================
        # Deep Supervision Outputs
        # ====================================================

        return [
            final_output,
            auxiliary_output_2,
            auxiliary_output_3,
            auxiliary_output_4
        ]