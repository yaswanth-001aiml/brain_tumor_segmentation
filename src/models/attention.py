import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ConvBlock3D,
    Downsample3D,
    Upsample3D,
    AttentionGate3D,
    SegmentationHead3D,
)


class AttentionDecoderBlock3D(nn.Module):
    """
    Decoder block with an attention gate.

    Flow:
        Decoder feature
             ↓
         Upsampling
             ↓
       Attention Gate ← Encoder skip
             ↓
       Concatenation
             ↓
       Conv refinement
    """

    def __init__(
        self,
        decoder_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        self.up = Upsample3D(
            decoder_channels,
            out_channels
        )

        self.attention = AttentionGate3D(
            skip_channels=skip_channels,
            gating_channels=out_channels,
            intermediate_channels=max(out_channels // 2, 1)
        )

        self.refine = ConvBlock3D(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        # ====================================================
        # Upsampling
        # ====================================================

        x = self.up(x)

        # Match spatial dimensions if necessary
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
        # Feature refinement
        # ====================================================

        x = self.refine(x)

        return x


class Attention3DUNet(nn.Module):
    """
    E2: Attention 3D U-Net.

    Attention gates are applied to encoder skip
    connections.

    Supports:
        - Normal inference
        - Deep-supervision training

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

        self.encoder1 = ConvBlock3D(
            in_channels,
            c1
        )

        self.down1 = Downsample3D(c1)

        self.encoder2 = ConvBlock3D(
            c1,
            c2
        )

        self.down2 = Downsample3D(c2)

        self.encoder3 = ConvBlock3D(
            c2,
            c3
        )

        self.down3 = Downsample3D(c3)

        self.encoder4 = ConvBlock3D(
            c3,
            c4
        )

        self.down4 = Downsample3D(c4)

        # ====================================================
        # Bottleneck
        # ====================================================

        self.bottleneck = ConvBlock3D(
            c4,
            c5
        )

        # ====================================================
        # Attention Decoder
        # ====================================================

        self.decoder4 = AttentionDecoderBlock3D(
            decoder_channels=c5,
            skip_channels=c4,
            out_channels=c4
        )

        self.decoder3 = AttentionDecoderBlock3D(
            decoder_channels=c4,
            skip_channels=c3,
            out_channels=c3
        )

        self.decoder2 = AttentionDecoderBlock3D(
            decoder_channels=c3,
            skip_channels=c2,
            out_channels=c2
        )

        self.decoder1 = AttentionDecoderBlock3D(
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

        e3 = self.encoder3(
            self.down2(e2)
        )

        e4 = self.encoder4(
            self.down3(e3)
        )

        # ====================================================
        # Bottleneck
        # ====================================================

        b = self.bottleneck(
            self.down4(e4)
        )

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