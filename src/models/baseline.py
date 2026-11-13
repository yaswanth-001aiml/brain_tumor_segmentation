import torch
import torch.nn as nn

from .blocks import (
    ConvBlock3D,
    Downsample3D,
    DecoderBlock3D,
    SegmentationHead3D,
)


class Baseline3DUNet(nn.Module):
    """
    E0: Baseline 3D U-Net.

    Supports:
        1. Normal inference
        2. Deep-supervision training

    Input:
        [B, 4, D, H, W]

    Output during normal inference:
        [B, 4, D, H, W]

    Output during deep supervision:
        [
            final_output,
            auxiliary_output_2,
            auxiliary_output_3,
            auxiliary_output_4
        ]
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
        # Decoder
        # ====================================================

        self.decoder4 = DecoderBlock3D(
            decoder_channels=c5,
            skip_channels=c4,
            out_channels=c4,
            block_type="conv"
        )

        self.decoder3 = DecoderBlock3D(
            decoder_channels=c4,
            skip_channels=c3,
            out_channels=c3,
            block_type="conv"
        )

        self.decoder2 = DecoderBlock3D(
            decoder_channels=c3,
            skip_channels=c2,
            out_channels=c2,
            block_type="conv"
        )

        self.decoder1 = DecoderBlock3D(
            decoder_channels=c2,
            skip_channels=c1,
            out_channels=c1,
            block_type="conv"
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