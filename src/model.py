import torch
import torch.nn as nn


# =========================================================
# Basic 3D Convolution Block
# =========================================================

class ConvBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.InstanceNorm3d(out_channels),

            nn.LeakyReLU(
                inplace=True
            ),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.InstanceNorm3d(out_channels),

            nn.LeakyReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# =========================================================
# 3D Baseline U-Net
# =========================================================

class UNet3D(nn.Module):

    def __init__(
        self,
        in_channels=4,
        num_classes=4,
        base_channels=16
    ):

        super().__init__()

        # -----------------------------
        # Encoder
        # -----------------------------

        self.enc1 = ConvBlock3D(
            in_channels,
            base_channels
        )

        self.enc2 = ConvBlock3D(
            base_channels,
            base_channels * 2
        )

        self.enc3 = ConvBlock3D(
            base_channels * 2,
            base_channels * 4
        )

        self.enc4 = ConvBlock3D(
            base_channels * 4,
            base_channels * 8
        )

        # -----------------------------
        # Bottleneck
        # -----------------------------

        self.bottleneck = ConvBlock3D(
            base_channels * 8,
            base_channels * 16
        )

        # -----------------------------
        # Downsampling
        # -----------------------------

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        # -----------------------------
        # Decoder
        # -----------------------------

        self.up4 = nn.ConvTranspose3d(
            base_channels * 16,
            base_channels * 8,
            kernel_size=2,
            stride=2
        )

        self.dec4 = ConvBlock3D(
            base_channels * 16,
            base_channels * 8
        )

        self.up3 = nn.ConvTranspose3d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ConvBlock3D(
            base_channels * 8,
            base_channels * 4
        )

        self.up2 = nn.ConvTranspose3d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ConvBlock3D(
            base_channels * 4,
            base_channels * 2
        )

        self.up1 = nn.ConvTranspose3d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ConvBlock3D(
            base_channels * 2,
            base_channels
        )

        # -----------------------------
        # Final segmentation layer
        # -----------------------------

        self.output = nn.Conv3d(
            base_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # =============================
        # Encoder
        # =============================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # =============================
        # Bottleneck
        # =============================

        b = self.bottleneck(
            self.pool(e4)
        )

        # =============================
        # Decoder
        # =============================

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # =============================
        # Segmentation
        # =============================

        output = self.output(d1)

        return output