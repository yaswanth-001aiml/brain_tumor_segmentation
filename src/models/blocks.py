import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic Conv3D Block
# ============================================================

class ConvBlock3D(nn.Module):
    """
    Standard 3D convolutional block.

    Conv3D -> InstanceNorm3D -> LeakyReLU
    Conv3D -> InstanceNorm3D -> LeakyReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock3D(nn.Module):
    """
    Residual 3D block.

    Main path:
        Conv3D -> InstanceNorm -> ReLU
        Conv3D -> InstanceNorm

    Shortcut:
        Identity or 1x1x1 Conv3D

    Output:
        Main path + Shortcut -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm1 = nn.InstanceNorm3d(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.InstanceNorm3d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.InstanceNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.LeakyReLU(
            negative_slope=0.01,
            inplace=True
        )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity

        out = self.activation(out)

        return out


# ============================================================
# Attention Gate
# ============================================================

class AttentionGate3D(nn.Module):
    """
    3D Attention Gate.

    Uses:
        - Decoder gating signal
        - Encoder skip feature

    to generate an attention map that suppresses
    irrelevant encoder features.
    """

    def __init__(
        self,
        skip_channels,
        gating_channels,
        intermediate_channels
    ):
        super().__init__()

        self.theta = nn.Conv3d(
            skip_channels,
            intermediate_channels,
            kernel_size=2,
            stride=2,
            bias=False
        )

        self.phi = nn.Conv3d(
            gating_channels,
            intermediate_channels,
            kernel_size=1,
            bias=False
        )

        self.psi = nn.Conv3d(
            intermediate_channels,
            1,
            kernel_size=1,
            bias=True
        )

        self.norm_theta = nn.InstanceNorm3d(
            intermediate_channels
        )

        self.norm_phi = nn.InstanceNorm3d(
            intermediate_channels
        )

        self.norm_psi = nn.InstanceNorm3d(1)

        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, skip, gating):

        theta_x = self.theta(skip)
        theta_x = self.norm_theta(theta_x)

        phi_g = self.phi(gating)
        phi_g = self.norm_phi(phi_g)

        # Make spatial dimensions equal
        if theta_x.shape[2:] != phi_g.shape[2:]:
            phi_g = F.interpolate(
                phi_g,
                size=theta_x.shape[2:],
                mode="trilinear",
                align_corners=False
            )

        attention = theta_x + phi_g
        attention = self.relu(attention)

        attention = self.psi(attention)

        # Resize attention map to skip feature resolution
        attention = F.interpolate(
            attention,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        attention = self.sigmoid(attention)

        return skip * attention


# ============================================================
# Multi-Scale Feature Extraction Block
# ============================================================

class MultiScaleBlock3D(nn.Module):
    """
    Multi-scale 3D feature extraction.

    Parallel branches:

        Branch 1 -> 1x1x1 Conv
        Branch 2 -> 3x3x3 Conv
        Branch 3 -> 5x5 receptive field using two 3x3 Conv layers
        Branch 4 -> Dilated 3x3 Conv

    The features are concatenated and fused.
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        branch_channels = out_channels // 4

        # Prevent zero channels for very small configurations
        branch_channels = max(branch_channels, 1)

        self.branch1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                branch_channels,
                kernel_size=1,
                bias=False
            ),
            nn.InstanceNorm3d(branch_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(branch_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(branch_channels),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv3d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(branch_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False
            ),
            nn.InstanceNorm3d(branch_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        total_channels = branch_channels * 4

        self.fusion = nn.Sequential(
            nn.Conv3d(
                total_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        features = torch.cat(
            [b1, b2, b3, b4],
            dim=1
        )

        return self.fusion(features)


# ============================================================
# Downsampling Block
# ============================================================

class Downsample3D(nn.Module):
    """
    3D downsampling using strided convolution.
    """

    def __init__(self, channels):

        super().__init__()

        self.downsample = nn.Conv3d(
            channels,
            channels,
            kernel_size=2,
            stride=2,
            bias=False
        )

    def forward(self, x):
        return self.downsample(x)


# ============================================================
# Upsampling Block
# ============================================================

class Upsample3D(nn.Module):
    """
    3D transposed convolution for decoder upsampling.
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        return self.up(x)


# ============================================================
# Decoder Fusion Block
# ============================================================

class DecoderBlock3D(nn.Module):
    """
    Decoder block.

    Upsampling
        +
    Skip connection
        +
    Feature refinement
    """

    def __init__(
        self,
        decoder_channels,
        skip_channels,
        out_channels,
        block_type="conv"
    ):

        super().__init__()

        self.up = Upsample3D(
            decoder_channels,
            out_channels
        )

        total_channels = out_channels + skip_channels

        if block_type == "residual":

            self.refine = ResidualBlock3D(
                total_channels,
                out_channels
            )

        elif block_type == "multiscale":

            self.refine = MultiScaleBlock3D(
                total_channels,
                out_channels
            )

        else:

            self.refine = ConvBlock3D(
                total_channels,
                out_channels
            )

    def forward(self, x, skip):

        x = self.up(x)

        # Handle possible size differences caused by
        # odd input dimensions.
        if x.shape[2:] != skip.shape[2:]:

            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False
            )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.refine(x)

        return x


# ============================================================
# Segmentation Head
# ============================================================

class SegmentationHead3D(nn.Module):
    """
    Converts decoder features into class logits.
    """

    def __init__(
        self,
        in_channels,
        num_classes
    ):

        super().__init__()

        self.head = nn.Conv3d(
            in_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        return self.head(x)