from .blocks import (
    ConvBlock3D,
    ResidualBlock3D,
    AttentionGate3D,
    MultiScaleBlock3D,
    Downsample3D,
    Upsample3D,
    DecoderBlock3D,
    SegmentationHead3D,
)
from .baseline import Baseline3DUNet
from .residual import Residual3DUNet
from .attention import Attention3DUNet
from .multiscale import MultiScale3DUNet
from .combined import Combined3DUNet

MODEL_REGISTRY = {
    "baseline": Baseline3DUNet,
    "residual": Residual3DUNet,
    "attention": Attention3DUNet,
    "multiscale": MultiScale3DUNet,
    "combined": Combined3DUNet,
}


def get_model(name="baseline", in_channels=4, num_classes=4, base_channels=32, **kwargs):
    """
    Model factory to instantiate any model architecture by name.
    Supported names: 'baseline', 'residual', 'attention', 'multiscale', 'combined'.
    """
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture: '{name}'. "
            f"Available architectures: {list(MODEL_REGISTRY.keys())}"
        )
    model_cls = MODEL_REGISTRY[key]
    return model_cls(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        **kwargs
    )