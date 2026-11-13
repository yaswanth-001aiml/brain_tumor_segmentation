import argparse
import torch
import torch.nn as nn

from src.models import get_model, MODEL_REGISTRY
from src.losses import CombinedLoss
from src.training.deep_supervision import DeepSupervisionLossWrapper
from src.model import UNet3D as LegacyUNet3D


def parse_args():
    parser = argparse.ArgumentParser(description="Test and Verify 3D BraTS Model Architectures")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "legacy"] + list(MODEL_REGISTRY.keys()),
        help="Model architecture to test (default: all)"
    )
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size for test forward pass")
    parser.add_argument("--base_channels", type=int, default=8, help="Base channel count for test")
    return parser.parse_args()


def test_single_model(name, model, patch_size=32, device="cpu", test_ds=True):
    model = model.to(device)
    ps = patch_size
    x = torch.randn(1, 4, ps, ps, ps, device=device)
    target = torch.randint(0, 4, (1, ps, ps, ps), device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f" Testing Architecture: {name.upper()}")
    print(f"{'='*60}")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")
    print(f"  Model Size (Weights) : {model_size_mb:.2f} MB")
    print(f"  Input Tensor Shape   : {list(x.shape)}")

    # 1. Standard Forward Pass
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
    assert out.shape == (1, 4, ps, ps, ps), f"Expected shape (1, 4, {ps}, {ps}, {ps}), got {out.shape}"
    print(f"  Standard Forward Pass: SUCCESS -> Output Shape: {list(out.shape)}")

    # 2. Backward Pass & Gradient Flow
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CombinedLoss()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    # Check for NaN or uncomputed gradients
    grad_ok = any(p.grad is not None and not torch.isnan(p.grad).any() for p in model.parameters())
    assert grad_ok, "Backward pass failed to compute valid gradients!"
    optimizer.step()
    print(f"  Backward Pass Flow   : SUCCESS -> Loss: {loss.item():.4f}")

    # 3. Deep Supervision (if supported)
    if test_ds:
        optimizer.zero_grad()
        ds_wrapper = DeepSupervisionLossWrapper()
        ds_outputs = model(x, deep_supervision=True)
        assert isinstance(ds_outputs, (list, tuple)), f"Expected list/tuple for DS, got {type(ds_outputs)}"
        print(f"  Deep Supervision Out : SUCCESS -> {len(ds_outputs)} multi-scale heads:")
        for idx, sub_out in enumerate(ds_outputs):
            print(f"     - Head {idx + 1}: Shape {list(sub_out.shape)}")

        weighted_outputs = ds_wrapper(ds_outputs)
        ds_loss = 0.0
        for sub_out, weight in weighted_outputs:
            target_resized = ds_wrapper.prepare_target(target, sub_out)
            ds_loss += weight * criterion(sub_out, target_resized)
        ds_loss.backward()
        optimizer.step()
        print(f"  Deep Supervision Back: SUCCESS -> DS Loss: {ds_loss.item():.4f}")

    print(f"  Status               : ALL TESTS PASSED [OK]")
    return {
        "name": name,
        "total_params": total_params,
        "size_mb": model_size_mb,
        "status": "PASSED"
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Model Verification Suite on Device: {device}")

    results = []
    models_to_test = []

    if args.model == "all":
        models_to_test = list(MODEL_REGISTRY.keys())
    elif args.model == "legacy":
        models_to_test = ["legacy"]
    else:
        models_to_test = [args.model]

    for model_name in models_to_test:
        if model_name == "legacy":
            model = LegacyUNet3D(in_channels=4, num_classes=4, base_channels=args.base_channels)
            res = test_single_model("legacy_unet", model, patch_size=args.patch_size, device=device, test_ds=False)
        else:
            model = get_model(model_name, in_channels=4, num_classes=4, base_channels=args.base_channels)
            res = test_single_model(model_name, model, patch_size=args.patch_size, device=device, test_ds=True)
        results.append(res)

    print(f"\n\n{'='*65}")
    print(f" MODEL VERIFICATION SUMMARY TABLE")
    print(f"{'='*65}")
    print(f"{'Model':<15} | {'Parameters':<14} | {'Size (MB)':<10} | {'Status':<10}")
    print(f"{'-'*15}-+-{'-'*14}-+-{'-'*10}-+-{'-'*10}")
    for r in results:
        print(f"{r['name']:<15} | {r['total_params']:<14,} | {r['size_mb']:<10.2f} | {r['status']:<10}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()