import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import BraTSDataset
from src.data.split import split_dataset, get_patient_directories
from src.patch_dataset import BraTSPatchDataset
from src.models import get_model, MODEL_REGISTRY
from src.losses import get_loss
from src.train import find_dataset_directory


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS Quick Smoke Test")
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Model architecture: {list(MODEL_REGISTRY.keys())}"
    )
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size for rapid smoke test")
    parser.add_argument("--base_channels", type=int, default=4, help="Minimal channels for speed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    print(f"\n{'='*60}")
    print(f" BraTS Rapid Smoke Test ({args.model.upper()})")
    print(f"{'='*60}")
    print(f"Device        : {device}")
    print(f"Patch Size    : {args.patch_size}x{args.patch_size}x{args.patch_size}")
    print(f"Base Channels : {args.base_channels}")

    # Discover dataset
    dataset_dir = find_dataset_directory("dataset/BraTS2021_Training_Data/BraTS2021_Training_Data")
    patient_dirs = get_patient_directories(dataset_dir)
    print(f"Found Patients: {len(patient_dirs)} (subsampling 2 patients for fast smoke test)")

    # Rapid subsample of 2 patients
    sample_patients = patient_dirs[:2]
    base_dataset = BraTSDataset(sample_patients, training=True)

    patch_dataset = BraTSPatchDataset(
        base_dataset,
        patch_size=args.patch_size,
        patches_per_patient=2,
        cache_data=True
    )
    print(f"Training Patches : {len(patch_dataset)}")

    loader = DataLoader(
        patch_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    model = get_model(
        args.model,
        in_channels=4,
        num_classes=4,
        base_channels=args.base_channels
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters : {total_params:,}")

    criterion = get_loss("combined")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 1 mini-epoch
    model.train()
    print("\nRunning test forward & backward passes...")
    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"  Batch {batch_idx + 1}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Average Smoke Test Loss: {avg_loss:.4f}")

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{args.model}_smoke_test.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Checkpoint saved to: {out_path}")
    print(f"\n[OK] Smoke test completed successfully!\n")


if __name__ == "__main__":
    main()