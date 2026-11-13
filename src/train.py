import gc
import os
import sys
import json
import csv
import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import BraTSDataset
from src.data.split import split_dataset, get_patient_directories
from src.data.augmentation import RandomFlip3D, RandomIntensityShift, Compose3D
from src.patch_dataset import BraTSPatchDataset
from src.losses import get_loss, LOSS_REGISTRY
from src.metrics import BraTSMetricAggregator, compute_brats_dice
from src.models import get_model, MODEL_REGISTRY
from src.training.deep_supervision import DeepSupervisionLossWrapper


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_dataset_directory(preferred_path):
    """
    Locates the actual directory containing patient folders (BraTS2021_*).
    """
    candidates = [
        preferred_path,
        os.path.join(PROJECT_ROOT, preferred_path),
        os.path.join(PROJECT_ROOT, "dataset", "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
        os.path.join(PROJECT_ROOT, "dataset", "BraTS2021_Training_Data", "BraTS2021_Training_Data"),
        os.path.join(PROJECT_ROOT, "dataset", "BraTS2021_Training_Data"),
        os.path.join(PROJECT_ROOT, "dataset"),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            subdirs = [
                d for d in os.listdir(candidate)
                if os.path.isdir(os.path.join(candidate, d)) and ("BraTS2021_" in d or "BraTS-GLI-" in d)
            ]
            if len(subdirs) > 0:
                return os.path.abspath(candidate)

    # Fallback to search recursively for any directory containing patient folders
    for root, dirs, _ in os.walk(os.path.join(PROJECT_ROOT, "dataset")):
        patient_dirs = [d for d in dirs if (d.startswith("BraTS2021_") or d.startswith("BraTS-GLI-"))]
        if len(patient_dirs) > 0:
            return os.path.abspath(root)

    return preferred_path


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS 3D Brain Tumor Segmentation Training Pipeline")

    # Architecture
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Model architecture: {list(MODEL_REGISTRY.keys())}"
    )
    parser.add_argument("--base_channels", type=int, default=16, help="Initial conv channels (e.g. 16, 32)")
    parser.add_argument("--deep_supervision", action="store_true", help="Enable multi-scale deep supervision")

    # Loss function
    parser.add_argument(
        "--loss",
        type=str,
        default="combined",
        choices=list(LOSS_REGISTRY.keys()),
        help=f"Loss function: {list(LOSS_REGISTRY.keys())}"
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/BraTS2021_Training_Data/BraTS2021_Training_Data",
        help="Path to patient folders"
    )
    parser.add_argument("--patch_size", type=int, default=64, help="Cubic patch size for 3D training (e.g. 32, 64, 96)")
    parser.add_argument("--patches_per_patient", type=int, default=2, help="Number of random/tumor patches per patient")
    parser.add_argument("--no_cache", action="store_true", help="Disable in-memory volume caching (saves RAM for large datasets)")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--max_patients", type=int, default=None, help="Limit total patients for rapid debug/smoke tests")

    # Optimization
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP) on CUDA")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (use 0 on Windows)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Checkpoint and Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save checkpoints and logs")
    parser.add_argument("--save_prefix", type=str, default=None, help="Custom prefix for saved model weights")

    return parser.parse_args()


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    deep_supervision=False,
    ds_wrapper=None,
    scaler=None,
    grad_accum_steps=1
):
    model.train()
    running_loss = 0.0
    use_amp = (scaler is not None) and (device.type == "cuda")

    optimizer.zero_grad()

    t_start = time.time()
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                if deep_supervision:
                    outputs = model(images, deep_supervision=True)
                    weighted_outputs = ds_wrapper(outputs)
                    loss = 0.0
                    for out, weight in weighted_outputs:
                        target_resized = ds_wrapper.prepare_target(masks, out)
                        loss += weight * criterion(out, target_resized)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                loss_to_backward = loss / grad_accum_steps

            scaler.scale(loss_to_backward).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            if deep_supervision:
                outputs = model(images, deep_supervision=True)
                weighted_outputs = ds_wrapper(outputs)
                loss = 0.0
                for out, weight in weighted_outputs:
                    target_resized = ds_wrapper.prepare_target(masks, out)
                    loss += weight * criterion(out, target_resized)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)

            loss_to_backward = loss / grad_accum_steps
            loss_to_backward.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            elapsed = time.time() - t_start
            avg_b = elapsed / (batch_idx + 1)
            eta = avg_b * (len(loader) - (batch_idx + 1))
            cur_loss = running_loss / (batch_idx + 1)
            print(f'  [Train] Batch {batch_idx + 1}/{len(loader)} | Loss: {cur_loss:.4f} | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m', flush=True)
            gc.collect()

    avg_loss = running_loss / max(len(loader), 1)
    return avg_loss


def validate(
    model,
    loader,
    criterion,
    device
):
    model.eval()
    running_loss = 0.0
    aggregator = BraTSMetricAggregator()

    if hasattr(loader.dataset, '_lru_cache'):
        loader.dataset._lru_cache.clear()
    gc.collect()

    t_val_start = time.time()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            pred_labels = torch.argmax(outputs, dim=1)
            for b in range(pred_labels.shape[0]):
                aggregator.update(pred_labels[b], masks[b])

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                elapsed = time.time() - t_val_start
                print(f'  [Val] Batch {batch_idx + 1}/{len(loader)} | Elapsed: {elapsed/60:.1f}m', flush=True)
                gc.collect()

    if hasattr(loader.dataset, '_lru_cache'):
        loader.dataset._lru_cache.clear()
    gc.collect()

    avg_loss = running_loss / max(len(loader), 1)
    metrics_summary = aggregator.summary()
    return avg_loss, metrics_summary


def save_history(history, output_dir, prefix):
    json_path = os.path.join(output_dir, f"{prefix}_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    csv_path = os.path.join(output_dir, f"{prefix}_history.csv")
    if history:
        keys = list(history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" BraTS Brain Tumor Segmentation Training Pipeline")
    print(f"{'='*60}")
    print(f"Model Architecture : {args.model.upper()}")
    print(f"Device             : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Loss Function      : {args.loss.upper()}")
    print(f"Base Channels      : {args.base_channels}")
    print(f"Patch Size         : {args.patch_size}x{args.patch_size}x{args.patch_size}")
    print(f"Deep Supervision   : {args.deep_supervision}")
    print(f"Mixed Precision AMP: {args.amp and device.type == 'cuda'}")
    print(f"Grad Accum Steps   : {args.grad_accum_steps}")
    print(f"Epochs             : {args.epochs}")
    print(f"Batch Size         : {args.batch_size}")
    print(f"Learning Rate      : {args.lr}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.save_prefix or f"{args.model}_ds" if args.deep_supervision else args.save_prefix or args.model

    # Dataset discovery and split
    dataset_dir = find_dataset_directory(args.data_dir)
    print(f"Dataset Directory  : {dataset_dir}")

    train_pts, val_pts, test_pts = split_dataset(
        dataset_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    if args.max_patients is not None:
        train_pts = train_pts[:max(1, int(args.max_patients * 0.7))]
        val_pts = val_pts[:max(1, int(args.max_patients * 0.3))]
        print(f"Debug Subsampled   : Train={len(train_pts)}, Val={len(val_pts)}")
    else:
        print(f"Data Splits        : Train={len(train_pts)}, Val={len(val_pts)}, Test={len(test_pts)}")

    # Data Augmentations
    train_transforms = Compose3D([
        RandomFlip3D(probability=0.5),
        RandomIntensityShift(probability=0.3, max_shift=0.1)
    ])

    train_base = BraTSDataset(train_pts, training=True)
    val_base = BraTSDataset(val_pts, training=False)

    cache_data = not args.no_cache
    print(f"Memory Caching     : {'ENABLED' if cache_data else 'DISABLED'}")

    train_patch_ds = BraTSPatchDataset(
        train_base,
        patch_size=args.patch_size,
        patches_per_patient=args.patches_per_patient,
        transform=train_transforms,
        cache_data=cache_data
    )
    val_patch_ds = BraTSPatchDataset(
        val_base,
        patch_size=args.patch_size,
        patches_per_patient=args.patches_per_patient,
        transform=None,
        cache_data=cache_data
    )

    train_loader = DataLoader(
        train_patch_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_patch_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Model instantiation
    model = get_model(
        args.model,
        in_channels=4,
        num_classes=4,
        base_channels=args.base_channels
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters   : {total_params:,}")

    # Loss, Optimizer & Scheduler
    criterion = get_loss(args.loss)
    ds_wrapper = DeepSupervisionLossWrapper() if args.deep_supervision else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP Scaler
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    # Resume if requested
    start_epoch = 1
    best_val_dice = -1.0
    history = []

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict)
            if "optimizer" in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
            if "best_val_dice" in checkpoint and checkpoint["best_val_dice"] > 0:
                best_val_dice = checkpoint["best_val_dice"]
            else:
                best_path = os.path.join(args.output_dir, f"{prefix}_best.pth")
                if os.path.isfile(best_path):
                    try:
                        b_ckpt = torch.load(best_path, map_location=device)
                        if "best_val_dice" in b_ckpt and b_ckpt["best_val_dice"] > 0:
                            best_val_dice = b_ckpt["best_val_dice"]
                    except Exception:
                        pass

            hist_path = os.path.join(args.output_dir, f"{prefix}_history.json")
            if os.path.isfile(hist_path):
                try:
                    with open(hist_path, "r") as f:
                        history = json.load(f)
                    history = [h for h in history if h.get("epoch", 0) < start_epoch]
                except Exception:
                    pass

            print(f"Resumed successfully at Epoch {start_epoch} (Best Val Dice: {best_val_dice:.4f})")
        else:
            print(f"Warning: Checkpoint not found at '{args.resume}'. Starting fresh.")

    # Early stopping tracking
    epochs_no_improve = 0
    start_time = time.time()

    print(f"\nStarting training loop (Epochs {start_epoch} to {args.epochs})...")
    print(f"{'-'*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            deep_supervision=args.deep_supervision,
            ds_wrapper=ds_wrapper,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps
        )

        val_loss, val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        wt_dice = val_metrics.get("wt_dice_mean", 0.0)
        tc_dice = val_metrics.get("tc_dice_mean", 0.0)
        et_dice = val_metrics.get("et_dice_mean", 0.0)
        mean_dice = val_metrics.get("mean_dice_mean", 0.0)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_wt_dice": round(wt_dice, 4),
            "val_tc_dice": round(tc_dice, 4),
            "val_et_dice": round(et_dice, 4),
            "val_mean_dice": round(mean_dice, 4),
            "lr": current_lr,
            "time_sec": round(epoch_time, 2)
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"WT: {wt_dice:.4f} | TC: {tc_dice:.4f} | ET: {et_dice:.4f} | Mean: {mean_dice:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Checkpoint payload
        ckpt_payload = {
            "epoch": epoch,
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
            "val_metrics": val_metrics,
            "args": vars(args)
        }

        # Save best checkpoint first if improved
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(args.output_dir, f"{prefix}_best.pth")
            ckpt_payload["best_val_dice"] = best_val_dice
            torch.save(ckpt_payload, best_ckpt_path)
            print(f"  --> Saved new best model (Mean Dice: {mean_dice:.4f}) to {best_ckpt_path}")
        else:
            epochs_no_improve += 1

        ckpt_payload["best_val_dice"] = best_val_dice
        last_ckpt_path = os.path.join(args.output_dir, f"{prefix}_last.pth")
        torch.save(ckpt_payload, last_ckpt_path)
        if args.patience is not None and epochs_no_improve >= args.patience:
            print(f"\n[Early Stopping] No improvement in validation Dice for {args.patience} epochs.")
            break

        save_history(history, args.output_dir, prefix)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete! Total Time: {total_time // 60:.0f}m {total_time % 60:.1f}s")
    print(f"Best Val Mean Dice: {best_val_dice:.4f}")
    print(f"Checkpoints & logs saved in: {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
