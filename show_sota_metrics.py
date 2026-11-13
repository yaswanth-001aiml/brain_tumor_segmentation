import torch

checkpoint_path = "outputs/combined_sota_best.pth"
print(f"Loading checkpoint: {checkpoint_path} ...\n")

ckpt = torch.load(checkpoint_path, map_location="cpu")
epoch = ckpt.get("epoch", "N/A")
m = ckpt.get("val_metrics", {})

print(f"{'='*45}")
print(f" COMBINED SOTA MODEL METRICS (Best Epoch: {epoch})")
print(f"{'='*45}")
print(f" Mean Dice         : {m.get('mean_dice_mean', 0):.4f} (Median: {m.get('mean_dice_median', 0):.4f})")
print(f" Whole Tumor (WT)  : {m.get('wt_dice_mean', 0):.4f} (Median: {m.get('wt_dice_median', 0):.4f})")
print(f" Tumor Core (TC)   : {m.get('tc_dice_mean', 0):.4f} (Median: {m.get('tc_dice_median', 0):.4f})")
print(f" Enhancing T. (ET) : {m.get('et_dice_mean', 0):.4f} (Median: {m.get('et_dice_median', 0):.4f})")
if 'val_loss' in m:
    print(f" Validation Loss   : {m['val_loss']:.4f}")
print(f"{'='*45}")
