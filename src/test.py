import os
import sys
import json
import csv
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import BraTSDataset
from src.data.split import split_dataset
from src.data.preprocessing import preprocess_image, convert_mask
from src.metrics import compute_brats_dice, compute_sensitivity_specificity, extract_brats_regions, BraTSMetricAggregator
from src.models import get_model, MODEL_REGISTRY


def find_dataset_directory(preferred_path):
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
    for root, dirs, _ in os.walk(os.path.join(PROJECT_ROOT, "dataset")):
        patient_dirs = [d for d in dirs if (d.startswith("BraTS2021_") or d.startswith("BraTS-GLI-"))]
        if len(patient_dirs) > 0:
            return os.path.abspath(root)
    return preferred_path


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS Evaluation and Inference Pipeline")
    parser.add_argument("--model", type=str, default="baseline", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, default="outputs/baseline_best.pth", help="Model weights path")
    parser.add_argument("--data_dir", type=str, default="dataset/BraTS2021_Training_Data/BraTS2021_Training_Data")
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size for sliding window inference")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap fraction")
    parser.add_argument("--output_dir", type=str, default="outputs/test_results")
    parser.add_argument("--max_patients", type=int, default=None, help="Limit number of test patients")
    parser.add_argument("--save_nifti", action="store_true", help="Save predictions as .nii.gz")
    parser.add_argument("--save_plots", action="store_true", default=True, help="Save PNG comparison figures")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sliding_window_inference(model, image, patch_size=64, overlap=0.5, num_classes=4, device="cpu"):
    """
    Performs 3D sliding window inference with linear overlap blending.
    image: [4, D, H, W] tensor
    Returns:
        [D, H, W] predicted label map
    """
    model.eval()
    _, depth, height, width = image.shape
    stride = max(1, int(patch_size * (1.0 - overlap)))

    accum_logits = torch.zeros((num_classes, depth, height, width), dtype=torch.float32)
    count_map = torch.zeros((depth, height, width), dtype=torch.float32)

    d_starts = list(range(0, depth - patch_size + 1, stride))
    if (depth - patch_size) not in d_starts:
        d_starts.append(depth - patch_size)

    h_starts = list(range(0, height - patch_size + 1, stride))
    if (height - patch_size) not in h_starts:
        h_starts.append(height - patch_size)

    w_starts = list(range(0, width - patch_size + 1, stride))
    if (width - patch_size) not in w_starts:
        w_starts.append(width - patch_size)

    with torch.no_grad():
        for d in d_starts:
            for h in h_starts:
                for w in w_starts:
                    patch = image[:, d:d + patch_size, h:h + patch_size, w:w + patch_size].unsqueeze(0).to(device)
                    output = model(patch)
                    if isinstance(output, (list, tuple)):
                        output = output[0]  # Take primary output if deep supervision
                    accum_logits[:, d:d + patch_size, h:h + patch_size, w:w + patch_size] += output.squeeze(0).cpu()
                    count_map[d:d + patch_size, h:h + patch_size, w:w + patch_size] += 1.0

    mask = count_map > 0
    accum_logits[:, mask] /= count_map[mask]
    pred_labels = torch.argmax(accum_logits, dim=0).numpy()
    return pred_labels


def save_visualization(image_np, gt_mask_np, pred_mask_np, patient_id, save_path):
    """
    Generates side-by-side visualization of modalities and segmentation comparison.
    """
    # Select axial slice with maximum tumor in GT or center slice
    tumor_slices = np.argwhere(gt_mask_np > 0)
    if len(tumor_slices) > 0:
        slice_idx = int(np.median(tumor_slices[:, 2]))
    else:
        slice_idx = gt_mask_np.shape[2] // 2

    t1 = image_np[0, :, :, slice_idx]
    t1ce = image_np[1, :, :, slice_idx]
    t2 = image_np[2, :, :, slice_idx]
    flair = image_np[3, :, :, slice_idx]
    gt_slice = gt_mask_np[:, :, slice_idx]
    pred_slice = pred_mask_np[:, :, slice_idx]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(t1.T, cmap="gray", origin="lower")
    axes[0].set_title("T1", fontsize=14)

    axes[1].imshow(t1ce.T, cmap="gray", origin="lower")
    axes[1].set_title("T1ce", fontsize=14)

    axes[2].imshow(t2.T, cmap="gray", origin="lower")
    axes[2].set_title("T2", fontsize=14)

    # FLAIR + GT
    axes[3].imshow(flair.T, cmap="gray", origin="lower")
    if np.any(gt_slice > 0):
        axes[3].imshow(np.ma.masked_where(gt_slice.T == 0, gt_slice.T), cmap="jet", alpha=0.5, origin="lower")
    axes[3].set_title("FLAIR + Ground Truth", fontsize=14)

    # FLAIR + Prediction
    axes[4].imshow(flair.T, cmap="gray", origin="lower")
    if np.any(pred_slice > 0):
        axes[4].imshow(np.ma.masked_where(pred_slice.T == 0, pred_slice.T), cmap="jet", alpha=0.5, origin="lower")
    axes[4].set_title("FLAIR + Prediction", fontsize=14)

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"Patient {patient_id} (Axial Slice {slice_idx})", fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f" BraTS Model Evaluation & Testing")
    print(f"{'='*65}")
    print(f"Model Architecture : {args.model.upper()}")
    print(f"Checkpoint Path    : {args.checkpoint}")
    print(f"Device             : {device}")
    print(f"Patch Size         : {args.patch_size}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    nii_dir = os.path.join(args.output_dir, "nifti")

    dataset_dir = find_dataset_directory(args.data_dir)
    _, _, test_patients = split_dataset(dataset_dir, seed=args.seed)

    if args.max_patients is not None:
        test_patients = test_patients[:args.max_patients]

    print(f"Test Patients      : {len(test_patients)}")

    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)

    # Auto-detect base_channels from weights if possible
    detected_base_channels = args.base_channels
    for k in ["encoder1.block.0.weight", "encoder1.conv1.weight", "enc1.block.0.weight"]:
        if k in state_dict:
            detected_base_channels = state_dict[k].shape[0]
            break

    if detected_base_channels != args.base_channels:
        print(f"Auto-detected base_channels from checkpoint: {detected_base_channels}")

    # Instantiate model and load checkpoint
    model = get_model(
        args.model,
        in_channels=4,
        num_classes=4,
        base_channels=detected_base_channels
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    print("Checkpoint loaded successfully.")

    dataset = BraTSDataset(test_patients, training=False)
    aggregator = BraTSMetricAggregator()
    patient_records = []

    print(f"\nRunning inference on test patients...")
    print(f"{'-'*65}")

    for idx in range(len(dataset)):
        raw_image, raw_mask = dataset[idx]
        patient_path = test_patients[idx]
        patient_id = os.path.basename(patient_path)

        norm_image = preprocess_image(raw_image)
        gt_mask = convert_mask(raw_mask)

        # Sliding window inference
        pred_labels = sliding_window_inference(
            model=model,
            image=norm_image,
            patch_size=args.patch_size,
            overlap=args.overlap,
            device=device
        )

        # Compute metrics
        dice_dict = compute_brats_dice(pred_labels, gt_mask)
        aggregator.update(pred_labels, gt_mask, compute_sens_spec=True)

        pred_wt, pred_tc, pred_et = extract_brats_regions(pred_labels)
        gt_wt, gt_tc, gt_et = extract_brats_regions(gt_mask.numpy())

        wt_sens, wt_spec = compute_sensitivity_specificity(pred_wt, gt_wt)
        tc_sens, tc_spec = compute_sensitivity_specificity(pred_tc, gt_tc)
        et_sens, et_spec = compute_sensitivity_specificity(pred_et, gt_et)

        record = {
            "patient_id": patient_id,
            "wt_dice": round(dice_dict["wt_dice"], 4),
            "tc_dice": round(dice_dict["tc_dice"], 4),
            "et_dice": round(dice_dict["et_dice"], 4),
            "mean_dice": round(dice_dict["mean_dice"], 4),
            "wt_sens": round(wt_sens, 4),
            "wt_spec": round(wt_spec, 4),
            "tc_sens": round(tc_sens, 4),
            "tc_spec": round(tc_spec, 4),
            "et_sens": round(et_sens, 4),
            "et_spec": round(et_spec, 4),
        }
        patient_records.append(record)

        print(
            f"[{idx + 1:02d}/{len(dataset):02d}] {patient_id} | "
            f"WT: {record['wt_dice']:.4f} | TC: {record['tc_dice']:.4f} | "
            f"ET: {record['et_dice']:.4f} | Mean: {record['mean_dice']:.4f}"
        )

        # Save Visualizations
        if args.save_plots:
            vis_path = os.path.join(vis_dir, f"{patient_id}_comparison.png")
            save_visualization(norm_image.numpy(), gt_mask.numpy(), pred_labels, patient_id, vis_path)

        # Save NIfTI
        if args.save_nifti:
            os.makedirs(nii_dir, exist_ok=True)
            # Map back: 3 -> 4 for enhancing tumor in official BraTS notation
            nii_pred = np.zeros_like(pred_labels, dtype=np.uint8)
            nii_pred[pred_labels == 1] = 1
            nii_pred[pred_labels == 2] = 2
            nii_pred[pred_labels == 3] = 4

            # Retrieve affine and header from original patient file
            ref_affine = np.eye(4)
            ref_header = None
            for fname in os.listdir(patient_path):
                if fname.endswith(".nii.gz"):
                    try:
                        ref_nii = nib.load(os.path.join(patient_path, fname))
                        ref_affine = ref_nii.affine
                        ref_header = ref_nii.header
                        break
                    except Exception:
                        pass

            nii_img = nib.Nifti1Image(nii_pred, affine=ref_affine, header=ref_header)
            nib.save(nii_img, os.path.join(nii_dir, f"{patient_id}_pred.nii.gz"))

    # Summary and reporting
    summary = aggregator.summary()
    print(f"\n{'='*65}")
    print("Test Evaluation Summary")
    print(f"{'='*65}")
    print(aggregator.summary_table())

    csv_path = os.path.join(args.output_dir, "test_patient_results.csv")
    if patient_records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(patient_records[0].keys()))
            writer.writeheader()
            writer.writerows(patient_records)

    json_path = os.path.join(args.output_dir, "test_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetailed CSV results saved to: {csv_path}")
    print(f"Summary JSON saved to: {json_path}")
    if args.save_plots:
        print(f"Visualizations saved to: {vis_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
