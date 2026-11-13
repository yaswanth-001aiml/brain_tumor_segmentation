import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import get_model, MODEL_REGISTRY
from src.data.preprocessing import normalize_volume
from src.metrics import extract_brats_regions


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS Patient MRI Inference & Clinical Tumor Reporting")
    parser.add_argument("--patient_dir", type=str, default=None, help="Path to patient directory containing NIfTI files")
    parser.add_argument("--t1", type=str, default=None, help="Path to T1 NIfTI (.nii / .nii.gz)")
    parser.add_argument("--t1ce", type=str, default=None, help="Path to T1ce NIfTI (.nii / .nii.gz)")
    parser.add_argument("--t2", type=str, default=None, help="Path to T2 NIfTI (.nii / .nii.gz)")
    parser.add_argument("--flair", type=str, default=None, help="Path to FLAIR NIfTI (.nii / .nii.gz)")

    parser.add_argument("--model", type=str, default="baseline", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, default="outputs/baseline_best.pth", help="Model weights path")
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=64, help="Sliding window patch size")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap fraction")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions", help="Directory to save output files")
    return parser.parse_args()


def find_modality_file(patient_dir, suffix):
    for f in os.listdir(patient_dir):
        if suffix in f and f.endswith(".nii.gz"):
            return os.path.join(patient_dir, f)
    raise FileNotFoundError(f"Missing {suffix} file in {patient_dir}")


def sliding_window_predict(model, image_tensor, patch_size=64, overlap=0.5, num_classes=4, device="cpu"):
    """
    3D sliding window inference with Gaussian / Hann weighting.
    """
    model.eval()
    _, depth, height, width = image_tensor.shape
    stride = max(1, int(patch_size * (1.0 - overlap)))

    accum_logits = torch.zeros((num_classes, depth, height, width), dtype=torch.float32)
    count_map = torch.zeros((depth, height, width), dtype=torch.float32)

    # 3D Hann window for smoother boundary blending
    hann_1d = torch.hann_window(patch_size, periodic=False)
    hann_3d = hann_1d[:, None, None] * hann_1d[None, :, None] * hann_1d[None, None, :]
    hann_3d = hann_3d.to(torch.float32) + 1e-4

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
                    patch = image_tensor[:, d:d + patch_size, h:h + patch_size, w:w + patch_size].unsqueeze(0).to(device)
                    out = model(patch)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    logits = out.squeeze(0).cpu() * hann_3d.unsqueeze(0)

                    accum_logits[:, d:d + patch_size, h:h + patch_size, w:w + patch_size] += logits
                    count_map[d:d + patch_size, h:h + patch_size, w:w + patch_size] += hann_3d

    mask = count_map > 0
    accum_logits[:, mask] /= count_map[mask]
    pred_labels = torch.argmax(accum_logits, dim=0).numpy()
    return pred_labels


def calculate_volumetrics(pred_mask, voxel_volume_mm3):
    """
    Calculates voxel counts and volumes in cm3 (mL).
    Classes: 0=BG, 1=NCR/NET, 2=ED, 3=ET
    """
    wt_mask, tc_mask, et_mask = extract_brats_regions(pred_mask)
    ed_mask = (pred_mask == 2)
    ncr_mask = (pred_mask == 1)

    voxel_to_cm3 = voxel_volume_mm3 / 1000.0  # 1 mm3 = 0.001 cm3 / mL

    metrics = {
        "whole_tumor": {
            "voxels": int(np.sum(wt_mask)),
            "volume_cm3": round(float(np.sum(wt_mask) * voxel_to_cm3), 3)
        },
        "tumor_core": {
            "voxels": int(np.sum(tc_mask)),
            "volume_cm3": round(float(np.sum(tc_mask) * voxel_to_cm3), 3)
        },
        "enhancing_tumor": {
            "voxels": int(np.sum(et_mask)),
            "volume_cm3": round(float(np.sum(et_mask) * voxel_to_cm3), 3)
        },
        "edema": {
            "voxels": int(np.sum(ed_mask)),
            "volume_cm3": round(float(np.sum(ed_mask) * voxel_to_cm3), 3)
        },
        "necrotic_core": {
            "voxels": int(np.sum(ncr_mask)),
            "volume_cm3": round(float(np.sum(ncr_mask) * voxel_to_cm3), 3)
        }
    }
    return metrics


def save_prediction_visuals(norm_image, pred_mask, patient_id, save_path, volumetrics=None):
    """
    Generates multi-panel axial modalities + tri-planar segmentation overlays.
    """
    tumor_voxels = np.argwhere(pred_mask > 0)
    if len(tumor_voxels) > 0:
        center_d = int(np.median(tumor_voxels[:, 0]))
        center_h = int(np.median(tumor_voxels[:, 1]))
        center_w = int(np.median(tumor_voxels[:, 2]))
    else:
        center_d = pred_mask.shape[0] // 2
        center_h = pred_mask.shape[1] // 2
        center_w = pred_mask.shape[2] // 2

    cmap_tumor = mcolors.ListedColormap(["none", "#FF3333", "#33CC33", "#FFDD00"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_cmap = mcolors.BoundaryNorm(bounds, cmap_tumor.N)

    fig = plt.figure(figsize=(24, 11))

    # Row 1: Axial modalities
    t1 = norm_image[0, :, :, center_w]
    t1ce = norm_image[1, :, :, center_w]
    t2 = norm_image[2, :, :, center_w]
    flair = norm_image[3, :, :, center_w]
    mask_axial = pred_mask[:, :, center_w]

    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(t1.T, cmap="gray", origin="lower")
    ax1.set_title("T1 (Axial)", fontsize=13, fontweight="bold")
    ax1.axis("off")

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(t1ce.T, cmap="gray", origin="lower")
    ax2.set_title("T1ce (Contrast)", fontsize=13, fontweight="bold")
    ax2.axis("off")

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(t2.T, cmap="gray", origin="lower")
    ax3.set_title("T2 (Axial)", fontsize=13, fontweight="bold")
    ax3.axis("off")

    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(flair.T, cmap="gray", origin="lower")
    ax4.set_title("FLAIR (Axial)", fontsize=13, fontweight="bold")
    ax4.axis("off")

    # Row 2: Tri-planar Segmentations + Volumetric Info
    # Axial Overlay
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(flair.T, cmap="gray", origin="lower")
    if np.any(mask_axial > 0):
        ax5.imshow(mask_axial.T, cmap=cmap_tumor, norm=norm_cmap, alpha=0.6, origin="lower")
    ax5.set_title(f"Axial Segmentation (Slice {center_w})", fontsize=13, fontweight="bold")
    ax5.axis("off")

    # Coronal Overlay
    flair_cor = norm_image[3, :, center_h, :]
    mask_cor = pred_mask[:, center_h, :]
    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(flair_cor.T, cmap="gray", origin="lower")
    if np.any(mask_cor > 0):
        ax6.imshow(mask_cor.T, cmap=cmap_tumor, norm=norm_cmap, alpha=0.6, origin="lower")
    ax6.set_title(f"Coronal Segmentation (Slice {center_h})", fontsize=13, fontweight="bold")
    ax6.axis("off")

    # Sagittal Overlay
    flair_sag = norm_image[3, center_d, :, :]
    mask_sag = pred_mask[center_d, :, :]
    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(flair_sag.T, cmap="gray", origin="lower")
    if np.any(mask_sag > 0):
        ax7.imshow(mask_sag.T, cmap=cmap_tumor, norm=norm_cmap, alpha=0.6, origin="lower")
    ax7.set_title(f"Sagittal Segmentation (Slice {center_d})", fontsize=13, fontweight="bold")
    ax7.axis("off")

    # Panel 8: Clinical Volumetrics Summary Card
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis("off")
    summary_text = f"PATIENT REPORT: {patient_id}\n"
    summary_text += "=" * 32 + "\n"
    if volumetrics:
        wt = volumetrics["whole_tumor"]
        tc = volumetrics["tumor_core"]
        et = volumetrics["enhancing_tumor"]
        ed = volumetrics["edema"]
        ncr = volumetrics["necrotic_core"]
        summary_text += f"- Whole Tumor (WT) : {wt['volume_cm3']:.2f} cm³ ({wt['voxels']:,} vx)\n"
        summary_text += f"- Tumor Core (TC)  : {tc['volume_cm3']:.2f} cm³ ({tc['voxels']:,} vx)\n"
        summary_text += f"- Enhancing (ET)   : {et['volume_cm3']:.2f} cm³ ({et['voxels']:,} vx)\n"
        summary_text += f"- Edema (ED)       : {ed['volume_cm3']:.2f} cm³ ({ed['voxels']:,} vx)\n"
        summary_text += f"- Necrotic (NCR)   : {ncr['volume_cm3']:.2f} cm³ ({ncr['voxels']:,} vx)\n\n"
    summary_text += "Subregion Color Legend:\n"
    summary_text += "  [■ Red]    Necrotic / Non-enhancing\n"
    summary_text += "  [■ Green]  Peritumoral Edema\n"
    summary_text += "  [■ Yellow] Active Enhancing Tumor\n"

    ax8.text(0.05, 0.5, summary_text, fontsize=12, family="monospace", verticalalignment="center",
             bbox=dict(boxstyle="round,pad=1.0", facecolor="#F4F6F9", edgecolor="#A0AEC0"))

    plt.suptitle(
        f"Automated 3D Brain Tumor Segmentation: {patient_id}",
        fontsize=16,
        fontweight="bold"
    )
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.04, right=0.96, hspace=0.25, wspace=0.15)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f" BraTS Automated MRI Tumor Segmentation & Reporting")
    print(f"{'='*65}")
    print(f"Model Architecture : {args.model.upper()}")
    print(f"Weights Path       : {args.checkpoint}")
    print(f"Inference Device   : {device}")

    # Determine modality paths
    if args.patient_dir:
        patient_id = os.path.basename(os.path.abspath(args.patient_dir))
        t1_path = find_modality_file(args.patient_dir, "_t1")
        t1ce_path = find_modality_file(args.patient_dir, "_t1ce")
        t2_path = find_modality_file(args.patient_dir, "_t2")
        flair_path = find_modality_file(args.patient_dir, "_flair")
    elif all([args.t1, args.t1ce, args.t2, args.flair]):
        patient_id = "custom_patient"
        t1_path, t1ce_path, t2_path, flair_path = args.t1, args.t1ce, args.t2, args.flair
    else:
        # Default fallback to first patient in dataset
        default_dir = os.path.join(PROJECT_ROOT, "dataset", "BraTS2021_Training_Data", "BraTS2021_Training_Data")
        if os.path.isdir(default_dir):
            patients = sorted([os.path.join(default_dir, p) for p in os.listdir(default_dir) if "BraTS2021_" in p])
            if patients:
                args.patient_dir = patients[0]
                patient_id = os.path.basename(args.patient_dir)
                t1_path = find_modality_file(args.patient_dir, "_t1")
                t1ce_path = find_modality_file(args.patient_dir, "_t1ce")
                t2_path = find_modality_file(args.patient_dir, "_t2")
                flair_path = find_modality_file(args.patient_dir, "_flair")
            else:
                raise ValueError("No patient directories found in default path.")
        else:
            raise ValueError("Must specify either --patient_dir or all four of --t1, --t1ce, --t2, --flair.")

    print(f"Target Patient     : {patient_id}")
    print(f"FLAIR Image Path   : {flair_path}")

    # Load NIfTI volumes
    nii_ref = nib.load(flair_path)
    ref_affine = nii_ref.affine
    ref_header = nii_ref.header
    zooms = ref_header.get_zooms()[:3]
    voxel_volume_mm3 = float(zooms[0] * zooms[1] * zooms[2])

    t1_np = nib.load(t1_path).get_fdata().astype(np.float32)
    t1ce_np = nib.load(t1ce_path).get_fdata().astype(np.float32)
    t2_np = nib.load(t2_path).get_fdata().astype(np.float32)
    flair_np = nii_ref.get_fdata().astype(np.float32)

    # Normalize
    t1_norm = normalize_volume(t1_np)
    t1ce_norm = normalize_volume(t1ce_np)
    t2_norm = normalize_volume(t2_np)
    flair_norm = normalize_volume(flair_np)

    raw_stack = np.stack([t1_norm, t1ce_norm, t2_norm, flair_norm], axis=0)
    tensor_input = torch.tensor(raw_stack, dtype=torch.float32)

    # Load Model Checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Auto-detect base_channels if possible
    detected_base_channels = args.base_channels
    for k in ["encoder1.block.0.weight", "encoder1.conv1.weight", "enc1.block.0.weight"]:
        if k in state_dict:
            detected_base_channels = state_dict[k].shape[0]
            break

    if detected_base_channels != args.base_channels:
        print(f"Auto-detected base_channels from checkpoint: {detected_base_channels}")

    model = get_model(args.model, in_channels=4, num_classes=4, base_channels=detected_base_channels).to(device)
    model.load_state_dict(state_dict)
    print("Model checkpoint loaded successfully.")

    # Predict
    print("\nRunning sliding-window 3D inference...")
    pred_mask = sliding_window_predict(
        model=model,
        image_tensor=tensor_input,
        patch_size=args.patch_size,
        overlap=args.overlap,
        device=device
    )

    # Calculate metrics
    volumetrics = calculate_volumetrics(pred_mask, voxel_volume_mm3)

    print(f"\n{'='*65}")
    print(" Tumoral Volumetric Findings")
    print(f"{'='*65}")
    print(f"Whole Tumor (WT)  : {volumetrics['whole_tumor']['volume_cm3']} cm³ ({volumetrics['whole_tumor']['voxels']:,} voxels)")
    print(f"Tumor Core (TC)   : {volumetrics['tumor_core']['volume_cm3']} cm³ ({volumetrics['tumor_core']['voxels']:,} voxels)")
    print(f"Enhancing (ET)    : {volumetrics['enhancing_tumor']['volume_cm3']} cm³ ({volumetrics['enhancing_tumor']['voxels']:,} voxels)")
    print(f"Edema (ED)        : {volumetrics['edema']['volume_cm3']} cm³ ({volumetrics['edema']['voxels']:,} voxels)")
    print(f"Necrotic (NCR)    : {volumetrics['necrotic_core']['volume_cm3']} cm³ ({volumetrics['necrotic_core']['voxels']:,} voxels)")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # Save NIfTI mask
    nii_pred = np.zeros_like(pred_mask, dtype=np.uint8)
    nii_pred[pred_mask == 1] = 1
    nii_pred[pred_mask == 2] = 2
    nii_pred[pred_mask == 3] = 4  # Official BraTS enhancing label
    out_nii_path = os.path.join(args.output_dir, f"{patient_id}_segmentation.nii.gz")
    nib.save(nib.Nifti1Image(nii_pred, affine=ref_affine, header=ref_header), out_nii_path)
    print(f"\nSegmentation mask saved to: {out_nii_path}")

    # Save Visuals
    vis_path = os.path.join(args.output_dir, f"{patient_id}_report.png")
    save_prediction_visuals(raw_stack, pred_mask, patient_id, vis_path, volumetrics=volumetrics)
    print(f"Visual report saved to     : {vis_path}")

    # Save JSON Report
    report_dict = {
        "patient_id": patient_id,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "voxel_spacing_mm": [float(z) for z in zooms],
        "volumetrics": volumetrics
    }
    json_path = os.path.join(args.output_dir, f"{patient_id}_report.json")
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Structured report saved to : {json_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
