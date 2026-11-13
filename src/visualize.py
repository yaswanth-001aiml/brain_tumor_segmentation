import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import BraTSDataset
from src.data.preprocessing import preprocess_image, convert_mask


def find_dataset_directory(preferred_path):
    candidates = [
        preferred_path,
        os.path.join(PROJECT_ROOT, preferred_path),
        os.path.join(PROJECT_ROOT, "dataset", "BraTS2021_Training_Data", "BraTS2021_Training_Data"),
        os.path.join(PROJECT_ROOT, "dataset", "BraTS2021_Training_Data"),
        os.path.join(PROJECT_ROOT, "dataset"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d)) and "BraTS2021_" in d]
            if subdirs:
                return os.path.abspath(c)
    for root, dirs, _ in os.walk(os.path.join(PROJECT_ROOT, "dataset")):
        if any(d.startswith("BraTS2021_") for d in dirs):
            return os.path.abspath(root)
    return preferred_path


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS MRI Volume Visualizer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/BraTS2021_Training_Data/BraTS2021_Training_Data",
        help="Path to dataset directory containing patient folders"
    )
    parser.add_argument("--patient_idx", type=int, default=0, help="Index of patient to visualize")
    parser.add_argument("--output", type=str, default="outputs/mri_sample_visualization.png", help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Display interactive window")
    return parser.parse_args()


def plot_patient(raw_image, raw_mask, patient_id, save_path=None, show=False):
    """
    Plots 5 modalities/ground-truth slices plus 3-plane views (Axial, Coronal, Sagittal).
    """
    # Preprocess
    norm_image = preprocess_image(raw_image).numpy()
    mask_np = convert_mask(raw_mask).numpy()

    # Find slice with maximum tumor or central slice
    tumor_voxels = np.argwhere(mask_np > 0)
    if len(tumor_voxels) > 0:
        center_d = int(np.median(tumor_voxels[:, 0]))
        center_h = int(np.median(tumor_voxels[:, 1]))
        center_w = int(np.median(tumor_voxels[:, 2]))
    else:
        center_d = mask_np.shape[0] // 2
        center_h = mask_np.shape[1] // 2
        center_w = mask_np.shape[2] // 2

    # Custom colormap for tumor subregions:
    # 0 = Background (transparent)
    # 1 = NCR/NET (Red)
    # 2 = Edema (Green)
    # 3 = Enhancing Tumor (Yellow)
    cmap_custom = mcolors.ListedColormap(["none", "#FF3333", "#33CC33", "#FFDD00"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_cmap = mcolors.BoundaryNorm(bounds, cmap_custom.N)

    fig = plt.figure(figsize=(24, 10))

    # Row 1: 5-panel Axial Modalities
    t1 = norm_image[0, :, :, center_w]
    t1ce = norm_image[1, :, :, center_w]
    t2 = norm_image[2, :, :, center_w]
    flair = norm_image[3, :, :, center_w]
    mask_axial = mask_np[:, :, center_w]

    ax1 = plt.subplot(2, 5, 1)
    ax1.imshow(t1.T, cmap="gray", origin="lower")
    ax1.set_title("T1 (Axial)", fontsize=13, fontweight="bold")
    ax1.axis("off")

    ax2 = plt.subplot(2, 5, 2)
    ax2.imshow(t1ce.T, cmap="gray", origin="lower")
    ax2.set_title("T1ce (Contrast)", fontsize=13, fontweight="bold")
    ax2.axis("off")

    ax3 = plt.subplot(2, 5, 3)
    ax3.imshow(t2.T, cmap="gray", origin="lower")
    ax3.set_title("T2 (Axial)", fontsize=13, fontweight="bold")
    ax3.axis("off")

    ax4 = plt.subplot(2, 5, 4)
    ax4.imshow(flair.T, cmap="gray", origin="lower")
    ax4.set_title("FLAIR (Axial)", fontsize=13, fontweight="bold")
    ax4.axis("off")

    ax5 = plt.subplot(2, 5, 5)
    ax5.imshow(flair.T, cmap="gray", origin="lower")
    if np.any(mask_axial > 0):
        ax5.imshow(mask_axial.T, cmap=cmap_custom, norm=norm_cmap, alpha=0.6, origin="lower")
    ax5.set_title("FLAIR + Ground Truth", fontsize=13, fontweight="bold")
    ax5.axis("off")

    # Row 2: Multi-Planar Views (Sagittal, Coronal, Axial Overlays)
    # Sagittal view (YZ plane through center_d)
    flair_sag = norm_image[3, center_d, :, :]
    mask_sag = mask_np[center_d, :, :]

    # Coronal view (XZ plane through center_h)
    flair_cor = norm_image[3, :, center_h, :]
    mask_cor = mask_np[:, center_h, :]

    ax6 = plt.subplot(2, 3, 4)
    ax6.imshow(flair_sag.T, cmap="gray", origin="lower")
    if np.any(mask_sag > 0):
        ax6.imshow(mask_sag.T, cmap=cmap_custom, norm=norm_cmap, alpha=0.6, origin="lower")
    ax6.set_title(f"Sagittal View (Slice {center_d})", fontsize=13, fontweight="bold")
    ax6.axis("off")

    ax7 = plt.subplot(2, 3, 5)
    ax7.imshow(flair_cor.T, cmap="gray", origin="lower")
    if np.any(mask_cor > 0):
        ax7.imshow(mask_cor.T, cmap=cmap_custom, norm=norm_cmap, alpha=0.6, origin="lower")
    ax7.set_title(f"Coronal View (Slice {center_h})", fontsize=13, fontweight="bold")
    ax7.axis("off")

    ax8 = plt.subplot(2, 3, 6)
    ax8.imshow(flair.T, cmap="gray", origin="lower")
    if np.any(mask_axial > 0):
        ax8.imshow(mask_axial.T, cmap=cmap_custom, norm=norm_cmap, alpha=0.6, origin="lower")
    ax8.set_title(f"Axial View (Slice {center_w})", fontsize=13, fontweight="bold")
    ax8.axis("off")

    plt.suptitle(
        f"BraTS Multi-Modal Visualization: Patient {patient_id}\n"
        f"[Red: Necrotic Core (NCR) | Green: Edema (ED) | Yellow: Enhancing Tumor (ET)]",
        fontsize=16,
        fontweight="bold"
    )
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.95, hspace=0.3, wspace=0.2)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Visualization saved to: {os.path.abspath(save_path)}")

    if show:
        plt.show()
    plt.close()


def main():
    args = parse_args()
    data_dir = find_dataset_directory(args.data_dir)
    print(f"Searching dataset in: {data_dir}")

    patient_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and "BraTS2021_" in d
    ])

    if not patient_dirs:
        raise RuntimeError(f"No BraTS patient directories found in {data_dir}")

    if args.patient_idx >= len(patient_dirs):
        print(f"Index {args.patient_idx} exceeds patient count {len(patient_dirs)}. Using index 0.")
        args.patient_idx = 0

    patient_path = patient_dirs[args.patient_idx]
    patient_id = os.path.basename(patient_path)
    print(f"Selected Patient: {patient_id} ({args.patient_idx + 1}/{len(patient_dirs)})")

    dataset = BraTSDataset([patient_path], training=False)
    raw_image, raw_mask = dataset[0]

    plot_patient(raw_image, raw_mask, patient_id, save_path=args.output, show=args.show)


if __name__ == "__main__":
    main()