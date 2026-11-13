import os
import sys
import json
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import get_model, MODEL_REGISTRY
from src.data.dataset import BraTSDataset
from src.data.split import split_dataset
from src.data.preprocessing import preprocess_image, convert_mask
from src.metrics import compute_brats_dice, compute_sensitivity_specificity, extract_brats_regions, BraTSMetricAggregator
from src.test import sliding_window_inference, find_dataset_directory


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and Benchmark All Trained BraTS Models")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory containing model checkpoints")
    parser.add_argument("--data_dir", type=str, default="dataset/BraTS2021_Training_Data/BraTS2021_Training_Data")
    parser.add_argument("--patch_size", type=int, default=64, help="Sliding window patch size")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap")
    parser.add_argument("--max_test_patients", type=int, default=5, help="Limit number of test patients for fast benchmarking")
    parser.add_argument("--output_file", type=str, default="outputs/model_benchmark_comparison.csv")
    parser.add_argument("--plot_file", type=str, default="outputs/model_benchmark_comparison.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", "--summary_only", action="store_true", help="Print existing benchmark leaderboard without re-running inference")
    return parser.parse_args()


def discover_checkpoints(outputs_dir):
    checkpoints = []
    if not os.path.isdir(outputs_dir):
        return checkpoints

    for f in os.listdir(outputs_dir):
        if f.endswith("_best.pth"):
            full_path = os.path.join(outputs_dir, f)
            # Infer model name
            name_part = f.replace("_best.pth", "")
            base_model = "baseline"
            for k in MODEL_REGISTRY.keys():
                if k in name_part:
                    base_model = k
                    break
            checkpoints.append({
                "filename": f,
                "path": full_path,
                "model_name": base_model,
                "display_name": name_part
            })
    return checkpoints


def evaluate_checkpoint(ckpt_info, test_dataset, device, patch_size=64, overlap=0.5):
    path = ckpt_info["path"]
    model_name = ckpt_info["model_name"]
    ckpt_data = torch.load(path, map_location=device)
    state_dict = ckpt_data.get("state_dict", ckpt_data)

    # Detect base_channels
    detected_base_channels = 16
    if isinstance(ckpt_data, dict) and "args" in ckpt_data and isinstance(ckpt_data["args"], dict) and "base_channels" in ckpt_data["args"]:
        detected_base_channels = ckpt_data["args"]["base_channels"]
    else:
        for k in ["encoder1.block.0.weight", "encoder1.conv1.weight", "enc1.block.0.weight", "encoder1.fusion.0.weight"]:
            if k in state_dict:
                detected_base_channels = state_dict[k].shape[0]
                break

    model = get_model(model_name, in_channels=4, num_classes=4, base_channels=detected_base_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    aggregator = BraTSMetricAggregator()
    patient_times = []

    for i in range(len(test_dataset)):
        raw_img, raw_mask = test_dataset[i]
        norm_img = preprocess_image(raw_img)
        gt_mask = convert_mask(raw_mask)

        t0 = time.time()
        pred_labels = sliding_window_inference(
            model=model,
            image=norm_img,
            patch_size=patch_size,
            overlap=overlap,
            device=device
        )
        patient_times.append(time.time() - t0)

        aggregator.update(pred_labels, gt_mask, compute_sens_spec=True)

    summary = aggregator.summary()
    avg_inference_time = float(np.mean(patient_times)) if patient_times else 0.0

    return {
        "model": ckpt_info["display_name"],
        "architecture": model_name,
        "base_channels": detected_base_channels,
        "parameters": total_params,
        "size_mb": round(model_size_mb, 2),
        "mean_dice": round(summary.get("mean_dice_mean", 0.0), 4),
        "wt_dice": round(summary.get("wt_dice_mean", 0.0), 4),
        "tc_dice": round(summary.get("tc_dice_mean", 0.0), 4),
        "et_dice": round(summary.get("et_dice_mean", 0.0), 4),
        "wt_sensitivity": round(summary.get("wt_sens_mean", 0.0), 4),
        "wt_specificity": round(summary.get("wt_spec_mean", 0.0), 4),
        "tc_sensitivity": round(summary.get("tc_sens_mean", 0.0), 4),
        "tc_specificity": round(summary.get("tc_spec_mean", 0.0), 4),
        "et_sensitivity": round(summary.get("et_sens_mean", 0.0), 4),
        "et_specificity": round(summary.get("et_spec_mean", 0.0), 4),
        "avg_inference_sec": round(avg_inference_time, 2)
    }


def plot_benchmark_chart(results, save_path):
    if not results:
        return

    models = [r["model"] for r in results]
    wt_scores = [r["wt_dice"] for r in results]
    tc_scores = [r["tc_dice"] for r in results]
    et_scores = [r["et_dice"] for r in results]
    mean_scores = [r["mean_dice"] for r in results]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - 1.5 * width, wt_scores, width, label="Whole Tumor (WT)", color="#4A90E2")
    rects2 = ax.bar(x - 0.5 * width, tc_scores, width, label="Tumor Core (TC)", color="#E94E77")
    rects3 = ax.bar(x + 0.5 * width, et_scores, width, label="Enhancing Tumor (ET)", color="#F5A623")
    rects4 = ax.bar(x + 1.5 * width, mean_scores, width, label="Mean Dice", color="#2ECC71")

    ax.set_ylabel("Dice Similarity Coefficient (DSC)", fontsize=13, fontweight="bold")
    ax.set_title("BraTS Model Architecture Benchmark Comparison", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Attach score values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    try:
        plt.savefig(save_path, dpi=150)
        print(f"Benchmark comparison chart saved to: {save_path}")
    except Exception as e:
        alt_path = save_path.replace(".png", "_latest.png")
        try:
            plt.savefig(alt_path, dpi=150)
            print(f"[Warning] Could not overwrite '{save_path}' (file locked by viewer): {e}. Saved to '{alt_path}' instead.")
        except Exception:
            print(f"[Warning] Could not save chart: {e}")
    plt.close()


def display_leaderboard(results, output_file, plot_file):
    for r in results:
        r["parameters"] = int(r["parameters"]) if r.get("parameters") else 0
        r["size_mb"] = float(r["size_mb"]) if r.get("size_mb") else 0.0
        for k in ["mean_dice", "wt_dice", "tc_dice", "et_dice", "wt_sensitivity", "wt_specificity", "tc_sensitivity", "tc_specificity", "et_sensitivity", "et_specificity", "avg_inference_sec"]:
            r[k] = float(r[k]) if r.get(k) else 0.0

    sorted_results = sorted(results, key=lambda x: x["mean_dice"], reverse=True)

    print(f"\n{'='*105}")
    print(f" 1. SEGMENTATION DICE SIMILARITY & COMPUTATIONAL EFFICIENCY")
    print(f"{'='*105}")
    print(f"{'Model':<14} | {'Parameters':<10} | {'Size(MB)':<8} | {'Mean Dice':<9} | {'WT Dice':<8} | {'TC Dice':<8} | {'ET Dice':<8} | {'Latency':<8}")
    print(f"{'-'*14}-+-{'-'*10}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for r in sorted_results:
        print(f"{r['model']:<14} | {r['parameters']:<10,} | {r['size_mb']:<8.2f} | {r['mean_dice']:<9.4f} | {r['wt_dice']:<8.4f} | {r['tc_dice']:<8.4f} | {r['et_dice']:<8.4f} | {r['avg_inference_sec']:<7.1f}s")
    print(f"{'='*105}")

    print(f"\n{'='*105}")
    print(f" 2. CLINICAL DIAGNOSTIC METRICS (SENSITIVITY & SPECIFICITY)")
    print(f"{'='*105}")
    print(f"{'Model':<14} | {'WT Sens':<8} | {'WT Spec':<8} | {'TC Sens':<8} | {'TC Spec':<8} | {'ET Sens':<8} | {'ET Spec':<8}")
    print(f"{'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for r in sorted_results:
        print(f"{r['model']:<14} | {r['wt_sensitivity']:<8.4f} | {r['wt_specificity']:<8.4f} | {r['tc_sensitivity']:<8.4f} | {r['tc_specificity']:<8.4f} | {r['et_sensitivity']:<8.4f} | {r['et_specificity']:<8.4f}")
    print(f"{'='*105}")

    print(f"\n[Detailed CSV Report]  : {os.path.abspath(output_file)}")
    print(f"[Comparison Chart PNG] : {os.path.abspath(plot_file)}")
    print(f"{'='*105}\n")


def main():
    args = parse_args()

    if args.show:
        if os.path.exists(args.output_file):
            with open(args.output_file, "r") as f:
                reader = csv.DictReader(f)
                results = list(reader)
            display_leaderboard(results, args.output_file, args.plot_file)
            return
        else:
            print(f"Benchmark file '{args.output_file}' not found. Please run evaluation first.")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f" BraTS Model Evaluation & Benchmarking Suite")
    print(f"{'='*70}")
    print(f"Checkpoints Directory : {args.outputs_dir}")
    print(f"Device                : {device}")

    checkpoints = discover_checkpoints(args.outputs_dir)
    if not checkpoints:
        print(f"No '*_best.pth' checkpoints found in {args.outputs_dir}. Run training first.")
        return

    print(f"Discovered Checkpoints: {len(checkpoints)}")
    for c in checkpoints:
        print(f"  - {c['filename']} ({c['model_name'].upper()})")

    # Load dataset
    data_dir = find_dataset_directory(args.data_dir)
    _, _, test_patients = split_dataset(data_dir, seed=args.seed)

    if args.max_test_patients:
        test_patients = test_patients[:args.max_test_patients]

    print(f"Test Subjects         : {len(test_patients)}")
    test_dataset = BraTSDataset(test_patients, training=False)

    results = []
    for idx, c in enumerate(checkpoints, start=1):
        print(f"\n[{idx}/{len(checkpoints)}] Evaluating: {c['display_name']} ({c['path']})...")
        res = evaluate_checkpoint(c, test_dataset, device, patch_size=args.patch_size, overlap=args.overlap)
        results.append(res)
        print(f"  -> Mean Dice: {res['mean_dice']} | WT: {res['wt_dice']} | TC: {res['tc_dice']} | ET: {res['et_dice']}")

    # Save CSV
    saved_csv = args.output_file
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    try:
        with open(args.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    except PermissionError:
        saved_csv = args.output_file.replace(".csv", "_all.csv")
        print(f"[Warning] '{args.output_file}' is currently locked (e.g. by Excel). Saving to '{saved_csv}' instead.")
        with open(saved_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    json_file = saved_csv.replace(".csv", ".json")
    try:
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"[Warning] Could not save JSON: {e}")

    # Plot Chart
    plot_benchmark_chart(results, args.plot_file)

    # Print Full Leaderboard
    display_leaderboard(results, saved_csv, args.plot_file)


if __name__ == "__main__":
    main()
