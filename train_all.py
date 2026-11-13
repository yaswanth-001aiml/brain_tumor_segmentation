import os
import sys
import time
import json
import csv
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Sequential Trainer for All BraTS Models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help=f"List of model architectures to train. Default: {list(MODEL_REGISTRY.keys())}"
    )
    parser.add_argument("--base_channels", type=int, default=16, help="Base channel count (e.g., 16, 32)")
    parser.add_argument("--deep_supervision", action="store_true", help="Enable multi-scale deep supervision")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per model")
    parser.add_argument("--batch_size", type=int, default=1, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size (e.g., 32, 64, 96)")
    parser.add_argument("--patches_per_patient", type=int, default=2, help="Patches per volume")
    parser.add_argument("--loss", type=str, default="combined", help="Loss function")
    parser.add_argument("--data_dir", type=str, default="dataset/BraTS2021_Training_Data/BraTS2021_Training_Data")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Outputs directory")
    parser.add_argument("--max_patients", type=int, default=None, help="Limit number of patients for quick training")
    parser.add_argument("--no_cache", action="store_true", help="Disable in-memory volume caching")
    parser.add_argument("--amp", action="store_true", help="Enable Mixed Precision AMP on CUDA")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_model_history(output_dir, prefix):
    json_path = os.path.join(output_dir, f"{prefix}_history.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r") as f:
                history = json.load(f)
            if history:
                best_mean_dice = max(h.get("val_mean_dice", 0.0) for h in history)
                last = history[-1]
                return {
                    "best_val_mean_dice": best_mean_dice,
                    "final_val_wt": last.get("val_wt_dice", 0.0),
                    "final_val_tc": last.get("val_tc_dice", 0.0),
                    "final_val_et": last.get("val_et_dice", 0.0),
                    "final_train_loss": last.get("train_loss", 0.0),
                    "final_val_loss": last.get("val_loss", 0.0),
                }
        except Exception:
            pass
    return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" BraTS Multi-Model Sequential Training Suite")
    print(f"{'='*70}")
    print(f"Models to Train    : {args.models}")
    print(f"Epochs Per Model   : {args.epochs}")
    print(f"Base Channels      : {args.base_channels}")
    print(f"Patch Size         : {args.patch_size}x{args.patch_size}x{args.patch_size}")
    print(f"Deep Supervision   : {args.deep_supervision}")
    print(f"Loss Function      : {args.loss}")
    print(f"Output Directory   : {os.path.abspath(args.output_dir)}")
    print(f"{'='*70}\n")

    summary_records = []
    suite_start = time.time()

    for idx, model_name in enumerate(args.models, start=1):
        prefix = f"{model_name}_ds" if args.deep_supervision else model_name
        print(f"\n[{idx}/{len(args.models)}] >>> Starting Training for: {model_name.upper()} ({prefix})")
        print(f"{'-'*70}")

        cmd = [
            sys.executable,
            "-u",
            os.path.join(PROJECT_ROOT, "src", "train.py"),
            "--model", model_name,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--base_channels", str(args.base_channels),
            "--patch_size", str(args.patch_size),
            "--patches_per_patient", str(args.patches_per_patient),
            "--loss", args.loss,
            "--lr", str(args.lr),
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
            "--grad_accum_steps", str(args.grad_accum_steps),
            "--seed", str(args.seed),
        ]

        if args.deep_supervision:
            cmd.append("--deep_supervision")
        if args.no_cache:
            cmd.append("--no_cache")
        if args.amp:
            cmd.append("--amp")
        if args.max_patients is not None:
            cmd.extend(["--max_patients", str(args.max_patients)])

        m_start = time.time()
        success = True
        try:
            res = subprocess.run(cmd, check=True)
            elapsed = time.time() - m_start
            print(f"Model {model_name.upper()} training completed in {elapsed // 60:.0f}m {elapsed % 60:.1f}s.")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - m_start
            print(f"Error training {model_name}: Process exited with code {e.returncode}")
            success = False

        # Gather metrics from saved history
        metrics = load_model_history(args.output_dir, prefix) or {}
        record = {
            "model": model_name,
            "deep_supervision": args.deep_supervision,
            "base_channels": args.base_channels,
            "epochs": args.epochs,
            "best_val_mean_dice": metrics.get("best_val_mean_dice", "N/A"),
            "val_wt_dice": metrics.get("final_val_wt", "N/A"),
            "val_tc_dice": metrics.get("final_val_tc", "N/A"),
            "val_et_dice": metrics.get("final_val_et", "N/A"),
            "train_loss": metrics.get("final_train_loss", "N/A"),
            "val_loss": metrics.get("final_val_loss", "N/A"),
            "duration_sec": round(elapsed, 1),
            "status": "COMPLETED" if success else "FAILED"
        }
        summary_records.append(record)

    total_time = time.time() - suite_start

    # Save summary files
    summary_csv = os.path.join(args.output_dir, "all_models_training_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)

    summary_json = os.path.join(args.output_dir, "all_models_training_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_records, f, indent=2)

    # Print summary leaderboard
    print(f"\n\n{'='*75}")
    print(f" ALL MODELS TRAINING BENCHMARK SUMMARY")
    print(f"{'='*75}")
    print(f"{'Model':<12} | {'DS':<5} | {'Best Mean':<10} | {'WT':<7} | {'TC':<7} | {'ET':<7} | {'Time':<8} | {'Status':<9}")
    print(f"{'-'*12}-+-{'-'*5}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}")
    for r in summary_records:
        bm = f"{r['best_val_mean_dice']:.4f}" if isinstance(r['best_val_mean_dice'], float) else str(r['best_val_mean_dice'])
        wt = f"{r['val_wt_dice']:.4f}" if isinstance(r['val_wt_dice'], float) else str(r['val_wt_dice'])
        tc = f"{r['val_tc_dice']:.4f}" if isinstance(r['val_tc_dice'], float) else str(r['val_tc_dice'])
        et = f"{r['val_et_dice']:.4f}" if isinstance(r['val_et_dice'], float) else str(r['val_et_dice'])
        dur = f"{r['duration_sec']:.0f}s"
        ds_str = "Yes" if r['deep_supervision'] else "No"
        print(f"{r['model']:<12} | {ds_str:<5} | {bm:<10} | {wt:<7} | {tc:<7} | {et:<7} | {dur:<8} | {r['status']:<9}")

    print(f"{'='*75}")
    print(f"Total Suite Runtime: {total_time // 60:.0f}m {total_time % 60:.1f}s")
    print(f"Summary saved to   : {summary_csv}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
