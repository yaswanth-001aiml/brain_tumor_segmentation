# 🧠 3D Brain Tumor Segmentation (BraTS 2021)

A complete, end-to-end deep learning framework for multi-modal 3D MRI brain tumor segmentation based on the **BraTS 2021 (Brain Tumor Segmentation)** benchmark.

---

## 📋 Table of Contents
- [Overview & Medical Background](#-overview--medical-background)
- [Architecture Suite](#-architecture-suite)
- [Loss Functions](#-loss-functions)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
  - [1. Verification & Testing](#1-verification--testing)
  - [2. Multi-Modal MRI Visualizer](#2-multi-modal-mri-visualizer)
  - [3. Training a Model](#3-training-a-model)
  - [4. Training All Models (Batch Orchestration)](#4-training-all-models-batch-orchestration)
  - [5. Model Evaluation & Benchmarking](#5-model-evaluation--benchmarking)
  - [6. Clinical Prediction & Volumetric Reporting](#6-clinical-prediction--volumetric-reporting)
- [Hardware & Memory Recommendations](#-hardware--memory-recommendations)

---

## 🔬 Overview & Medical Background

This pipeline processes four core MRI modalities per subject:
1. **T1**: Native T1-weighted MRI (anatomical structure)
2. **T1ce**: T1-weighted contrast-enhanced with Gadolinium (vascularity & BBB disruption)
3. **T2**: T2-weighted MRI (edema & free water)
4. **FLAIR**: Fluid-Attenuated Inversion Recovery (peritumoral hyperintensities)

### Target Tumor Subregions (BraTS Standard)
The model segments three nested clinical tumor subregions:
- **Whole Tumor (WT)**: Encompasses all tumor tissue (Necrotic Core + Edema + Enhancing Tumor)
- **Tumor Core (TC)**: Resectable tumor mass (Necrotic Core + Enhancing Tumor)
- **Enhancing Tumor (ET)**: Actively growing hypervascular tumor margin

---

## 🏗 Architecture Suite

The repository includes five modern 3D neural network architectures designed specifically for volumetric medical imaging:

| Architecture | Identifier | Description |
|---|---|---|
| **Baseline 3D U-Net** | `baseline` | Symmetrical 4-stage encoder-decoder with 3D convolutions and InstanceNorm. |
| **Residual 3D U-Net** | `residual` | Residual skip connections within encoder/decoder blocks for enhanced gradient propagation. |
| **Attention 3D U-Net** | `attention` | Soft 3D Attention Gates on skip connections that filter out irrelevant background activations. |
| **Multi-Scale 3D U-Net** | `multiscale` | Parallel inception-style multi-scale convolutional branches (1x1, 3x3, 5x5, dilated) for variable lesion scales. |
| **Combined 3D U-Net** | `combined` | SOTA hybrid uniting Residual feature learning, Multi-Scale feature fusion, and Attention gating. |

All architectures support **Multi-Scale Deep Supervision** (`--deep_supervision`) to supervise intermediate decoder stages.

---

## 🎯 Loss Functions

Selectable via the `--loss` flag in `train.py`:
- `combined` *(default)*: Soft Dice Loss + Cross-Entropy Loss
- `focal_dice`: Focal Loss + Soft Dice Loss (addresses extreme foreground-background class imbalance)
- `generalized_dice`: Generalized Dice Loss (inversely weights by volume squared for tiny enhancing tumor subregions)
- `focal`: Multi-class Focal Loss
- `dice`: Soft Multi-class Dice Loss
- `ce`: Standard Cross-Entropy Loss

---

## 📁 Project Structure

```
brain_tumor_segmentation/
├── dataset/                        # BraTS patient folders (BraTS2021_*)
├── outputs/                        # Checkpoints, logs, figures, CSV summaries
│   ├── predictions/                # Output segmentations and clinical reports
│   └── test_results/               # Evaluation results and visual comparisons
├── src/
│   ├── data/
│   │   ├── dataset.py              # BraTS volume loader
│   │   ├── augmentation.py         # 3D spatial & intensity transforms
│   │   ├── preprocessing.py        # Z-score normalization & label mapping
│   │   └── split.py                # Patient-level reproducible splits
│   ├── models/
│   │   ├── __init__.py             # get_model factory & MODEL_REGISTRY
│   │   ├── blocks.py               # 3D conv, residual, attention, & multiscale blocks
│   │   ├── baseline.py             # Baseline3DUNet
│   │   ├── residual.py             # Residual3DUNet
│   │   ├── attention.py            # Attention3DUNet
│   │   ├── multiscale.py           # MultiScale3DUNet
│   │   └── combined.py             # Combined3DUNet
│   ├── training/
│   │   └── deep_supervision.py     # Multi-scale deep supervision wrappers
│   ├── losses.py                   # Loss functions & get_loss factory
│   ├── metrics.py                  # BraTS Dice (WT, TC, ET), Sens/Spec, HD95
│   ├── patch_dataset.py            # Bounding-box cached patch extraction
│   ├── train.py                    # Core training engine
│   ├── test.py                     # Sliding-window testing engine
│   └── visualize.py                # Multi-modal volume visualizer
├── download_dataset.py             # BraTS dataset downloader with resume support
├── test_dataset.py                 # Dataset loader sanity test
├── test_model.py                   # Multi-architecture verification test suite
├── train_smoke_test.py             # Rapid end-to-end forward/backward smoke test
├── train_all.py                    # Sequential batch trainer for all models
├── evaluate_all.py                 # Multi-model benchmarking & leaderboard generator
├── predict.py                      # Single-subject inference & volumetric reporting
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Verification & Testing

Verify that your dataset and all five model architectures are functioning correctly:

```bash
# Verify dataset integrity and patient count
python download_dataset.py --verify_only

# Test dataset loading and preprocessing
python test_dataset.py

# Test all 5 model architectures, parameter counts, and gradient flow
python test_model.py --model all

# Rapid 3-second smoke test (forward pass, loss, backward pass, optimizer step)
python train_smoke_test.py --model baseline
```

---

### 2. Multi-Modal MRI Visualizer

Generate publication-ready 5-panel slice views and tri-planar (axial, coronal, sagittal) tumor overlays:

```bash
python src/visualize.py --patient_idx 0 --output outputs/mri_sample_visualization.png
```

---

### 3. Training a Model

Train any of the 5 architectures with modern optimizations:

```bash
# Train Baseline 3D U-Net (Recommended starting recipe)
python src/train.py --model baseline --epochs 20 --base_channels 16 --patch_size 64 --batch_size 1 --lr 1e-3

# Train Residual 3D U-Net with Deep Supervision and Mixed Precision AMP (GPU)
python src/train.py --model residual --deep_supervision --amp --epochs 20 --base_channels 16

# Train Attention 3D U-Net with Focal-Dice Loss
python src/train.py --model attention --loss focal_dice --epochs 20 --base_channels 16

# Resume training from a previous checkpoint
python src/train.py --model baseline --resume outputs/baseline_last.pth --epochs 40
```

#### Key Training Arguments:
- `--model`: `baseline`, `residual`, `attention`, `multiscale`, `combined`
- `--deep_supervision`: Enable multi-scale deep supervision
- `--loss`: `combined`, `focal_dice`, `generalized_dice`, `focal`, `dice`, `ce`
- `--patch_size`: 3D cubic patch size (e.g. `32`, `64`, `96`)
- `--base_channels`: Base feature channels (e.g. `8`, `16`, `32`)
- `--amp`: Automatic Mixed Precision (FP16 on CUDA for 2x faster training and 50% less VRAM)
- `--grad_accum_steps`: Accumulate gradients over N steps (e.g. `2`, `4`)
- `--no_cache`: Disable RAM caching (recommended when training on full 1,000+ patient dataset)
- `--patience`: Early stopping patience (epochs without validation improvement)

---

### 4. Training All Models (Batch Orchestration)

To train all architectures sequentially and generate a consolidated comparison summary:

```bash
python train_all.py --models baseline residual attention multiscale combined --epochs 15 --base_channels 16
```

This will automatically train each model, handle errors, save checkpoints to `outputs/`, and produce:
- `outputs/all_models_training_summary.csv`
- `outputs/all_models_training_summary.json`

---

### 5. Model Evaluation & Benchmarking

Evaluate all trained models on the test split to compute Mean Dice, WT, TC, ET, Sensitivity, Specificity, and latency:

```bash
python evaluate_all.py --outputs_dir outputs --max_test_patients 10
```

This generates:
1. **Interactive Console Leaderboard**
2. **`outputs/model_benchmark_comparison.csv`**: Detailed metrics for every model
3. **`outputs/model_benchmark_comparison.png`**: Side-by-side grouped bar chart comparing all models

---

### 6. Clinical Prediction & Volumetric Reporting

Run inference on any individual patient folder or custom 4-modality MRI scans:

```bash
# Predict on a patient directory using your best checkpoint
python predict.py --checkpoint outputs/baseline_best.pth --patient_dir dataset/BraTS2021_Training_Data/BraTS2021_Training_Data/BraTS2021_01017
```

Outputs generated:
- `outputs/predictions/{PATIENT_ID}_segmentation.nii.gz`: 3D NIfTI segmentation mask preserving original affine & header (compatible with 3D Slicer and ITK-SNAP)
- `outputs/predictions/{PATIENT_ID}_report.png`: Publication-ready multi-planar visual comparison with tumor subregion color coding
- `outputs/predictions/{PATIENT_ID}_report.json`: Clinical report with volumetric tumor burden (WT, TC, ET, ED, NCR in cm³ and voxel counts)

---

## ⚡ Hardware & Memory Recommendations

| Setup | Recommended Settings |
|---|---|
| **NVIDIA GPU (>= 8 GB VRAM)** | `--patch_size 64 --base_channels 16 --batch_size 1 --amp` |
| **High-End GPU (>= 16 GB VRAM)** | `--patch_size 96 --base_channels 32 --batch_size 2 --amp` |
| **CPU / Development** | `--patch_size 32 --base_channels 8 --max_patients 10` |
| **Full BraTS Dataset (1,251 patients)** | Add `--no_cache` to avoid filling system RAM |
