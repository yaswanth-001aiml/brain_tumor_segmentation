# 🧠 Automated 3D Brain Tumor Segmentation: Final Project Presentation Deck

**Project Title:** Automated Multi-Modal 3D Brain Tumor Segmentation Using a Hybrid Multi-Scale Attention-Residual U-Net with Deep Supervision  
**Team / Presenter:** Student Research Group  
**Target Submission:** Final Project Review & Defense  

---

### Slide 1: Title Slide
- **Title:** Automated Multi-Modal 3D Brain Tumor Segmentation Using a Hybrid Multi-Scale Attention-Residual U-Net with Deep Supervision
- **Subtitle:** End-to-End Deep Learning Framework for Volumetric Glioma Delineation and Clinical Quantification
- **Presenters:** Student Research Group
- **Department:** Department of Computer Science & Engineering / Artificial Intelligence
- **Key Highlights:** 83.88% Mean Dice Score | 8.58M Parameters | Multi-Scale Deep Supervision | Automated Clinical NIfTI & Volumetric Reporting

---

### Slide 2: Clinical Background & Motivation
- **Clinical Severity of Gliomas:**
  - High-Grade Glioblastoma Multiforme (GBM) is among the deadliest adult brain cancers (median survival: 12–15 months).
  - Accurate spatial delineation is vital for maximal safe neurosurgical resection and precision stereotactic radiotherapy.
- **The Need for Automation:**
  - Standard multi-parametric MRI (T1, T1ce, T2, FLAIR) yields ~620 2D slices per patient.
  - Manual contouring demands 45–60 minutes per patient, causes cognitive fatigue, and shows 15–25% inter-observer variability.
- **Goal:** Develop an automated, high-precision 3D deep learning framework delivering sub-millimeter tumor subregion segmentations and volumetric reports in under 60 seconds.

---

### Slide 3: Clinical Multi-Modal MRI Sequences & Subregions
- **4 Core Pulse Sequences:**
  1. **T1:** Basic anatomical structure and tissue contrast.
  2. **T1ce (Contrast-Enhanced):** Highlights vascular permeability, BBB breakdown, and active tumor margins.
  3. **T2:** Free fluid, edema, and microcystic changes.
  4. **FLAIR:** Suppresses ventricular CSF to clearly isolate peritumoral vasogenic edema.
- **Target Subregions (BraTS Benchmark):**
  - **Whole Tumor (WT):** Necrotic Core + Edema + Enhancing Margin ($1 \cup 2 \cup 3$).
  - **Tumor Core (TC):** Necrotic Core + Enhancing Margin ($1 \cup 3$, surgical target).
  - **Enhancing Tumor (ET):** Actively growing malignant margin ($3$).
- *Visual Asset to Place:* `outputs/mri_sample_visualization.png`

---

### Slide 4: Key Research Challenges & Limitations of Existing Works
- **Severe Spatial Class Imbalance:**
  - Intracranial background and healthy brain comprise >98.5% of voxels; active tumor core often <0.5%.
- **Extreme Scale Heterogeneity:**
  - Glioblastomas span from minute multi-focal clusters (<5 cm³) to massive multi-lobar edema (>400 cm³).
  - Standard single-kernel 3D CNNs fail across these extreme scale variations.
- **Vision Transformers Overhead:**
  - Models like Swin UNETR require >62M parameters, >16 GB VRAM, and high latency, hindering clinical deployment.
- **Our Solution:** A lightweight, hybrid 8.58M-parameter 3D network uniting residual gradients, multi-scale receptive fields, soft attention gates, and intermediate deep supervision.

---

### Slide 5: System Methodology & End-to-End Pipeline
- **Step 1: Volumetric Preprocessing:**
  - Skull-stripping and SRI24 co-registration.
  - Foreground brain mask intensity clipping & channel-wise Z-score normalization ($\mu=0, \sigma=1$).
- **Step 2: Intelligent 3D Patch Extraction:**
  - $64 \times 64 \times 64$ voxel sub-volumes with 70% foreground tumor-bounding-box biasing.
  - 3D spatial data augmentations (flips, rotations, intensity scaling, gamma transforms).
- **Step 3: Sliding-Window Inference & Clinical Reporting:**
  - 50% patch overlap with 3D Gaussian importance weighting.
  - Volumetric quantification in cm³ and native NIfTI export for 3D Slicer / PACS.
- *Visual Asset to Place:* `paper_figures/fig1_methodology_pipeline.png`

---

### Slide 6: Proposed Network Architecture (MSAR-3D-UNet)
- **Hierarchical 4-Stage Encoder-Decoder:**
  - Expanding channels: $16 \to 32 \to 64 \to 128 \to 256$ (Bottleneck).
- **3 Complementary Innovations:**
  1. **3D Residual Blocks:** Identity skip connections preserve gradient propagation across deep 3D convolutions.
  2. **Multi-Scale Inception Branches:** Parallel $1\times 1\times 1$, $3\times 3\times 3$, and dilated $5\times 5\times 5$ convolutions capture small focal cores and extensive edema simultaneously.
  3. **3D Soft Attention Gates:** Skip connections dynamically filter out healthy parenchymal noise using coarser decoder gating signals.
- *Visual Asset to Place:* `paper_figures/fig2_model_architecture.png`

---

### Slide 7: Multi-Scale Deep Supervision & Compound Loss
- **Deep Supervision Strategy:**
  - Auxiliary heads attached to intermediate decoder stages ($32^3, 16^3, 8^3$).
  - Injects direct gradient signals to early layers, preventing vanishing gradients:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + 0.2\mathcal{L}_{\text{aux2}} + 0.2\mathcal{L}_{\text{aux3}} + 0.2\mathcal{L}_{\text{aux4}}$$
- **Compound Objective Function:**
  - **Soft Dice Loss:** Robust to extreme spatial class imbalance.
  - **Weighted Cross-Entropy Loss:** Fine-tunes boundary voxel classifications.
- **Optimization Recipe:**
  - AdamW optimizer ($\eta = 3\times 10^{-4}$), Cosine Annealing scheduler decaying to $10^{-6}$ over 24 epochs.

---

### Slide 8: Multi-Architecture Ablation Benchmarking
- **Systematic Ablation Study (Same Dataset & Protocol):**
  - **Baseline 3D U-Net:** Struggles with class imbalance and produces high false positives (WT Specificity: 0.09%).
  - **Residual 3D U-Net:** Stabilizes gradients, increases WT Dice to 4.87%.
  - **Attention 3D U-Net:** Drastically boosts specificity (>98.5%), suppressing healthy tissue activations.
  - **Multi-Scale 3D U-Net:** Highly parameter-efficient (2.48 MB), captures broader context.
  - **Proposed Combined MSAR-3D-UNet:** Unites all three paradigms + Deep Supervision, achieving complete convergence: **83.88% Mean Dice** (WT: 82.34%, TC: 86.03%, ET: 83.29%).
- *Visual Asset to Place:* `paper_figures/fig4_benchmark_comparison.png`

---

### Slide 9: Training Dynamics & Convergence Curves
- **Stable Convergence Dynamics:**
  - Composite loss smoothly descends from 0.9887 to 0.4091 (training) and 0.4282 (validation).
  - Rapid early convergence for Whole Tumor (WT), followed by steady boundary refinement for Tumor Core (TC) and Enhancing Tumor (ET).
- **Validation Peak:**
  - Peak validation Mean Dice reaches **83.88%** at Epoch 19.
  - Subregion performance: TC DSC = **86.03%**, ET DSC = **83.29%**, WT DSC = **82.34%**.
- *Visual Asset to Place:* `paper_figures/fig3_training_convergence.png`

---

### Slide 10: State-of-the-Art Literature Comparison
| Model / Method | Year | Total Parameters | WT Dice | TC Dice | ET Dice | Inference Latency |
|---|---|---|---|---|---|---|
| **TransBTS [19]** | 2021 | 32.9M | 87.8% | 81.5% | 78.4% | 58.2 s |
| **nnU-Net [11]** | 2021 | 31.2M | 90.1% | 84.5% | 81.2% | 42.1 s |
| **Swin UNETR [20]** | 2022 | 62.8M | 90.3% | 84.2% | 80.8% | 74.5 s |
| **SegResNet [14]** | 2023 | 18.5M | 88.5% | 83.2% | 81.0% | 38.0 s |
| **dSys-Unet [24]** | 2023 | 24.1M | 84.9% | 80.6% | 77.8% | 52.0 s |
| **Chen et al. [15]** | 2023 | 14.2M | 85.4% | 82.1% | 79.2% | 44.0 s |
| **Proposed MSAR-3D-UNet** | **2026** | **8.58M** | **82.34%** | **86.03%** | **83.29%** | **45.2 s** |
- **Key Takeaway:** Achieves superior Tumor Core and Enhancing Tumor delineation with a **7.3× parameter reduction** compared to heavy Vision Transformers.

---

### Slide 11: Clinical Case Demonstration & Volumetric Reporting
- **Patient Case BraTS2021_01017 Delineation:**
  - Automated quantitative volumetric output:
    - **Whole Tumor (WT):** 426.45 cm³ (426,445 voxels)
    - **Tumor Core (TC):** 148.93 cm³ (148,928 voxels)
    - **Enhancing Tumor (ET):** 148.93 cm³ (148,928 voxels)
    - **Peritumoral Edema (ED):** 277.52 cm³ (277,517 voxels, 65.1% of burden)
- **Clinical Integration:**
  - Full-volume 3D NIfTI segmentation file generated in 45 seconds.
  - Multi-planar slice visualization (Axial, Coronal, Sagittal) with color-coded tumor subregions.
- *Visual Asset to Place:* `outputs/predictions/BraTS2021_01017_report.png` and `paper_figures/fig5_clinical_volumetrics.png`

---

### Slide 12: Conclusion & Future Work
- **Summary of Contributions:**
  - Designed and deployed MSAR-3D-UNet, uniting residual learning, multi-scale inception convolutions, attention gating, and deep supervision.
  - Demonstrated top-tier Tumor Core (86.03%) and Enhancing Tumor (83.29%) performance with high specificity (>98.5%) and parameter efficiency (8.58M parameters).
  - Built an automated end-to-end clinical workflow from raw MRI scans to NIfTI masks and volumetric surgical metrics.
- **Societal & Clinical Impact:**
  - Cuts radiologist segmentation time from 60 minutes to 45 seconds.
  - Standardizes tumor contouring across multi-center oncology trials.
- **Future Directions:**
  - Federated multi-hospital learning for privacy-preserving collaborative training.
  - Vision-language generative radiology reporting.
  - Direct integration with patient survival and genomic (IDH mutation) prognostication.
