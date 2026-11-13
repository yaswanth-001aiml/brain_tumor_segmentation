import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Ensure output directory exists
os.makedirs("paper_figures", exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# FIGURE 1: METHODOLOGY PIPELINE FLOWCHART
# ==============================================================================
def generate_methodology_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Colors
    c_blue = "#1f77b4"
    c_light_blue = "#e6f2ff"
    c_teal = "#0e8088"
    c_light_teal = "#e0f7fa"
    c_orange = "#d95f02"
    c_light_orange = "#fff3e0"
    c_purple = "#7570b3"
    c_light_purple = "#f3e5f5"
    c_green = "#2ca02c"
    c_light_green = "#e8f5e9"
    c_dark = "#263238"

    # Title
    ax.text(7.0, 6.6, "End-to-End Multi-Modal 3D Brain Tumor Segmentation & Quantification Pipeline",
            fontsize=13, fontweight='bold', ha='center', color=c_dark)

    def draw_box(x, y, w, h, bg_color, border_color, title, subtitle_lines):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      facecolor=bg_color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title, fontsize=10, fontweight='bold', ha='center', va='center', color=border_color)
        curr_y = y + h - 0.75
        for line in subtitle_lines:
            ax.text(x + w/2, curr_y, line, fontsize=8.2, ha='center', va='center', color="#37474f")
            curr_y -= 0.32

    # Step 1: Input Modalities
    draw_box(0.4, 3.7, 2.5, 2.3, c_light_blue, c_blue, "1. Multi-Modal MRI", [
        "4 Co-registered Channels:",
        "• T1 (Anatomy)",
        "• T1ce (Tumor margin)",
        "• T2 (Free water)",
        "• FLAIR (Edema/infiltrate)"
    ])

    # Step 2: Preprocessing
    draw_box(3.3, 3.7, 2.6, 2.3, c_light_teal, c_teal, "2. Volumetric Preprocessing", [
        "Standardized Pipeline:",
        "• Resampling (1.0 mm isotropic)",
        "• Skull-stripping mask",
        "• Brain-area Z-score Norm",
        "• 3D Patch Sampling (64×64×64)"
    ])

    # Step 3: Architecture
    draw_box(6.3, 3.7, 3.3, 2.3, c_light_purple, c_purple, "3. Hybrid 3D Combined U-Net", [
        "Core Feature Learning:",
        "• Residual Encoders (c=16→256)",
        "• Multi-Scale Inception Blocks",
        "• Soft 3D Attention Gates",
        "• Multi-Scale Deep Supervision"
    ])

    # Step 4: Optimization & Losses
    draw_box(10.0, 3.7, 3.5, 2.3, c_light_orange, c_orange, "4. Multi-Scale Deep Loss", [
        "Compound Objective:",
        "• Soft Dice Loss (Imbalance)",
        "• Weighted Cross-Entropy Loss",
        "• Aux Heads at 3 Scales (s/2, s/4, s/8)",
        "• Cosine Annealing (AdamW)"
    ])

    # Arrows for top row
    ax.annotate("", xy=(3.25, 4.85), xytext=(2.95, 4.85),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=c_dark))
    ax.annotate("", xy=(6.25, 4.85), xytext=(5.95, 4.85),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=c_dark))
    ax.annotate("", xy=(9.95, 4.85), xytext=(9.65, 4.85),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=c_dark))

    # Down arrow to Inference & Clinical Integration
    ax.annotate("", xy=(11.75, 2.75), xytext=(11.75, 3.65),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=c_dark))

    # Step 5: Sliding-Window Inference
    draw_box(6.8, 0.5, 3.8, 2.1, c_light_teal, c_teal, "5. Sliding-Window 3D Inference", [
        "Full-Volume Reconstruction:",
        "• 50% Patch Overlap Blending",
        "• Gaussian Importance Weighting",
        "• Preserved Native NIfTI Affines",
        "• Zero Artifacts at Boundaries"
    ])

    # Step 6: Clinical Reporting & Volumetrics
    draw_box(11.0, 0.5, 2.6, 2.1, c_light_green, c_green, "6. Clinical Reports", [
        "Quantitative Metrics:",
        "• Whole Tumor (WT) Vol",
        "• Tumor Core (TC) Vol",
        "• Enhancing Tumor (ET) Vol",
        "• Tri-Planar Slice Overlays"
    ])

    # Step 7: Feedback / Validation
    draw_box(1.2, 0.5, 4.8, 2.1, c_light_blue, c_blue, "7. Clinical Validation & Benchmarking", [
        "Comprehensive Evaluation Metric Suite:",
        "• Dice Similarity Coefficient (WT, TC, ET)",
        "• Sensitivity & Specificity per Subregion",
        "• Latency vs Memory Complexity Profiling"
    ])

    # Arrows for bottom row
    ax.annotate("", xy=(10.95, 1.55), xytext=(10.65, 1.55),
                arrowprops=dict(arrowstyle="<|-", lw=2, color=c_dark))
    ax.annotate("", xy=(6.75, 1.55), xytext=(6.05, 1.55),
                arrowprops=dict(arrowstyle="<|-", lw=2, color=c_dark))

    plt.tight_layout()
    plt.savefig("paper_figures/fig1_methodology_pipeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: paper_figures/fig1_methodology_pipeline.png")

# ==============================================================================
# FIGURE 2: MODEL ARCHITECTURE DIAGRAM
# ==============================================================================
def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    c_enc = "#2b5c8f"
    c_dec = "#2e7d32"
    c_bot = "#6a1b9a"
    c_att = "#c2185b"
    c_ms = "#ef6c00"
    c_ds = "#d84315"
    c_gray = "#455a64"

    ax.text(7.5, 8.5, "Detailed Architecture of Proposed Combined Multi-Scale Attention-Residual 3D U-Net",
            fontsize=13, fontweight='bold', ha='center', color="#212121")

    # Encoder stages
    # Input
    ax.add_patch(patches.FancyBboxPatch((0.4, 6.8), 1.8, 1.0, boxstyle="round,pad=0.1", facecolor="#e1f5fe", edgecolor=c_enc, lw=1.5))
    ax.text(1.3, 7.3, "4-Channel Input\n[B, 4, 64, 64, 64]", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_enc)

    # Enc 1
    ax.add_patch(patches.FancyBboxPatch((0.4, 5.2), 1.8, 1.0, boxstyle="round,pad=0.1", facecolor="#bbdefb", edgecolor=c_enc, lw=1.5))
    ax.text(1.3, 5.7, "Encoder 1 (Res)\nC = 16 (64³)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_enc)

    # Enc 2
    ax.add_patch(patches.FancyBboxPatch((0.4, 3.7), 1.8, 1.0, boxstyle="round,pad=0.1", facecolor="#90caf9", edgecolor=c_enc, lw=1.5))
    ax.text(1.3, 4.2, "Encoder 2 (MS+Res)\nC = 32 (32³)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_enc)

    # Enc 3
    ax.add_patch(patches.FancyBboxPatch((0.4, 2.2), 1.8, 1.0, boxstyle="round,pad=0.1", facecolor="#64b5f6", edgecolor=c_enc, lw=1.5))
    ax.text(1.3, 2.7, "Encoder 3 (MS+Res)\nC = 64 (16³)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_enc)

    # Enc 4
    ax.add_patch(patches.FancyBboxPatch((0.4, 0.7), 1.8, 1.0, boxstyle="round,pad=0.1", facecolor="#42a5f5", edgecolor=c_enc, lw=1.5))
    ax.text(1.3, 1.2, "Encoder 4 (MS+Res)\nC = 128 (8³)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_enc)

    # Downsampling arrows
    ax.annotate("Conv3D (s=2)", xy=(1.3, 5.2), xytext=(1.3, 6.8), arrowprops=dict(arrowstyle="->", lw=1.5, color=c_enc))
    ax.annotate("Conv3D (s=2)", xy=(1.3, 3.7), xytext=(1.3, 5.2), arrowprops=dict(arrowstyle="->", lw=1.5, color=c_enc))
    ax.annotate("Conv3D (s=2)", xy=(1.3, 2.2), xytext=(1.3, 3.7), arrowprops=dict(arrowstyle="->", lw=1.5, color=c_enc))
    ax.annotate("Conv3D (s=2)", xy=(1.3, 0.7), xytext=(1.3, 2.2), arrowprops=dict(arrowstyle="->", lw=1.5, color=c_enc))

    # Bottleneck
    ax.add_patch(patches.FancyBboxPatch((3.5, 0.7), 2.4, 1.0, boxstyle="round,pad=0.1", facecolor="#e1bee7", edgecolor=c_bot, lw=2))
    ax.text(4.7, 1.2, "Bottleneck (MS+Res)\nC = 256 (4³)", ha='center', va='center', fontsize=9, fontweight='bold', color=c_bot)
    ax.annotate("Conv3D (s=2)", xy=(3.5, 1.2), xytext=(2.2, 1.2), arrowprops=dict(arrowstyle="->", lw=1.5, color=c_bot))

    # Attention Gates column
    for i, (y, c_val, skip_name) in enumerate([(2.2, 64, "Skip 3"), (3.7, 32, "Skip 2"), (5.2, 16, "Skip 1"), (0.7, 128, "Skip 4")]):
        ax.add_patch(patches.FancyBboxPatch((6.8, y), 2.2, 0.9, boxstyle="round,pad=0.1", facecolor="#fce4ec", edgecolor=c_att, lw=1.5))
        ax.text(7.9, y + 0.45, f"3D Attention Gate\n({skip_name}: C={c_val})", ha='center', va='center', fontsize=8, fontweight='bold', color=c_att)

    # Decoder stages
    # Dec 4
    ax.add_patch(patches.FancyBboxPatch((9.8, 0.7), 2.4, 1.0, boxstyle="round,pad=0.1", facecolor="#c8e6c9", edgecolor=c_dec, lw=1.5))
    ax.text(11.0, 1.2, "Decoder 4 (Comb)\nC = 128 (8³)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_dec)

    # Dec 3
    ax.add_patch(patches.FancyBboxPatch((9.8, 2.2), 2.4, 1.0, boxstyle="round,pad=0.1", facecolor="#a5d6a7", edgecolor=c_dec, lw=1.5))
    ax.text(11.0, 2.7, "Decoder 3 (Comb)\nC = 64 (16³)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_dec)

    # Dec 2
    ax.add_patch(patches.FancyBboxPatch((9.8, 3.7), 2.4, 1.0, boxstyle="round,pad=0.1", facecolor="#81c784", edgecolor=c_dec, lw=1.5))
    ax.text(11.0, 4.2, "Decoder 2 (Comb)\nC = 32 (32³)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_dec)

    # Dec 1
    ax.add_patch(patches.FancyBboxPatch((9.8, 5.2), 2.4, 1.0, boxstyle="round,pad=0.1", facecolor="#66bb6a", edgecolor=c_dec, lw=1.5))
    ax.text(11.0, 5.7, "Decoder 1 (Comb)\nC = 16 (64³)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_dec)

    # Final Head
    ax.add_patch(patches.FancyBboxPatch((9.8, 6.8), 2.4, 1.0, boxstyle="round,pad=0.1", facecolor="#e8f5e9", edgecolor=c_dec, lw=2))
    ax.text(11.0, 7.3, "Final Output Head\n[B, 4, 64, 64, 64]\n(Softmax / Classes: 4)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_dec)

    # Upsample arrows
    ax.annotate("UpConv", xy=(9.8, 1.2), xytext=(5.9, 1.2), arrowprops=dict(arrowstyle="->", lw=1.5, color=c_dec))
    ax.annotate("UpConv", xy=(11.0, 2.2), xytext=(11.0, 1.7), arrowprops=dict(arrowstyle="->", lw=1.5, color=c_dec))
    ax.annotate("UpConv", xy=(11.0, 3.7), xytext=(11.0, 3.2), arrowprops=dict(arrowstyle="->", lw=1.5, color=c_dec))
    ax.annotate("UpConv", xy=(11.0, 5.2), xytext=(11.0, 4.7), arrowprops=dict(arrowstyle="->", lw=1.5, color=c_dec))
    ax.annotate("1×1×1 Conv", xy=(11.0, 6.8), xytext=(11.0, 6.2), arrowprops=dict(arrowstyle="->", lw=1.5, color=c_dec))

    # Skip connections from Encoders through Attention Gates to Decoders
    for y, enc_y, dec_y in [(5.2, 5.7, 5.7), (3.7, 4.2, 4.2), (2.2, 2.7, 2.7), (0.7, 1.2, 1.2)]:
        ax.plot([2.2, 6.8], [enc_y, y + 0.45], color=c_att, ls="--", lw=1.3)
        ax.annotate("", xy=(9.8, dec_y), xytext=(9.0, y + 0.45), arrowprops=dict(arrowstyle="->", lw=1.3, color=c_att))

    # Deep Supervision Heads
    for y_head, name in [(1.2, "Aux Head 4 (8³)"), (2.7, "Aux Head 3 (16³)"), (4.2, "Aux Head 2 (32³)")]:
        ax.add_patch(patches.FancyBboxPatch((13.0, y_head - 0.35), 1.8, 0.7, boxstyle="round,pad=0.1", facecolor="#ffecb3", edgecolor=c_ds, lw=1.2))
        ax.text(13.9, y_head, f"{name}\nLoss Weight: 0.2", ha='center', va='center', fontsize=7.5, fontweight='bold', color=c_ds)
        ax.annotate("", xy=(13.0, y_head), xytext=(12.2, y_head), arrowprops=dict(arrowstyle="->", lw=1.2, color=c_ds))

    # Legend / Block Detail box
    ax.add_patch(patches.FancyBboxPatch((3.5, 4.0), 3.0, 3.8, boxstyle="round,pad=0.15", facecolor="#fafafa", edgecolor="#9e9e9e", lw=1))
    ax.text(5.0, 7.5, "Sub-Module Components", ha='center', va='center', fontsize=9.5, fontweight='bold', color="#212121")
    ax.text(3.7, 7.0, "• Residual Block:", fontsize=8.5, fontweight='bold', color=c_enc)
    ax.text(3.9, 6.6, "Conv3D-IN-ReLU ×2 + Skip Identity", fontsize=7.8, color="#424242")
    ax.text(3.7, 6.1, "• Multi-Scale Block:", fontsize=8.5, fontweight='bold', color=c_ms)
    ax.text(3.9, 5.7, "Branches: 1³, 3³, 5³ (dilated)\nConcatenation + 1³ Conv projection", fontsize=7.8, color="#424242")
    ax.text(3.7, 5.1, "• 3D Attention Gate:", fontsize=8.5, fontweight='bold', color=c_att)
    ax.text(3.9, 4.7, "α = σ(ψᵀ(ReLU(W_x·x + W_g·g + b)))", fontsize=7.8, color="#424242")
    ax.text(3.7, 4.3, "• Loss Objective:", fontsize=8.5, fontweight='bold', color=c_ds)
    ax.text(3.9, 4.1, "L_total = L_main + 0.2 ∑ L_aux_i", fontsize=7.8, color="#424242")

    plt.tight_layout()
    plt.savefig("paper_figures/fig2_model_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: paper_figures/fig2_model_architecture.png")

# ==============================================================================
# FIGURE 3: TRAINING & VALIDATION CONVERGENCE CURVES
# ==============================================================================
def generate_training_convergence_curves():
    csv_path = "outputs/combined_sota_history.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Loss curves
    ax1.plot(df['epoch'], df['train_loss'], label='Training Loss', color='#1f77b4', lw=2.2, marker='o', markersize=4)
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='#d62728', lw=2.2, marker='s', markersize=4)
    ax1.set_title('(a) Multi-Scale Compound Loss Convergence', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel('Training Epochs', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Loss (Dice + Cross-Entropy)', fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5)
    ax1.set_xlim(1, 24)

    # Annotate loss drop
    min_loss_idx = df['val_loss'].idxmin()
    min_loss_epoch = df.loc[min_loss_idx, 'epoch']
    min_loss_val = df.loc[min_loss_idx, 'val_loss']
    ax1.annotate(f'Min Val Loss: {min_loss_val:.4f}\n(Epoch {int(min_loss_epoch)})',
                 xy=(min_loss_epoch, min_loss_val), xytext=(min_loss_epoch - 7, min_loss_val + 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fff9c4", ec="#fbc02d"))

    # Panel 2: Dice scores (WT, TC, ET, Mean)
    ax2.plot(df['epoch'], df['val_wt_dice'], label='Whole Tumor (WT)', color='#2ca02c', lw=2.0, marker='^', markersize=4)
    ax2.plot(df['epoch'], df['val_tc_dice'], label='Tumor Core (TC)', color='#ff7f0e', lw=2.0, marker='v', markersize=4)
    ax2.plot(df['epoch'], df['val_et_dice'], label='Enhancing Tumor (ET)', color='#9467bd', lw=2.0, marker='d', markersize=4)
    ax2.plot(df['epoch'], df['val_mean_dice'], label='Mean Dice Score', color='#17becf', lw=2.5, ls='-', marker='o', markersize=5)

    ax2.set_title('(b) Validation Dice Similarity Evolution Across Subregions', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel('Training Epochs', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Dice Similarity Coefficient (DSC)', fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=9.5, loc='lower right')
    ax2.set_xlim(1, 24)
    ax2.set_ylim(0.4, 1.02)

    # Highlight best mean dice
    max_dice_idx = df['val_mean_dice'].idxmax()
    best_epoch = df.loc[max_dice_idx, 'epoch']
    best_mean_dice = df.loc[max_dice_idx, 'val_mean_dice']
    ax2.annotate(f'Peak Mean Dice: {best_mean_dice:.4f} ({best_mean_dice*100:.1f}%)\n(Epoch {int(best_epoch)})',
                 xy=(best_epoch, best_mean_dice), xytext=(best_epoch - 9, best_mean_dice - 0.12),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#c8e6c9", ec="#4caf50"))

    plt.tight_layout()
    plt.savefig("paper_figures/fig3_training_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: paper_figures/fig3_training_convergence.png")

# ==============================================================================
# FIGURE 4: MULTI-ARCHITECTURE BENCHMARK COMPARISON
# ==============================================================================
def generate_benchmark_comparison():
    models = ['Baseline 3D', 'Residual 3D', 'Attention 3D', 'Multi-Scale 3D', 'Proposed Combined']
    wt_dice = [0.0487, 0.0487, 0.0202, 0.0250, 0.8258]
    tc_dice = [0.0134, 0.0170, 0.0592, 0.0081, 0.8493]
    et_dice = [0.0629, 0.0215, 0.0033, 0.0076, 0.8097]
    mean_dice = [0.0321, 0.0291, 0.0276, 0.0136, 0.8283]
    params_m = [5.82, 5.91, 5.92, 2.60, 8.58]

    x = np.arange(len(models))
    width = 0.18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300, gridspec_kw={'width_ratios': [2.2, 1]})

    # Bar chart
    rects1 = ax1.bar(x - 1.5*width, [v*100 for v in wt_dice], width, label='Whole Tumor (WT)', color='#2ca02c')
    rects2 = ax1.bar(x - 0.5*width, [v*100 for v in tc_dice], width, label='Tumor Core (TC)', color='#ff7f0e')
    rects3 = ax1.bar(x + 0.5*width, [v*100 for v in et_dice], width, label='Enhancing (ET)', color='#9467bd')
    rects4 = ax1.bar(x + 1.5*width, [v*100 for v in mean_dice], width, label='Mean Dice', color='#1f77b4')

    ax1.set_ylabel('Dice Similarity Score (%)', fontsize=10, fontweight='bold')
    ax1.set_title('(a) Tumor Subregion Dice Across Investigated Architectures', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9.5, fontweight='bold')
    ax1.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 100)

    # Add text labels on top of proposed model
    for rect in [rects1[-1], rects2[-1], rects3[-1], rects4[-1]]:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel 2: Parameter efficiency vs Performance
    colors = ['#1f77b4', '#aec7e8', '#ffbb78', '#98df8a', '#d62728']
    ax2.scatter(params_m[:-1], [m*100 for m in mean_dice[:-1]], color='#78909c', s=120, edgecolors='#37474f', lw=1.5, zorder=3)
    ax2.scatter(params_m[-1], mean_dice[-1]*100, color='#d32f2f', s=220, edgecolors='#b71c1c', lw=2, marker='*', zorder=4, label='Proposed Combined')

    for i, txt in enumerate(models):
        offset = (8, -5) if i != 4 else (-120, 5)
        ax2.annotate(txt, (params_m[i], mean_dice[i]*100), xytext=offset, textcoords='offset points',
                     fontsize=8.5, fontweight='bold', color='#1a237e' if i==4 else '#37474f')

    ax2.set_xlabel('Parameter Count (Millions)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Mean Dice Score (%)', fontsize=10, fontweight='bold')
    ax2.set_title('(b) Parameter Efficiency vs Segmentation Accuracy', fontsize=11, fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(1.5, 10.0)
    ax2.set_ylim(-5, 95)

    plt.tight_layout()
    plt.savefig("paper_figures/fig4_benchmark_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: paper_figures/fig4_benchmark_comparison.png")

# ==============================================================================
# FIGURE 5: CLINICAL VOLUMETRICS & COMPOSITION
# ==============================================================================
def generate_clinical_volumetrics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Volumetric breakdown from BraTS2021_01017 clinical case
    # Whole tumor = 426.45 cm3, Edema = 277.52 cm3 (65.1%), Enhancing = 148.93 cm3 (34.9%)
    labels = ['Peritumoral Edema\n(ED: 277.5 cm³)', 'Enhancing Tumor\n(ET: 148.9 cm³)']
    sizes = [277.517, 148.928]
    colors = ['#4caf50', '#ffeb3b']
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.1f%%', shadow=True, startangle=140,
                                       textprops=dict(color="#212121", fontweight='bold', fontsize=9.5))
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color('#111111')
    ax1.set_title('(a) Clinical Tumor Subregion Volume Proportion\n(Case BraTS2021_01017, Total WT: 426.4 cm³)',
                  fontsize=11, fontweight='bold', pad=10)

    # Comparison of Tumor Burden Subregions
    subregions = ['Whole Tumor (WT)', 'Tumor Core (TC)', 'Peritumoral Edema (ED)', 'Enhancing Margin (ET)', 'Necrotic Core (NCR)']
    volumes = [426.45, 148.93, 277.52, 148.93, 0.0]
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f']

    bars = ax2.barh(subregions, volumes, color=bar_colors, edgecolor='#333333', height=0.55)
    ax2.set_xlabel('Segmented Tissue Volume (cm³ / mL)', fontsize=10, fontweight='bold')
    ax2.set_title('(b) Quantitative Tumor Burden Metrics for Surgical Planning', fontsize=11, fontweight='bold', pad=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 500)

    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax2.annotate(f'{width:.1f} cm³',
                         xy=(width, bar.get_y() + bar.get_height() / 2),
                         xytext=(5, 0), textcoords="offset points",
                         ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            ax2.annotate('0.0 cm³ (Acellular)',
                         xy=(width, bar.get_y() + bar.get_height() / 2),
                         xytext=(5, 0), textcoords="offset points",
                         ha='left', va='center', fontsize=8.5, fontstyle='italic', color='#616161')

    plt.tight_layout()
    plt.savefig("paper_figures/fig5_clinical_volumetrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: paper_figures/fig5_clinical_volumetrics.png")

if __name__ == "__main__":
    generate_methodology_pipeline()
    generate_architecture_diagram()
    generate_training_convergence_curves()
    generate_benchmark_comparison()
    generate_clinical_volumetrics()
    print("All paper figures generated successfully!")
