#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_draw_stcln_architecture.py
Objective: Generate STCLN architecture diagram for presentation
Author: Amo
Date: March 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
COLOR_INPUT = '#E3F2FD'      # Light Blue
COLOR_SPATIAL = '#C8E6C9'    # Light Green
COLOR_TEMPORAL = '#FFF9C4'   # Light Yellow
COLOR_STA = '#FFCDD2'        # Light Red
COLOR_CLASSIFIER = '#E1BEE7' # Light Purple
COLOR_OUTPUT = '#B2DFDB'     # Teal
COLOR_ARROW = '#333333'      # Dark Gray

# ============================================================================
# 1. INPUT BLOCK
# ============================================================================
input_box = FancyBboxPatch((3.5, 12.2), 3, 0.8, 
                            boxstyle="round,pad=0.1", 
                            facecolor=COLOR_INPUT, edgecolor='black', linewidth=2)
ax.add_patch(input_box)
ax.text(5, 12.6, 'INPUT: SITS (Satellite Image Time Series)', 
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 12.35, 'X ∈ ℝ^(B × T × C × H × W)', 
        ha='center', va='center', fontsize=9, style='italic')
ax.text(5, 12.1, 'B=batch, T=time, C=bands, H×W=image size', 
        ha='center', va='center', fontsize=7, color='gray')

# Arrow down
ax.annotate('', xy=(5, 11.5), xytext=(5, 12.1),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=2))

# ============================================================================
# 2. SPATIAL ENCODER (SEncoder)
# ============================================================================
spatial_box = FancyBboxPatch((2.5, 9.5), 5, 1.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=COLOR_SPATIAL, edgecolor='black', linewidth=2)
ax.add_patch(spatial_box)
ax.text(5, 10.9, 'SPATIAL ENCODER (SEncoder)', 
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 10.5, 'Process each image individually', 
        ha='center', va='center', fontsize=8, style='italic')

# Internal structure
ax.text(3.2, 10.0, 'DoubleConv', ha='center', va='center', fontsize=8, 
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray'))
ax.text(5, 10.0, '→', ha='center', va='center', fontsize=12)
ax.text(6.8, 10.0, 'DoubleConv', ha='center', va='center', fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray'))
ax.text(5, 9.7, 'Conv2D 3x3 + BN + ReLU', ha='center', va='center', fontsize=7)

ax.text(5, 9.2, 'Output: Z_SE ∈ ℝ^(B × H × W × T × C\')', 
        ha='center', va='center', fontsize=8, style='italic')

# Arrow down
ax.annotate('', xy=(5, 8.8), xytext=(5, 9.4),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=2))

# ============================================================================
# 3. TEMPORAL ENCODER (TEncoder)
# ============================================================================
temporal_box = FancyBboxPatch((2, 6.5), 6, 2.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=COLOR_TEMPORAL, edgecolor='black', linewidth=2)
ax.add_patch(temporal_box)
ax.text(5, 8.2, 'TEMPORAL ENCODER (TEncoder)', 
        ha='center', va='center', fontsize=11, fontweight='bold')

# Internal structure
ax.text(2.8, 7.7, 'GN', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray'))
ax.text(3.8, 7.7, '→', ha='center', va='center', fontsize=12)
ax.text(5, 7.7, 'PEnc', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray'))
ax.text(6.2, 7.7, '→', ha='center', va='center', fontsize=12)
ax.text(7.2, 7.7, 'Transformer', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray'))

# Transformer details
ax.text(7.2, 7.2, 'Multi-Head', ha='center', va='center', fontsize=7)
ax.text(7.2, 7.0, 'Attention', ha='center', va='center', fontsize=7)
ax.text(7.2, 6.8, '+ FFN', ha='center', va='center', fontsize=7)

ax.text(5, 6.9, 'Group Normalization + Positional Encoding + Transformer', 
        ha='center', va='center', fontsize=7, color='gray')
ax.text(5, 6.6, 'Output: Z_TE ∈ ℝ^(B × H × W × T × C\')', 
        ha='center', va='center', fontsize=8, style='italic')

# Arrow down
ax.annotate('', xy=(5, 6), xytext=(5, 6.4),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=2))

# ============================================================================
# 4. STA MODULE (Spatiotemporal Attention) - CORE CONTRIBUTION
# ============================================================================
sta_box = FancyBboxPatch((1.5, 3.8), 7, 2, 
                          boxstyle="round,pad=0.1", 
                          facecolor=COLOR_STA, edgecolor='red', linewidth=3)
ax.add_patch(sta_box)
ax.text(5, 5.5, 'SPATIOTEMPORAL ATTENTION (STA) MODULE', 
        ha='center', va='center', fontsize=11, fontweight='bold', color='red')

# q, k, v diagram
ax.text(2.5, 4.9, 'Z_SE', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(2.5, 4.6, '↓ Linear', ha='center', va='center', fontsize=7)
ax.text(2.5, 4.3, 'q', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="circle,pad=0.1", facecolor='white', edgecolor='black'))

ax.text(5, 4.9, 'Z_TE', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(5, 4.6, '↓ Linear', ha='center', va='center', fontsize=7)
ax.text(5, 4.3, 'k', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="circle,pad=0.1", facecolor='white', edgecolor='black'))

ax.text(7.5, 4.9, 'Z_TE', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(7.5, 4.6, '↓ Linear', ha='center', va='center', fontsize=7)
ax.text(7.5, 4.3, 'v', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="circle,pad=0.1", facecolor='white', edgecolor='black'))

# Attention formula
ax.text(5, 3.95, 'α = softmax( (q · kᵀ) / √d )', 
        ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))

ax.text(5, 3.6, 'm = α · v', ha='center', va='center', fontsize=9, fontweight='bold')

# Residual connection
ax.text(5, 3.3, 'Output = Z_TE + m (Residual Connection)', 
        ha='center', va='center', fontsize=8, style='italic', color='red')

# Arrows for q, k, v
ax.annotate('', xy=(2.5, 4.35), xytext=(2.5, 4.7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))
ax.annotate('', xy=(5, 4.35), xytext=(5, 4.7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))
ax.annotate('', xy=(7.5, 4.35), xytext=(7.5, 4.7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))

# Arrow down
ax.annotate('', xy=(5, 3.1), xytext=(5, 3.65),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=2))

# ============================================================================
# 5. CLASSIFIER HEAD
# ============================================================================
classifier_box = FancyBboxPatch((2.5, 2), 5, 0.9, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=COLOR_CLASSIFIER, edgecolor='black', linewidth=2)
ax.add_patch(classifier_box)
ax.text(5, 2.6, 'CLASSIFIER HEAD', 
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 2.3, 'Temporal Pooling (mean) → Linear → Softmax', 
        ha='center', va='center', fontsize=8)

# Arrow down
ax.annotate('', xy=(5, 1.5), xytext=(5, 1.9),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=2))

# ============================================================================
# 6. OUTPUT
# ============================================================================
output_box = FancyBboxPatch((3.5, 0.5), 3, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=COLOR_OUTPUT, edgecolor='black', linewidth=2)
ax.add_patch(output_box)
ax.text(5, 0.9, 'OUTPUT: Crop Type Map', 
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 0.65, 'Y ∈ ℝ^(B × C_out × H × W)', 
        ha='center', va='center', fontsize=8, style='italic')

# ============================================================================
# 7. TWO TRAINING PHASES (Side Diagram)
# ============================================================================
# Left: Pre-training
pretrain_box = FancyBboxPatch((0.2, 1.2), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E0E0E0', edgecolor='black', linewidth=1.5)
ax.add_patch(pretrain_box)
ax.text(1.45, 2.4, 'PRE-TRAINING', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(1.45, 2.1, 'Masked Input', ha='center', va='center', fontsize=7)
ax.text(1.45, 1.9, '→ Encoder →', ha='center', va='center', fontsize=7)
ax.text(1.45, 1.7, 'Reconstruction', ha='center', va='center', fontsize=7)
ax.text(1.45, 1.5, 'Loss: MSE', ha='center', va='center', fontsize=7)

# Right: Fine-tuning
finetune_box = FancyBboxPatch((8.3, 1.2), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E0E0E0', edgecolor='black', linewidth=1.5)
ax.add_patch(finetune_box)
ax.text(9.55, 2.4, 'FINE-TUNING', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(9.55, 2.1, 'Labeled Data', ha='center', va='center', fontsize=7)
ax.text(9.55, 1.9, '→ Encoder →', ha='center', va='center', fontsize=7)
ax.text(9.55, 1.7, 'Classification', ha='center', va='center', fontsize=7)
ax.text(9.55, 1.5, 'Loss: CrossEntropy', ha='center', va='center', fontsize=7)

# Arrow between pretrain and finetune (transfer weights)
ax.annotate('', xy=(8.1, 2.0), xytext=(2.9, 2.0),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2, linestyle='dashed'))
ax.text(5.5, 2.2, 'Transfer Pretrained Weights', ha='center', va='center', 
        fontsize=8, color='green', fontweight='bold')

# ============================================================================
# 8. TITLE
# ============================================================================
ax.text(5, 13.5, 'STCLN: SpatioTemporal Collaborative Learning Network', 
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(5, 13.2, 'For Crop Mapping with Limited Labels', 
        ha='center', va='center', fontsize=10, style='italic', color='gray')

# ============================================================================
# 9. LEGEND
# ============================================================================
legend_x = 8.2
legend_y = 10.5

ax.text(legend_x, legend_y, 'LEGEND:', ha='left', va='center', fontsize=9, fontweight='bold')

# Color boxes
ax.add_patch(patches.Rectangle((legend_x, legend_y-0.3), 0.2, 0.2, facecolor=COLOR_SPATIAL, edgecolor='black'))
ax.text(legend_x+0.3, legend_y-0.2, 'Spatial Encoder', ha='left', va='center', fontsize=7)

ax.add_patch(patches.Rectangle((legend_x, legend_y-0.6), 0.2, 0.2, facecolor=COLOR_TEMPORAL, edgecolor='black'))
ax.text(legend_x+0.3, legend_y-0.5, 'Temporal Encoder', ha='left', va='center', fontsize=7)

ax.add_patch(patches.Rectangle((legend_x, legend_y-0.9), 0.2, 0.2, facecolor=COLOR_STA, edgecolor='red'))
ax.text(legend_x+0.3, legend_y-0.8, 'STA Module (Core Contribution)', ha='left', va='center', fontsize=7, color='red')

ax.add_patch(patches.Rectangle((legend_x, legend_y-1.2), 0.2, 0.2, facecolor=COLOR_CLASSIFIER, edgecolor='black'))
ax.text(legend_x+0.3, legend_y-1.1, 'Classifier', ha='left', va='center', fontsize=7)

# ============================================================================
# 10. FORMULAS BOX
# ============================================================================
formula_box = FancyBboxPatch((0.2, 0.2), 2.8, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FAFAFA', edgecolor='gray', linewidth=1)
ax.add_patch(formula_box)
ax.text(1.6, 1.5, 'KEY FORMULAS', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(1.6, 1.2, 'α = softmax(q·kᵀ/√d)', ha='center', va='center', fontsize=7)
ax.text(1.6, 1.0, 'Output = Z_TE + α·v', ha='center', va='center', fontsize=7)
ax.text(1.6, 0.8, 'L_pretrain = MSE', ha='center', va='center', fontsize=7)
ax.text(1.6, 0.6, 'L_finetune = CE', ha='center', va='center', fontsize=7)

# ============================================================================
# SAVE THE FIGURE
# ============================================================================
plt.tight_layout()
plt.savefig('screenshots/11_stcln_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('screenshots/11_stcln_architecture.pdf', bbox_inches='tight', facecolor='white')
print("✅ Diagram saved to:")
print("   - screenshots/11_stcln_architecture.png")
print("   - screenshots/11_stcln_architecture.pdf")

# Show the figure
plt.show()