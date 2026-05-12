"""
Model Performance Comparison Chart for ETTh1 vs ETTh2
Run this script to generate the bar chart for your progress report
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for professional looking chart
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ============================================
# YOUR RESULTS - DO NOT CHANGE THESE NUMBERS
# ============================================

# Results for each model on ETTh1 and ETTh2
models = ['SegRNN', 'iTransformer', 'PatchTST']
etth1_mse = [0.3801, 0.3857, 0.3898]      # MSE on ETTh1
etth2_mse = [0.2891, 0.2938, 0.2983]      # MSE on ETTh2

# ============================================
# CREATE THE BAR CHART
# ============================================

# Create figure with specific size (good for screenshots)
fig, ax = plt.subplots(figsize=(10, 6))

# Set positions for bars
x = np.arange(len(models))  # [0, 1, 2]
width = 0.35                 # Width of each bar

# Create bars
bars1 = ax.bar(x - width/2, etth1_mse, width, label='ETTh1', color='#1f77b4', edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, etth2_mse, width, label='ETTh2', color='#ff7f0e', edgecolor='black', linewidth=1)

# Add value labels on top of each bar
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add labels and title
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('MSE (Lower is Better)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison on ETTh1 vs ETTh2\n(Input 96, Predict 96)', fontsize=14, fontweight='bold')

# Set x-axis ticks
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)

# Add legend
ax.legend(loc='upper right', fontsize=12)

# Add grid for better readability
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('model_comparison_chart.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved as: model_comparison_chart.png")

# Show the chart (a window will pop up)
plt.show()