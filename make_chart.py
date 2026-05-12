import matplotlib.pyplot as plt
import numpy as np

# Your results
models = ['SegRNN', 'iTransformer', 'PatchTST']
etth1 = [0.3801, 0.3857, 0.3898]
etth2 = [0.2891, 0.2938, 0.2983]

# Create bar chart
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, etth1, width, label='ETTh1', color='blue')
bars2 = ax.bar(x + width/2, etth2, width, label='ETTh2', color='orange')

# Labels
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('MSE (Lower is Better)', fontsize=14)
ax.set_title('Figure 1: Model Performance Comparison (ETTh1 vs ETTh2)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add numbers on top of bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, 
            f'{height:.4f}', ha='center', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, 
            f'{height:.4f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()