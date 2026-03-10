# ======================================
# EVALUATION 6 - COMPARISON BAR CHART
# ======================================
import matplotlib.pyplot as plt

model_names = list(baselines.keys())
metrics_keys = ['Accuracy', 'Macro F1', 'Weighted F1']
colors = ['#607D8B', '#9E9E9E', '#78909C', '#1976D2']

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(metrics_keys))
width = 0.18
offsets = np.linspace(-(len(model_names)-1)*width/2,
                       (len(model_names)-1)*width/2, len(model_names))

for idx, (model, offset) in enumerate(zip(model_names, offsets)):
    values = [baselines[model][m] for m in metrics_keys]
    bars = ax.bar(x + offset, values, width, label=model, color=colors[idx % len(colors)])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metrics_keys, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Model Comparison - Feature-Selected Pipeline', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_comparison.png")
