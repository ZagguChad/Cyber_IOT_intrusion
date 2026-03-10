# ======================================
# EVALUATION 1 - CLASS DISTRIBUTION VISUALIZATION
# ======================================
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class_names = le.classes_
class_counts = y.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0'][:len(class_names)]
bars = ax.bar(class_names, class_counts.values, color=colors, edgecolor='white', linewidth=1.2)

for bar, count in zip(bars, class_counts.values):
    pct = count / class_counts.sum() * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title('Class Distribution (Full Dataset)', fontsize=14, fontweight='bold')
ax.set_xlabel('Traffic Class')
ax.set_ylabel('Count')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: class_distribution.png")
