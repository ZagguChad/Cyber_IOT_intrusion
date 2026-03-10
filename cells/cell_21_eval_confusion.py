# ======================================
# EVALUATION 3 - CONFUSION MATRIX VISUALIZATION
# ======================================
import matplotlib.pyplot as plt

class_names_list = le.classes_
cm = confusion_matrix(y_test, preds)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left: Count Matrix
im0 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0].text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=9)

# Right: Percentage Matrix
im1 = axes[1].imshow(cm_pct, interpolation='nearest', cmap='Oranges')
axes[1].set_title('Confusion Matrix (%)', fontsize=13, fontweight='bold')
for i in range(cm_pct.shape[0]):
    for j in range(cm_pct.shape[1]):
        axes[1].text(j, i, f'{cm_pct[i,j]:.1f}%', ha='center', va='center',
                     color='white' if cm_pct[i,j] > 50 else 'black', fontsize=9)

for ax in axes:
    ax.set_xticks(range(len(class_names_list)))
    ax.set_yticks(range(len(class_names_list)))
    ax.set_xticklabels(class_names_list, rotation=45, ha='right')
    ax.set_yticklabels(class_names_list)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: confusion_matrix.png")
