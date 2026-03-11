# ======================================
# EVALUATION 2 - DETAILED PERFORMANCE METRICS TABLE
# ======================================
# FIX: Added ROC-AUC (one-vs-rest), balanced accuracy,
#      and per-class confidence intervals for small classes.

from sklearn.metrics import (precision_score, recall_score, f1_score,
                             balanced_accuracy_score, roc_auc_score)
from scipy import stats

class_names_list = le.classes_

accuracy_val  = accuracy_score(y_test, preds)
bal_acc_val   = balanced_accuracy_score(y_test, preds)
macro_f1      = f1_score(y_test, preds, average='macro')
weighted_f1   = f1_score(y_test, preds, average='weighted')
macro_prec    = precision_score(y_test, preds, average='macro')
weighted_prec = precision_score(y_test, preds, average='weighted')
macro_rec     = recall_score(y_test, preds, average='macro')
weighted_rec  = recall_score(y_test, preds, average='weighted')

print("=" * 55)
print("OVERALL METRICS")
print("=" * 55)
print(f"Accuracy:           {accuracy_val:.4f}")
print(f"Balanced Accuracy:  {bal_acc_val:.4f}")
print(f"Macro Precision:    {macro_prec:.4f}")
print(f"Macro Recall:       {macro_rec:.4f}")
print(f"Macro F1:           {macro_f1:.4f}")
print(f"Weighted Precision: {weighted_prec:.4f}")
print(f"Weighted Recall:    {weighted_rec:.4f}")
print(f"Weighted F1:        {weighted_f1:.4f}")

# --- ROC-AUC (if ensemble supports predict_proba) ---
try:
    y_proba = hybrid_model.predict_proba(X_test_w)
    roc_auc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    roc_auc_ovo = roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro')
    print(f"ROC-AUC (OvR):      {roc_auc_ovr:.4f}")
    print(f"ROC-AUC (OvO):      {roc_auc_ovo:.4f}")
except Exception as e:
    print(f"ROC-AUC: Could not compute ({e})")

print("\n" + "=" * 55)
print("PER-CLASS REPORT")
print("=" * 55)
print(classification_report(y_test, preds, target_names=class_names_list, digits=4))

# --- Per-class confidence intervals (Wilson score interval) ---
print("=" * 55)
print("PER-CLASS RECALL WITH 95% CONFIDENCE INTERVALS")
print("=" * 55)
print(f"{'Class':<15}{'Recall':<10}{'95% CI':<25}{'N_test':<10}")
print("-" * 60)

cm_eval = confusion_matrix(y_test, preds)
for i, cls in enumerate(class_names_list):
    n_total = cm_eval[i].sum()
    n_correct = cm_eval[i, i]
    recall_cls = n_correct / n_total if n_total > 0 else 0

    # Wilson score confidence interval
    if n_total > 0:
        z = 1.96  # 95% CI
        p_hat = recall_cls
        denominator = 1 + z**2 / n_total
        centre = (p_hat + z**2 / (2 * n_total)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denominator
        ci_low = max(0, centre - margin)
        ci_high = min(1, centre + margin)
        ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"
    else:
        ci_str = "N/A"

    flag = " ⚠️ SMALL SAMPLE" if n_total < 100 else ""
    print(f"{cls:<15}{recall_cls:<10.4f}{ci_str:<25}{n_total:<10}{flag}")

print(f"\n[INFO] Classes with < 100 test samples have wide confidence intervals.")
print(f"       Results for these classes should be interpreted with caution.")
