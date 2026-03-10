# ======================================
# EVALUATION 2 - DETAILED PERFORMANCE METRICS TABLE
# ======================================
from sklearn.metrics import precision_score, recall_score, f1_score

class_names_list = le.classes_

accuracy_val  = accuracy_score(y_test, preds)
macro_f1  = f1_score(y_test, preds, average='macro')
weighted_f1 = f1_score(y_test, preds, average='weighted')
macro_prec = precision_score(y_test, preds, average='macro')
weighted_prec = precision_score(y_test, preds, average='weighted')
macro_rec = recall_score(y_test, preds, average='macro')
weighted_rec = recall_score(y_test, preds, average='weighted')

print("=" * 55)
print("OVERALL METRICS")
print("=" * 55)
print(f"Accuracy:          {accuracy_val:.4f}")
print(f"Macro Precision:   {macro_prec:.4f}")
print(f"Macro Recall:      {macro_rec:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted Precision:{weighted_prec:.4f}")
print(f"Weighted Recall:   {weighted_rec:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")

print("\n" + "=" * 55)
print("PER-CLASS REPORT")
print("=" * 55)
print(classification_report(y_test, preds, target_names=class_names_list, digits=4))
