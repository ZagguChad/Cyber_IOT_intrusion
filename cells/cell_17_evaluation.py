# ======================================
# EVALUATION METRICS
# ======================================

from sklearn.metrics import balanced_accuracy_score

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='weighted')
recall = recall_score(y_test, preds, average='weighted')
f1 = f1_score(y_test, preds, average='weighted')
bal_acc = balanced_accuracy_score(y_test, preds)

# Macro metrics (important for imbalanced datasets)
macro_prec = precision_score(y_test, preds, average='macro')
macro_rec = recall_score(y_test, preds, average='macro')
macro_f1 = f1_score(y_test, preds, average='macro')

print("\n========== FINAL PERFORMANCE ==========")
print(f"Accuracy          : {accuracy:.6f}")
print(f"Balanced Accuracy : {bal_acc:.6f}")
print(f"Precision (W)     : {precision:.6f}")
print(f"Recall (W)        : {recall:.6f}")
print(f"F1-Score (W)      : {f1:.6f}")
print(f"Precision (Macro) : {macro_prec:.6f}")
print(f"Recall (Macro)    : {macro_rec:.6f}")
print(f"F1-Score (Macro)  : {macro_f1:.6f}")

print("\n========== CLASSIFICATION REPORT ==========")
class_names = le.classes_
print(classification_report(y_test, preds, target_names=class_names, digits=4))

print("========== CONFUSION MATRIX ==========")
cm = confusion_matrix(y_test, preds)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df)

# Summary
print("\n========== PIPELINE SUMMARY ==========")
print(f"Original features:    {len(X_train_full.columns)}")
print(f"After correlation:    {len(features_after_corr)}")
print(f"RFE selected:         {len(rfe_selected)}")
print(f"Red Ant selected:     {len(ant_selected_features)}")
print(f"Final (aggregated):   {len(final_features)}")
print(f"Models in ensemble:   XGBoost (GPU), LightGBM, Random Forest")
print(f"SHAP weighted:        Yes")
print(f"SMOTE applied:        Yes (training only)")
print(f"Data leakage:         NONE (split before FS & scaling)")
