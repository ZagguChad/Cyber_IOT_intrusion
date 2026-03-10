# ======================================
# EVALUATION 4 - 5-FOLD CROSS VALIDATION
# ======================================
# Uses fresh model clones. Does NOT touch the main trained model.

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

print("Running 5-Fold Stratified Cross-Validation...")
print("(Trains fresh model copies per fold - does NOT touch the main model)")
print()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_w, y_train), 1):
    t0 = time.time()

    X_f_train, X_f_val = X_train_w.iloc[train_idx], X_train_w.iloc[val_idx]
    y_f_train, y_f_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    fold_model = clone(hybrid_model)
    fold_model.fit(X_f_train, y_f_train)
    fold_preds = fold_model.predict(X_f_val)

    fold_acc = accuracy_score(y_f_val, fold_preds)
    fold_f1 = f1_score(y_f_val, fold_preds, average='weighted')

    fold_results.append({'Fold': fold, 'Accuracy': fold_acc, 'Weighted_F1': fold_f1})
    print(f"  Fold {fold}: Accuracy={fold_acc:.4f}  F1={fold_f1:.4f}  ({time.time()-t0:.1f}s)")

cv_df = pd.DataFrame(fold_results)
print(f"\nMean Accuracy: {cv_df['Accuracy'].mean():.4f} +/- {cv_df['Accuracy'].std():.4f}")
print(f"Mean F1:       {cv_df['Weighted_F1'].mean():.4f} +/- {cv_df['Weighted_F1'].std():.4f}")
