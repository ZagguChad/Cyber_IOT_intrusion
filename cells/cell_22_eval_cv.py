# ======================================
# EVALUATION 4 - 5-FOLD CROSS VALIDATION
# ======================================
# FIX: Upgraded from 3-fold to 5-fold for better generalization estimate.
# FIX: Added macro F1 and balanced accuracy per fold.
# FIX: Increased subsample to 300K for more representative evaluation.
# Uses a SINGLE XGBoost model (not the full ensemble) for speed,
# but with higher capacity than before.

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

print("Running 5-Fold Stratified Cross-Validation...")
print("(Uses XGBoost on subsample for speed — validates generalization)")
print()

# Subsample to keep runtime reasonable (max 300K rows)
CV_SAMPLE = min(300_000, len(X_train_w))
cv_idx = np.random.choice(len(X_train_w), CV_SAMPLE, replace=False)
X_cv = X_train_w.iloc[cv_idx].reset_index(drop=True)
y_cv = y_train.iloc[cv_idx].reset_index(drop=True)

print(f"CV dataset: {len(X_cv):,} samples x {X_cv.shape[1]} features")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv), 1):
    t0 = time.time()

    X_f_train, X_f_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
    y_f_train, y_f_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

    # XGBoost with moderate capacity (not the full ensemble, but stronger than before)
    fold_xgb = XGBClassifier(
        n_estimators=200,       # Raised from 100
        max_depth=10,           # Raised from 8
        learning_rate=0.05,
        tree_method='hist',
        device='cuda',
        random_state=RANDOM_SEED,
        verbosity=0
    )
    fold_xgb.fit(X_f_train, y_f_train)
    fold_preds = fold_xgb.predict(X_f_val)

    fold_acc = accuracy_score(y_f_val, fold_preds)
    fold_bal_acc = balanced_accuracy_score(y_f_val, fold_preds)
    fold_w_f1 = f1_score(y_f_val, fold_preds, average='weighted')
    fold_m_f1 = f1_score(y_f_val, fold_preds, average='macro')

    fold_results.append({
        'Fold': fold,
        'Accuracy': fold_acc,
        'Balanced_Acc': fold_bal_acc,
        'Weighted_F1': fold_w_f1,
        'Macro_F1': fold_m_f1
    })
    elapsed = time.time() - t0
    print(f"  Fold {fold}: Acc={fold_acc:.4f}  Bal.Acc={fold_bal_acc:.4f}  "
          f"W-F1={fold_w_f1:.4f}  M-F1={fold_m_f1:.4f}  ({elapsed:.1f}s)")

cv_df = pd.DataFrame(fold_results)
print(f"\n{'Metric':<20}{'Mean':<12}{'Std':<12}")
print("-" * 44)
for col in ['Accuracy', 'Balanced_Acc', 'Weighted_F1', 'Macro_F1']:
    print(f"{col:<20}{cv_df[col].mean():<12.4f}{cv_df[col].std():<12.4f}")

print(f"\n[INFO] 5-fold CV on {CV_SAMPLE:,} samples validates generalization.")
print(f"       Low std indicates stable model performance across folds.")
