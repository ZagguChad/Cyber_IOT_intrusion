# ======================================
# EVALUATION 4 - 3-FOLD CROSS VALIDATION
# ======================================
# Uses a SINGLE lightweight XGBoost model (not the full ensemble)
# on a subsample. This avoids the massive cost of cloning and
# training 3 models x 5 folds on SMOTE-inflated data.
#
# WHY NOT CLONE THE ENSEMBLE?
# The VotingClassifier with XGB+LGBM+RF (300 trees each) on
# SMOTE-inflated data (~2-5M rows) takes 10-20 min PER FOLD.
# 5 folds = 50-100 min and causes Jupyter to disconnect.

from sklearn.model_selection import StratifiedKFold

print("Running 3-Fold Stratified Cross-Validation...")
print("(Uses lightweight XGBoost on subsample for speed)")
print()

# Subsample to keep runtime reasonable (max 200K rows)
CV_SAMPLE = min(200_000, len(X_train_w))
cv_idx = np.random.choice(len(X_train_w), CV_SAMPLE, replace=False)
X_cv = X_train_w.iloc[cv_idx].reset_index(drop=True)
y_cv = y_train.iloc[cv_idx].reset_index(drop=True)

print(f"CV dataset: {len(X_cv):,} samples x {X_cv.shape[1]} features")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv), 1):
    t0 = time.time()

    X_f_train, X_f_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
    y_f_train, y_f_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

    # Lightweight XGBoost (not the full ensemble)
    fold_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=8,
        tree_method='hist',
        device='cuda',
        random_state=RANDOM_SEED,
        verbosity=0
    )
    fold_xgb.fit(X_f_train, y_f_train)
    fold_preds = fold_xgb.predict(X_f_val)

    fold_acc = accuracy_score(y_f_val, fold_preds)
    fold_f1 = f1_score(y_f_val, fold_preds, average='weighted')

    fold_results.append({'Fold': fold, 'Accuracy': fold_acc, 'Weighted_F1': fold_f1})
    elapsed = time.time() - t0
    print(f"  Fold {fold}: Accuracy={fold_acc:.4f}  F1={fold_f1:.4f}  ({elapsed:.1f}s)")

cv_df = pd.DataFrame(fold_results)
print(f"\nMean Accuracy: {cv_df['Accuracy'].mean():.4f} +/- {cv_df['Accuracy'].std():.4f}")
print(f"Mean F1:       {cv_df['Weighted_F1'].mean():.4f} +/- {cv_df['Weighted_F1'].std():.4f}")
