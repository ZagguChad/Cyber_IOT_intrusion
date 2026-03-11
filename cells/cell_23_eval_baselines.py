# ======================================
# EVALUATION 5 - BASELINE & NO-FS COMPARISON
# ======================================
# Compares individual models, our ensemble, and a no-FS baseline.
# FIX: No-FS baseline now uses the SAME test set (X_test_full)
#      for a fair comparison instead of creating a separate split.

print("Training Baseline Models for Comparison...")
print()

baselines = {}

# --- Individual model baselines (with feature selection) ---

# Standalone Random Forest (already trained as rf, just predict)
t0 = time.time()
rf_preds_base = rf.predict(X_test_w)
baselines['Random Forest (w/ FS)'] = {
    'Accuracy': accuracy_score(y_test, rf_preds_base),
    'Macro F1': f1_score(y_test, rf_preds_base, average='macro'),
    'Weighted F1': f1_score(y_test, rf_preds_base, average='weighted')
}
print(f"  Random Forest (w/ FS): {time.time()-t0:.1f}s")

# Standalone XGBoost (already trained as xgb, just predict)
t0 = time.time()
xgb_preds_base = xgb.predict(X_test_w)
baselines['XGBoost (w/ FS)'] = {
    'Accuracy': accuracy_score(y_test, xgb_preds_base),
    'Macro F1': f1_score(y_test, xgb_preds_base, average='macro'),
    'Weighted F1': f1_score(y_test, xgb_preds_base, average='weighted')
}
print(f"  XGBoost (w/ FS): {time.time()-t0:.1f}s")

# Standalone LightGBM (already trained as lgbm, just predict)
t0 = time.time()
lgbm_preds_base = lgbm.predict(X_test_w)
baselines['LightGBM (w/ FS)'] = {
    'Accuracy': accuracy_score(y_test, lgbm_preds_base),
    'Macro F1': f1_score(y_test, lgbm_preds_base, average='macro'),
    'Weighted F1': f1_score(y_test, lgbm_preds_base, average='weighted')
}
print(f"  LightGBM (w/ FS): {time.time()-t0:.1f}s")

# Our Adaptive Weighted Ensemble (with FS)
baselines['Adaptive Ensemble (w/ FS)'] = {
    'Accuracy': accuracy_score(y_test, preds),
    'Macro F1': f1_score(y_test, preds, average='macro'),
    'Weighted F1': f1_score(y_test, preds, average='weighted')
}
print(f"  Adaptive Ensemble: (pre-computed)")

# --- NO FEATURE SELECTION BASELINE ---
# FIX: Uses the SAME train/test split for fair comparison.
# Trains XGBoost on ALL original features (no feature selection).
print("\n  Training NO-Feature-Selection Baseline...")
print("  (Same train/test split for fair comparison)")
t0 = time.time()

# Use ALL features from training data (no feature selection)
# Use y_train_orig (pre-SMOTE) to apply SMOTE independently
X_nofs_train = X_train_full.copy()
X_nofs_test = X_test_full.copy()
y_nofs_train = y_train_orig.copy()  # pre-SMOTE labels from cell_12

# Handle NaN/Inf
X_nofs_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_nofs_train.fillna(0, inplace=True)
X_nofs_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_nofs_test.fillna(0, inplace=True)

# Scale — fit on train only
scaler_nofs = StandardScaler()
X_nofs_train_sc = pd.DataFrame(scaler_nofs.fit_transform(X_nofs_train),
                                columns=X_nofs_train.columns)
X_nofs_test_sc = pd.DataFrame(scaler_nofs.transform(X_nofs_test),
                               columns=X_nofs_test.columns)

# Apply SMOTE on training data
smote_nofs = SMOTE(random_state=RANDOM_SEED)
X_nofs_train_sm, y_nofs_train_sm = smote_nofs.fit_resample(X_nofs_train_sc, y_nofs_train)
print(f"    No-FS train size (after SMOTE): {len(X_nofs_train_sm):,}")

# Train XGBoost without feature selection
nofs_xgb = XGBClassifier(n_estimators=200, max_depth=10, tree_method='hist',
                          device='cuda', random_state=RANDOM_SEED, verbosity=0)
nofs_xgb.fit(X_nofs_train_sm, y_nofs_train_sm)
nofs_preds = nofs_xgb.predict(X_nofs_test_sc)

baselines['XGBoost (NO FS)'] = {
    'Accuracy': accuracy_score(y_test, nofs_preds),
    'Macro F1': f1_score(y_test, nofs_preds, average='macro'),
    'Weighted F1': f1_score(y_test, nofs_preds, average='weighted')
}
print(f"    No-FS Baseline done: {time.time()-t0:.1f}s")

# --- Print comparison table ---
print(f"\n{'='*70}")
print(f"{'Model':<30}{'Accuracy':<12}{'Macro F1':<12}{'Weighted F1':<12}")
print("-" * 70)
for model, metrics in baselines.items():
    print(f"{model:<30}{metrics['Accuracy']:<12.4f}{metrics['Macro F1']:<12.4f}{metrics['Weighted F1']:<12.4f}")

# --- Highlight improvement from feature selection ---
fs_acc = baselines['Adaptive Ensemble (w/ FS)']['Accuracy']
nofs_acc = baselines['XGBoost (NO FS)']['Accuracy']
fs_f1 = baselines['Adaptive Ensemble (w/ FS)']['Weighted F1']
nofs_f1 = baselines['XGBoost (NO FS)']['Weighted F1']
fs_m_f1 = baselines['Adaptive Ensemble (w/ FS)']['Macro F1']
nofs_m_f1 = baselines['XGBoost (NO FS)']['Macro F1']

print(f"\n{'='*70}")
print("FEATURE SELECTION IMPACT (same test set, fair comparison):")
print(f"  Accuracy improvement:    {(fs_acc - nofs_acc)*100:+.4f}%")
print(f"  Weighted F1 improvement: {(fs_f1 - nofs_f1)*100:+.4f}%")
print(f"  Macro F1 improvement:    {(fs_m_f1 - nofs_m_f1)*100:+.4f}%")
print(f"  Feature reduction: {len(X_train_full.columns)} -> {len(final_features)} "
      f"({100*(1 - len(final_features)/len(X_train_full.columns)):.1f}% reduction)")
