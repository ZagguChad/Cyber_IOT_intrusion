# ======================================
# EVALUATION 5 - BASELINE & NO-FEATURE-SELECTION COMPARISON
# ======================================
# Compares: (1) individual models, (2) our ensemble, and
# (3) a baseline ensemble trained WITHOUT feature selection
# to demonstrate the improvement from the hybrid FS module.

from sklearn.linear_model import LogisticRegression

print("Training Baseline Models for Comparison...")
print()

baselines = {}

# --- Individual model baselines (with feature selection) ---

# Logistic Regression
t0 = time.time()
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
lr.fit(X_train_w, y_train)
lr_preds = lr.predict(X_test_w)
baselines['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, lr_preds),
    'Macro F1': f1_score(y_test, lr_preds, average='macro'),
    'Weighted F1': f1_score(y_test, lr_preds, average='weighted')
}
print(f"  Logistic Regression: {time.time()-t0:.1f}s")

# Standalone Random Forest
t0 = time.time()
rf_base = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_SEED)
rf_base.fit(X_train_w, y_train)
rf_preds_base = rf_base.predict(X_test_w)
baselines['Random Forest'] = {
    'Accuracy': accuracy_score(y_test, rf_preds_base),
    'Macro F1': f1_score(y_test, rf_preds_base, average='macro'),
    'Weighted F1': f1_score(y_test, rf_preds_base, average='weighted')
}
print(f"  Random Forest: {time.time()-t0:.1f}s")

# Standalone XGBoost
t0 = time.time()
xgb_base = XGBClassifier(n_estimators=100, tree_method='hist', device='cuda',
                          random_state=RANDOM_SEED, verbosity=0)
xgb_base.fit(X_train_w, y_train)
xgb_preds_base = xgb_base.predict(X_test_w)
baselines['XGBoost'] = {
    'Accuracy': accuracy_score(y_test, xgb_preds_base),
    'Macro F1': f1_score(y_test, xgb_preds_base, average='macro'),
    'Weighted F1': f1_score(y_test, xgb_preds_base, average='weighted')
}
print(f"  XGBoost: {time.time()-t0:.1f}s")

# Our Adaptive Weighted Ensemble (with FS)
baselines['Adaptive Ensemble (w/ FS)'] = {
    'Accuracy': accuracy_score(y_test, preds),
    'Macro F1': f1_score(y_test, preds, average='macro'),
    'Weighted F1': f1_score(y_test, preds, average='weighted')
}

# --- NO FEATURE SELECTION BASELINE ---
# Train an ensemble on ALL original features (without feature selection)
# to demonstrate the improvement from our hybrid FS module.
print("\n  Training NO-Feature-Selection Baseline...")
t0 = time.time()

# Use X_full (all features) without feature selection
X_nofs_train, X_nofs_test, y_nofs_train, y_nofs_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=RANDOM_SEED, stratify=y_full
)

# Apply SMOTE to the no-FS training set
smote_nofs = SMOTE(random_state=RANDOM_SEED)
X_nofs_train_sm, y_nofs_train_sm = smote_nofs.fit_resample(X_nofs_train, y_nofs_train)

# Scale
scaler_nofs = StandardScaler()
X_nofs_train_sc = pd.DataFrame(scaler_nofs.fit_transform(X_nofs_train_sm),
                                columns=X_nofs_train_sm.columns)
X_nofs_test_sc = pd.DataFrame(scaler_nofs.transform(X_nofs_test),
                               columns=X_nofs_test.columns)

# Train a quick ensemble without feature selection
nofs_xgb = XGBClassifier(n_estimators=100, tree_method='hist', device='cuda',
                          random_state=RANDOM_SEED, verbosity=0)
nofs_xgb.fit(X_nofs_train_sc, y_nofs_train_sm)
nofs_preds = nofs_xgb.predict(X_nofs_test_sc)

baselines['XGBoost (NO FS)'] = {
    'Accuracy': accuracy_score(y_nofs_test, nofs_preds),
    'Macro F1': f1_score(y_nofs_test, nofs_preds, average='macro'),
    'Weighted F1': f1_score(y_nofs_test, nofs_preds, average='weighted')
}
print(f"  No-FS Baseline: {time.time()-t0:.1f}s")

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

print(f"\n{'='*70}")
print("FEATURE SELECTION IMPACT:")
print(f"  Accuracy improvement: {(fs_acc - nofs_acc)*100:+.4f}%")
print(f"  Weighted F1 improvement: {(fs_f1 - nofs_f1)*100:+.4f}%")
print(f"  Feature reduction: {len(X_full.columns)} -> {len(final_features)} "
      f"({100*(1 - len(final_features)/len(X_full.columns)):.1f}% reduction)")
