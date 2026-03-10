# ======================================
# EVALUATION 5 - BASELINE MODEL COMPARISON
# ======================================
# Trains lightweight baselines on the SAME weighted features for fair comparison.

from sklearn.linear_model import LogisticRegression

print("Training Baseline Models for Comparison...")
print("(Using same SHAP-weighted features)")
print()

baselines = {}

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
rf_preds = rf_base.predict(X_test_w)
baselines['Random Forest'] = {
    'Accuracy': accuracy_score(y_test, rf_preds),
    'Macro F1': f1_score(y_test, rf_preds, average='macro'),
    'Weighted F1': f1_score(y_test, rf_preds, average='weighted')
}
print(f"  Random Forest: {time.time()-t0:.1f}s")

# Standalone XGBoost
t0 = time.time()
xgb_base = XGBClassifier(n_estimators=100, tree_method='hist', device='cuda',
                          random_state=RANDOM_SEED, verbosity=0)
xgb_base.fit(X_train_w, y_train)
xgb_preds = xgb_base.predict(X_test_w)
baselines['XGBoost'] = {
    'Accuracy': accuracy_score(y_test, xgb_preds),
    'Macro F1': f1_score(y_test, xgb_preds, average='macro'),
    'Weighted F1': f1_score(y_test, xgb_preds, average='weighted')
}
print(f"  XGBoost: {time.time()-t0:.1f}s")

# Our Ensemble
baselines['Hybrid Ensemble'] = {
    'Accuracy': accuracy_score(y_test, preds),
    'Macro F1': f1_score(y_test, preds, average='macro'),
    'Weighted F1': f1_score(y_test, preds, average='weighted')
}

# Print comparison table
print(f"\n{'Model':<25}{'Accuracy':<12}{'Macro F1':<12}{'Weighted F1':<12}")
print("-" * 60)
for model, metrics in baselines.items():
    print(f"{model:<25}{metrics['Accuracy']:<12.4f}{metrics['Macro F1']:<12.4f}{metrics['Weighted F1']:<12.4f}")
