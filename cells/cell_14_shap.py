# ======================================
# SHAP EXPLAINABILITY
# ======================================
# Compute SHAP values for the XGBoost model to verify
# that selected features contribute meaningfully.

import shap

print("Running SHAP analysis (this may take 1-2 minutes)...")
t0 = time.time()

# Background and explanation samples
background = X_train.sample(200, random_state=RANDOM_SEED)
sample_data = X_train.sample(300, random_state=RANDOM_SEED)

explainer = shap.Explainer(xgb.predict, background)
shap_values = explainer(sample_data)

# Compute mean |SHAP| per feature
feature_importance = np.mean(np.abs(shap_values.values), axis=0)
shap_df = pd.DataFrame({
    'Feature': X_train.columns,
    'SHAP_Importance': feature_importance
}).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)

print(f"\nSHAP analysis completed in {time.time()-t0:.1f}s")
print(f"\nSHAP Feature Importance (selected features):")
print(f"{'Feature':<25}{'Mean |SHAP|':<15}")
print("-" * 40)
for _, row in shap_df.iterrows():
    print(f"{row['Feature']:<25}{row['SHAP_Importance']:.6f}")

# Verify all selected features have non-trivial SHAP importance
zero_shap = shap_df[shap_df['SHAP_Importance'] < 1e-6]
if len(zero_shap) == 0:
    print("\nAll selected features have meaningful SHAP contributions.")
else:
    print(f"\nWARNING: {len(zero_shap)} features have near-zero SHAP importance:")
    print(zero_shap['Feature'].tolist())

# Compute feature weights (normalized SHAP importance)
normalized_shap = feature_importance / feature_importance.max()
feature_weights = 0.5 + 0.5 * normalized_shap  # scale to [0.5, 1.0]
print(f"\nFeature weights computed (range: {feature_weights.min():.3f} - {feature_weights.max():.3f})")
