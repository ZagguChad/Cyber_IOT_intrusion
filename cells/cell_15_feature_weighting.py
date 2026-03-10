# ======================================
# FEATURE WEIGHTING
# ======================================
# Scale feature values using SHAP importance to amplify
# influential signals before ensemble prediction.

weights_vector = np.array(feature_weights)

X_train_w = X_train * weights_vector
X_test_w = X_test * weights_vector

print("Dynamic SHAP-based feature weighting applied.")
print(f"Weight range: [{weights_vector.min():.4f}, {weights_vector.max():.4f}]")
