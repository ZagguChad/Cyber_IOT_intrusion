# ======================================
# FEATURE WEIGHTING (MILD)
# ======================================
# Apply very mild SHAP-based weighting to preserve feature distributions.
# Changed from [0.5, 1.0] range to [0.9, 1.0] range.
# The old [0.5, 1.0] range was cutting low-SHAP feature values by 50%,
# which destroyed information needed for minority class detection.

weights_vector = np.array(feature_weights)

# Rescale to [0.9, 1.0] — very mild amplification
# Old: 0.5 + 0.5 * norm (range 0.5 to 1.0) — too aggressive
# New: 0.9 + 0.1 * norm (range 0.9 to 1.0) — gentle nudge
weights_mild = 0.9 + 0.1 * (weights_vector - weights_vector.min()) / (weights_vector.max() - weights_vector.min() + 1e-8)

X_train_w = X_train * weights_mild
X_test_w = X_test * weights_mild

print("Mild SHAP-based feature weighting applied.")
print(f"Weight range: [{weights_mild.min():.4f}, {weights_mild.max():.4f}]")
print(f"(Previous aggressive range was [0.5, 1.0] — now [0.9, 1.0])")
