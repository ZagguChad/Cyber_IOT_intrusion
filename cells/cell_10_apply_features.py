# ======================================
# APPLY FINAL FEATURE SUBSET & PREPROCESSING
# ======================================
# Filter the dataset to only the selected features.
# Apply scaling/normalization for downstream models.

print("Applying Feature Selection Results")
print("=" * 55)

# Apply final feature subset
X = X_full[final_features].copy()
y = y_full.copy()

print(f"Feature matrix after selection: {X.shape}")
print(f"Features: {list(X.columns)}")

# Handle any remaining NaN / Inf (safety)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# StandardScaler for normalization
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print(f"\nPreprocessing complete:")
print(f"  - NaN/Inf handled")
print(f"  - StandardScaler applied")
print(f"  - Final shape: {X_scaled.shape}")
