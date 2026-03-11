# ======================================
# APPLY FINAL FEATURE SUBSET & PREPROCESSING
# ======================================
# Filter BOTH train and test to selected features.
# FIX: Fit StandardScaler on TRAINING data only, then transform both.
# This prevents test data statistics from leaking into normalization.

print("Applying Feature Selection Results")
print("=" * 55)

# Apply final feature subset to TRAIN and TEST separately
X_train = X_train_full[final_features].copy()
X_test  = X_test_full[final_features].copy()
# y_train and y_test already exist from cell_03b

print(f"Training feature matrix: {X_train.shape}")
print(f"Testing feature matrix:  {X_test.shape}")
print(f"Features: {list(X_train.columns)}")

# Handle any remaining NaN / Inf (safety)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.fillna(0, inplace=True)

# StandardScaler — FIT on training data ONLY, transform both
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train),      # fit + transform on TRAIN
    columns=X_train.columns,
    index=X_train.index
)
X_test = pd.DataFrame(
    scaler.transform(X_test),           # transform only on TEST (no fit!)
    columns=X_test.columns,
    index=X_test.index
)

print(f"\nPreprocessing complete:")
print(f"  - NaN/Inf handled")
print(f"  - StandardScaler fitted on TRAINING data only (leakage-free)")
print(f"  - Train shape: {X_train.shape}")
print(f"  - Test shape:  {X_test.shape}")
print(f"\n[INFO] Scaler statistics computed from training data only.")
print(f"       Test data was transformed using training statistics.")
