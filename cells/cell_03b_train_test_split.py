# ======================================
# CELL 3b - EARLY TRAIN-TEST SPLIT
# ======================================
# Split BEFORE feature selection and scaling to prevent data leakage.
# Feature selection and scaler will only see training data.

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y_full,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_full
)

print(f"Train-Test Split (BEFORE feature selection & scaling)")
print(f"{'='*55}")
print(f"Training samples: {len(X_train_full):,}")
print(f"Testing samples:  {len(X_test_full):,}")
print(f"Feature count:    {X_train_full.shape[1]}")
print(f"\nClass distribution (train):")
print(y_train.value_counts().sort_index())
print(f"\nClass distribution (test):")
print(y_test.value_counts().sort_index())
print(f"\n[INFO] Feature selection will operate ONLY on training data.")
