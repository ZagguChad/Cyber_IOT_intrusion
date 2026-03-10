# ======================================
# TRAIN-TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")
print(f"Feature count:    {X_train.shape[1]}")
