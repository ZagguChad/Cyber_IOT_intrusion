# ======================================
# SMOTE OVERSAMPLING (TRAINING DATA ONLY)
# ======================================
# Applies SMOTE to balance class distribution in training data.
# Test data remains untouched to preserve real-world evaluation.

print("Class distribution BEFORE SMOTE:")
print(y_train.value_counts().sort_index())
print()

t0 = time.time()

smote = SMOTE(random_state=RANDOM_SEED)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Class distribution AFTER SMOTE:")
print(y_train.value_counts().sort_index())
print(f"\nTraining samples after SMOTE: {len(X_train):,}")
print(f"SMOTE completed in {time.time()-t0:.1f}s")
