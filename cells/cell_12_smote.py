# ======================================
# SMOTE OVERSAMPLING (TRAINING DATA ONLY)
# ======================================
# Applies SMOTE to balance class distribution in training data.
# Test data remains untouched to preserve real-world evaluation.
# NOTE: Saves pre-SMOTE labels as y_train_orig for the no-FS baseline.

print("Class distribution BEFORE SMOTE:")
print(pd.Series(y_train).value_counts().sort_index())
print()

t0 = time.time()

# Save pre-SMOTE y_train for the no-FS baseline in cell_23
y_train_orig = y_train.copy()

smote = SMOTE(random_state=RANDOM_SEED)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Ensure y_train is a proper Series with reset index for downstream iloc usage
y_train = pd.Series(y_train).reset_index(drop=True)
X_train = X_train.reset_index(drop=True)

print("Class distribution AFTER SMOTE:")
print(y_train.value_counts().sort_index())
print(f"\nTraining samples after SMOTE: {len(X_train):,}")
print(f"SMOTE completed in {time.time()-t0:.1f}s")
