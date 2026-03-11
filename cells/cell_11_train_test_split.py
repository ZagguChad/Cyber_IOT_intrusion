# ======================================
# (DEPRECATED — SPLIT MOVED TO CELL 03b)
# ======================================
# Train-test split is now performed in cell_03b_train_test_split.py
# BEFORE feature selection and scaling, to prevent data leakage.
#
# This cell is intentionally left as a pass-through to maintain
# cell numbering consistency with the notebook.

print("=" * 55)
print("[INFO] Train-test split was already done in Cell 03b")
print("       (before feature selection & scaling)")
print(f"       Training samples: {len(X_train):,}")
print(f"       Testing samples:  {len(X_test):,}")
print(f"       Feature count:    {X_train.shape[1]}")
print("=" * 55)
