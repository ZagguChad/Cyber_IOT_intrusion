# ======================================
# MODEL TRAINING (GPU OPTIMIZED)
# ======================================
# Train XGBoost, LightGBM, and Random Forest on the optimized feature set.

num_classes = y_train.nunique()
print(f"Number of classes: {num_classes}")
print(f"Training features: {X_train.shape[1]}")

t_start = time.time()

# -------------------------------
# XGBoost (GPU)
# -------------------------------
print("\nTraining XGBoost (GPU)...")
t0 = time.time()
xgb = XGBClassifier(
    n_estimators=300,
    objective="multi:softmax",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda",
    random_state=RANDOM_SEED,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=0
)
xgb.fit(X_train, y_train)
print(f"  XGBoost done in {time.time()-t0:.1f}s")

# -------------------------------
# LightGBM (CPU)
# -------------------------------
print("\nTraining LightGBM...")
t0 = time.time()
lgbm = LGBMClassifier(
    n_estimators=300,
    objective="multiclass",
    num_class=num_classes,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    verbose=-1,
    force_row_wise=True
)
lgbm.fit(X_train, y_train)
print(f"  LightGBM done in {time.time()-t0:.1f}s")

# -------------------------------
# Random Forest
# -------------------------------
print("\nTraining Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    class_weight='balanced'
)
rf.fit(X_train, y_train)
print(f"  Random Forest done in {time.time()-t0:.1f}s")

print(f"\nAll models trained in {time.time()-t_start:.1f}s")
