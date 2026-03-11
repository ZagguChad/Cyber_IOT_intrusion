# ======================================
# MODEL TRAINING (GPU OPTIMIZED)
# ======================================
# Train XGBoost, LightGBM, and Random Forest on the optimized feature set.
# All models tuned with higher capacity and class-weight awareness.

num_classes = y_train.nunique()
print(f"Number of classes: {num_classes}")
print(f"Training features: {X_train.shape[1]}")
print(f"Training samples: {len(X_train):,}")

t_start = time.time()

# -------------------------------
# XGBoost (GPU) — higher capacity
# -------------------------------
print("\nTraining XGBoost (GPU)...")
t0 = time.time()
xgb = XGBClassifier(
    n_estimators=500,         # Raised from 300
    objective="multi:softprob",  # softprob for better probability estimates
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda",
    random_state=RANDOM_SEED,
    max_depth=12,              # Raised from 10
    learning_rate=0.05,        # Lower LR + more trees = better generalization
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,        # Regularization for minority classes
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    verbosity=0
)
xgb.fit(X_train, y_train)
print(f"  XGBoost done in {time.time()-t0:.1f}s")

# -------------------------------
# LightGBM — class-aware
# -------------------------------
print("\nTraining LightGBM...")
t0 = time.time()
lgbm = LGBMClassifier(
    n_estimators=500,          # Raised from 300
    objective="multiclass",
    num_class=num_classes,
    max_depth=12,              # Raised from 10
    learning_rate=0.05,        # Lower LR + more trees
    num_leaves=63,             # More leaves for complex patterns
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    class_weight='balanced',   # Added class weighting
    random_state=RANDOM_SEED,
    verbose=-1,
    force_row_wise=True
)
lgbm.fit(X_train, y_train)
print(f"  LightGBM done in {time.time()-t0:.1f}s")

# -------------------------------
# Random Forest — deeper trees
# -------------------------------
print("\nTraining Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=500,          # Raised from 300
    max_depth=None,            # Changed from 15 to unlimited depth
    min_samples_leaf=1,        # Allow pure leaves for minority classes
    n_jobs=-1,
    random_state=RANDOM_SEED,
    class_weight='balanced_subsample'  # Better for imbalanced multi-class
)
rf.fit(X_train, y_train)
print(f"  Random Forest done in {time.time()-t0:.1f}s")

print(f"\nAll models trained in {time.time()-t_start:.1f}s")
