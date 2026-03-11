# ======================================
# ADAPTIVE WEIGHTED SOFT VOTING ENSEMBLE
# ======================================
# Instead of equal weights, each model's contribution is proportional
# to its validation balanced_accuracy. This keeps the ensemble structure
# but makes it adaptive to individual model performance.
#
# FIX: Ensemble uses SAME hyperparameters as the individually-trained models
#      (n_estimators=500, max_depth=12, etc.) to avoid inconsistency.
# FIX: Uses balanced_accuracy for weight computation.

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

print("Computing Validation Balanced Accuracy for Adaptive Weights...")
t0 = time.time()

# --- Get validation balanced accuracy for each trained model ---
# Use 3-fold CV on a subsample for speed
VAL_SAMPLE = min(100_000, len(X_train_w))
val_idx = np.random.choice(len(X_train_w), VAL_SAMPLE, replace=False)
X_val_sample = X_train_w.iloc[val_idx]
y_val_sample = y_train.iloc[val_idx]

print("  Evaluating XGBoost...")
xgb_val_scores = cross_val_score(xgb, X_val_sample, y_val_sample, cv=3,
                                  scoring='balanced_accuracy', n_jobs=-1)
xgb_val_acc = xgb_val_scores.mean()
print(f"    XGBoost val balanced acc: {xgb_val_acc:.4f}")

print("  Evaluating LightGBM...")
lgbm_val_scores = cross_val_score(lgbm, X_val_sample, y_val_sample, cv=3,
                                   scoring='balanced_accuracy', n_jobs=-1)
lgbm_val_acc = lgbm_val_scores.mean()
print(f"    LightGBM val balanced acc: {lgbm_val_acc:.4f}")

print("  Evaluating Random Forest...")
rf_val_scores = cross_val_score(rf, X_val_sample, y_val_sample, cv=3,
                                 scoring='balanced_accuracy', n_jobs=-1)
rf_val_acc = rf_val_scores.mean()
print(f"    Random Forest val balanced acc: {rf_val_acc:.4f}")

# --- Compute normalized weights ---
total_acc = xgb_val_acc + lgbm_val_acc + rf_val_acc
w_xgb = xgb_val_acc / total_acc
w_lgbm = lgbm_val_acc / total_acc
w_rf = rf_val_acc / total_acc

print(f"\n  Adaptive Weights (based on balanced accuracy):")
print(f"    XGBoost:       {w_xgb:.4f} (from {xgb_val_acc:.4f})")
print(f"    LightGBM:      {w_lgbm:.4f} (from {lgbm_val_acc:.4f})")
print(f"    Random Forest: {w_rf:.4f} (from {rf_val_acc:.4f})")

# --- Build weighted soft voting ensemble ---
# FIX: Use SAME hyperparameters as individually-trained models
print("\nTraining Adaptive Weighted Soft Voting Ensemble...")
print("(Using same hyperparameters as individual models for consistency)")

hybrid_model = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(
            n_estimators=500,           # Matches cell_13
            objective="multi:softprob", # softprob for soft voting
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            device="cuda",
            random_state=RANDOM_SEED,
            max_depth=12,               # Matches cell_13
            learning_rate=0.05,         # Matches cell_13
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,         # Matches cell_13
            reg_alpha=0.1,              # Matches cell_13
            reg_lambda=1.0,             # Matches cell_13
            verbosity=0
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=500,           # Matches cell_13
            objective="multiclass",
            num_class=num_classes,
            max_depth=12,               # Matches cell_13
            learning_rate=0.05,         # Matches cell_13
            num_leaves=63,              # Matches cell_13
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,       # Matches cell_13
            class_weight='balanced',    # Matches cell_13
            random_state=RANDOM_SEED,
            verbose=-1,
            force_row_wise=True
        )),
        ('rf', RandomForestClassifier(
            n_estimators=500,           # Matches cell_13
            max_depth=None,             # Matches cell_13 (unlimited)
            min_samples_leaf=1,         # Matches cell_13
            n_jobs=-1,
            random_state=RANDOM_SEED,
            class_weight='balanced_subsample'  # Matches cell_13
        ))
    ],
    voting='soft',
    weights=[w_xgb, w_lgbm, w_rf],  # Adaptive weights from balanced accuracy
    n_jobs=-1
)

hybrid_model.fit(X_train_w, y_train)
preds = hybrid_model.predict(X_test_w)

print(f"\nAdaptive Weighted Ensemble completed in {time.time()-t0:.1f}s")
print(f"Weights: XGBoost={w_xgb:.4f}, LightGBM={w_lgbm:.4f}, RF={w_rf:.4f}")
