# ======================================
# ADAPTIVE WEIGHTED SOFT VOTING ENSEMBLE
# ======================================
# Instead of equal weights, each model's contribution is proportional
# to its validation accuracy. This keeps the ensemble structure but
# makes it adaptive to individual model performance.
#
# Final Prediction = w1*RF_prob + w2*XGB_prob + w3*LGBM_prob
# where w_i = val_accuracy_i / sum(val_accuracies)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

print("Computing Validation Accuracy for Adaptive Weights...")
t0 = time.time()

# --- Get validation accuracy for each trained model ---
# Use 3-fold CV on a subsample for speed
VAL_SAMPLE = min(100_000, len(X_train_w))
val_idx = np.random.choice(len(X_train_w), VAL_SAMPLE, replace=False)
X_val_sample = X_train_w.iloc[val_idx]
y_val_sample = y_train.iloc[val_idx]

print("  Evaluating XGBoost...")
xgb_val_scores = cross_val_score(xgb, X_val_sample, y_val_sample, cv=3, scoring='accuracy', n_jobs=-1)
xgb_val_acc = xgb_val_scores.mean()
print(f"    XGBoost val accuracy: {xgb_val_acc:.4f}")

print("  Evaluating LightGBM...")
lgbm_val_scores = cross_val_score(lgbm, X_val_sample, y_val_sample, cv=3, scoring='accuracy', n_jobs=-1)
lgbm_val_acc = lgbm_val_scores.mean()
print(f"    LightGBM val accuracy: {lgbm_val_acc:.4f}")

print("  Evaluating Random Forest...")
rf_val_scores = cross_val_score(rf, X_val_sample, y_val_sample, cv=3, scoring='accuracy', n_jobs=-1)
rf_val_acc = rf_val_scores.mean()
print(f"    Random Forest val accuracy: {rf_val_acc:.4f}")

# --- Compute normalized weights ---
total_acc = xgb_val_acc + lgbm_val_acc + rf_val_acc
w_xgb = xgb_val_acc / total_acc
w_lgbm = lgbm_val_acc / total_acc
w_rf = rf_val_acc / total_acc

print(f"\n  Adaptive Weights:")
print(f"    XGBoost:      {w_xgb:.4f} (from {xgb_val_acc:.4f})")
print(f"    LightGBM:     {w_lgbm:.4f} (from {lgbm_val_acc:.4f})")
print(f"    Random Forest: {w_rf:.4f} (from {rf_val_acc:.4f})")

# --- Build weighted soft voting ensemble ---
print("\nTraining Adaptive Weighted Soft Voting Ensemble...")

hybrid_model = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(
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
        )),
        ('lgbm', LGBMClassifier(
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
        )),
        ('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        ))
    ],
    voting='soft',
    weights=[w_xgb, w_lgbm, w_rf],  # Adaptive weights!
    n_jobs=-1
)

hybrid_model.fit(X_train_w, y_train)
preds = hybrid_model.predict(X_test_w)

print(f"\nAdaptive Weighted Ensemble completed in {time.time()-t0:.1f}s")
print(f"Weights: XGBoost={w_xgb:.4f}, LightGBM={w_lgbm:.4f}, RF={w_rf:.4f}")
