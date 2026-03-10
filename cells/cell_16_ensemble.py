# ======================================
# ENSEMBLE MODEL (SOFT VOTING)
# ======================================
# Combine predictions using soft voting (probability averaging).

from sklearn.ensemble import VotingClassifier

print("Training Hybrid Soft Voting Ensemble (GPU)...")
t0 = time.time()

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
    n_jobs=-1
)

hybrid_model.fit(X_train_w, y_train)
preds = hybrid_model.predict(X_test_w)

print(f"Ensemble training completed in {time.time()-t0:.1f}s")
