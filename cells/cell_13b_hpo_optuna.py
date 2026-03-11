# ======================================
# HYPERPARAMETER OPTIMIZATION (OPTUNA)
# ======================================
# Lightweight Bayesian optimization for XGBoost using Optuna.
# Runs 20 trials on a subsample to find better hyperparameters.
# Retrains XGBoost with best params; LightGBM and RF keep their params.
#
# This addresses the audit concern about manual hyperparameter selection
# and brings our approach closer to the base paper's BO-TPE tuning.

import optuna
from sklearn.model_selection import cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Hyperparameter Optimization with Optuna (Bayesian Search)")
print("=" * 55)

t0 = time.time()

# Subsample for speed (max 200K from SMOTE-inflated training data)
HPO_SAMPLE = min(200_000, len(X_train))
hpo_idx = np.random.choice(len(X_train), HPO_SAMPLE, replace=False)
X_hpo = X_train.iloc[hpo_idx].reset_index(drop=True)
y_hpo = y_train.iloc[hpo_idx].reset_index(drop=True)
num_classes_hpo = y_hpo.nunique()

print(f"Optimization dataset: {len(X_hpo):,} samples")
print(f"Running 20 trials...\n")

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 700, step=100),
        'max_depth': trial.suggest_int('max_depth', 6, 16),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    
    model = XGBClassifier(
        **params,
        objective='multi:softprob',
        num_class=num_classes_hpo,
        tree_method='hist',
        device='cuda',
        random_state=RANDOM_SEED,
        verbosity=0
    )
    
    scores = cross_val_score(model, X_hpo, y_hpo, cv=3,
                             scoring='balanced_accuracy', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=20, show_progress_bar=False)

best_params = study.best_params
best_score = study.best_value

print(f"Best balanced accuracy: {best_score:.4f}")
print(f"\nBest hyperparameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# --- Retrain XGBoost with optimized hyperparameters ---
print(f"\nRetraining XGBoost with Optuna-optimized params...")
t1 = time.time()

xgb = XGBClassifier(
    **best_params,
    objective='multi:softprob',
    num_class=num_classes,
    eval_metric='mlogloss',
    tree_method='hist',
    device='cuda',
    random_state=RANDOM_SEED,
    verbosity=0
)
xgb.fit(X_train, y_train)

print(f"  XGBoost (Optuna) retrained in {time.time()-t1:.1f}s")
print(f"\nTotal HPO time: {time.time()-t0:.1f}s")
print(f"\n[INFO] XGBoost has been retrained with optimized params.")
print(f"       LightGBM and Random Forest retain their original params.")
print(f"       The ensemble in cell_16 will use the new XGBoost.")
