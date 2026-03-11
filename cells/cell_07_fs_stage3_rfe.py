# ======================================
# FEATURE SELECTION — STAGE 3
# RFE with XGBoost (GPU Accelerated)
# ======================================
# Recursive Feature Elimination using XGBoost as the estimator.
# Uses features that survived the correlation filter.
# NOTE: Target raised to 75% (from 60%) to keep more features.

from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
import time

print("Stage 3: RFE with XGBoost (GPU)")
print("=" * 55)

t0 = time.time()

# Work on the correlation-filtered feature set
X_rfe = X_full[features_after_corr].copy()

# Subsample for RFE speed
RFE_SAMPLE = min(100_000, len(X_rfe))
idx_rfe = np.random.choice(len(X_rfe), RFE_SAMPLE, replace=False)
X_rfe_sample = X_rfe.iloc[idx_rfe]
y_rfe_sample = y_full.iloc[idx_rfe]

# XGBoost estimator with GPU
xgb_rfe = XGBClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=y_full.nunique(),
    tree_method='hist',
    device='cuda',
    random_state=RANDOM_SEED,
    verbosity=0
)

# Target: select 75% of remaining features (minimum 15)
# Raised from 60% to be less aggressive
n_target = max(15, int(len(features_after_corr) * 0.75))

rfe = RFE(
    estimator=xgb_rfe,
    n_features_to_select=n_target,
    step=1,  # Remove 1 at a time for finer granularity
    verbose=0
)
rfe.fit(X_rfe_sample, y_rfe_sample)

rfe_selected = [f for f, s in zip(features_after_corr, rfe.support_) if s]
rfe_ranking = dict(zip(features_after_corr, rfe.ranking_))

print(f"\nOptimal feature count: {len(rfe_selected)}")
print(f"\nRFE Selected Features:")
print("-" * 40)
for i, f in enumerate(rfe_selected, 1):
    print(f"  {i:>3}. {f}")

print(f"\nRFE Ranking (all features):")
rfe_rank_df = pd.DataFrame({
    'Feature': features_after_corr,
    'RFE_Rank': [rfe_ranking[f] for f in features_after_corr]
}).sort_values('RFE_Rank')
print(rfe_rank_df.to_string(index=False))

rfe_selected_set = set(rfe_selected)
print(f"\nCompleted in {time.time()-t0:.1f}s")
