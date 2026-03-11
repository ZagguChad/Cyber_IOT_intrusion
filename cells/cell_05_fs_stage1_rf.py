# ======================================
# FEATURE SELECTION — STAGE 1
# Tree-Based Feature Importance Ranking
# ======================================
# Train a Random Forest on TRAINING data only and rank features
# by their Gini importance scores.
# FIX: Uses X_train_full instead of X_full to prevent data leakage.

from sklearn.ensemble import RandomForestClassifier
import time

print("Stage 1: Random Forest Feature Importance Ranking")
print("=" * 55)
print("[LEAKAGE-FREE] Using training data only")

t0 = time.time()

# Use a stratified subsample of TRAINING data for speed
SAMPLE_SIZE = min(200_000, len(X_train_full))
idx = np.random.choice(len(X_train_full), SAMPLE_SIZE, replace=False)
X_sample = X_train_full.iloc[idx]
y_sample = y_train.iloc[idx]

rf_selector = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    class_weight='balanced'
)
rf_selector.fit(X_sample, y_sample)

# Build importance table
importances = rf_selector.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': X_train_full.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False).reset_index(drop=True)
rf_importance_df['Rank'] = rf_importance_df.index + 1

print(f"\nCompleted in {time.time()-t0:.1f}s")
print(f"\n{'Rank':<6}{'Feature':<20}{'Importance':<12}")
print("-" * 38)
for _, row in rf_importance_df.iterrows():
    print(f"{int(row['Rank']):<6}{row['Feature']:<20}{row['Importance']:.6f}")

# Store feature-to-importance mapping for later stages
rf_importance_map = dict(zip(rf_importance_df['Feature'], rf_importance_df['Importance']))
rf_selected_features = set(rf_importance_df['Feature'].tolist())  # all features initially
print(f"\nTotal features ranked: {len(rf_importance_df)}")
