# ======================================
# FEATURE SELECTION — STAGE 5
# Feature Aggregation Strategy
# ======================================
# Combine outputs from all selection methods using a voting/scoring
# approach. Features earn points for being selected by each method.

print("Stage 5: Feature Aggregation Strategy")
print("=" * 55)

t0 = time.time()

all_features = list(X_full.columns)

# Score each feature: +1 point per method that selects it
# Method 1: RF Importance - top 75% of features by importance
rf_top_n = max(10, int(len(all_features) * 0.75))
rf_top_features = set(rf_importance_df.head(rf_top_n)['Feature'].tolist())

# Method 2: Correlation Filter survivors
corr_survivors = set(features_after_corr)

# Method 3: RFE-XGBoost selected
rfe_set = rfe_selected_set

# Method 4: Red Ant selected
ant_set = ant_selected_set

# Build aggregation table
agg_records = []
for feat in all_features:
    score = 0
    methods = []
    if feat in rf_top_features:
        score += 1
        methods.append("RF")
    if feat in corr_survivors:
        score += 1
        methods.append("Corr")
    if feat in rfe_set:
        score += 1
        methods.append("RFE")
    if feat in ant_set:
        score += 1
        methods.append("Ant")
    agg_records.append({
        'Feature': feat,
        'Score': score,
        'RF_Importance': rf_importance_map.get(feat, 0),
        'Methods': ', '.join(methods)
    })

agg_df = pd.DataFrame(agg_records).sort_values(
    ['Score', 'RF_Importance'], ascending=[False, False]
).reset_index(drop=True)

# Select features with score >= 3 (selected by at least 3 out of 4 methods)
# If too few, lower threshold to 2
MIN_FINAL_FEATURES = 10
score_threshold = 3
final_features = agg_df[agg_df['Score'] >= score_threshold]['Feature'].tolist()

if len(final_features) < MIN_FINAL_FEATURES:
    score_threshold = 2
    final_features = agg_df[agg_df['Score'] >= score_threshold]['Feature'].tolist()

print("\nAggregation Table (all features):")
print(f"{'Feature':<20}{'Score':<7}{'RF_Importance':<15}{'Methods'}")
print("-" * 65)
for _, row in agg_df.iterrows():
    marker = " [SELECTED]" if row['Feature'] in final_features else ""
    print(f"{row['Feature']:<20}{row['Score']:<7}{row['RF_Importance']:<15.6f}{row['Methods']}{marker}")

print(f"\nSelection threshold: score >= {score_threshold}")
print(f"Final selected features: {len(final_features)}")
print(f"Features: {final_features}")
print(f"\nCompleted in {time.time()-t0:.1f}s")
