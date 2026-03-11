# ======================================
# FEATURE SELECTION — STAGE 5
# Hybrid Feature Ranking Aggregation
# ======================================
# CONSERVATIVE approach: Use UNION-based selection instead of
# aggressive multiplicative scoring. A feature is kept if it was
# selected by ANY 2 of the 4 methods, ensuring minority-class
# features are not dropped by a single method's blind spot.
# FIX: Uses X_train_full.columns to prevent data leakage.

print("Stage 5: Hybrid Feature Ranking Aggregation")
print("=" * 55)
print("[LEAKAGE-FREE] Feature sets derived from training data only")

t0 = time.time()

all_features = list(X_train_full.columns)

# --- Score each feature: +1 per method that supports it ---

# 1) RF Importance: top 80% by importance
rf_top_n = max(15, int(len(all_features) * 0.80))
rf_top_features = set(rf_importance_df.head(rf_top_n)['Feature'].tolist())

# 2) Correlation Filter survivors
corr_survivors = set(features_after_corr)

# 3) RFE-XGBoost selected
rfe_set = rfe_selected_set

# 4) Red Ant selected
ant_set = ant_selected_set

# --- Build scoring table ---
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
        'Methods': ', '.join(methods),
        'N_Methods': len(methods)
    })

agg_df = pd.DataFrame(agg_records).sort_values(
    ['Score', 'RF_Importance'], ascending=[False, False]
).reset_index(drop=True)

# --- Conservative selection: keep features with score >= 2 ---
# This means a feature only needs 2 out of 4 methods to agree.
MIN_FINAL_FEATURES = 15
score_threshold = 2
final_features = agg_df[agg_df['Score'] >= score_threshold]['Feature'].tolist()

# If still too few, take the top features by RF importance
if len(final_features) < MIN_FINAL_FEATURES:
    final_features = agg_df.head(MIN_FINAL_FEATURES)['Feature'].tolist()

print("\nHybrid Aggregation Table:")
print(f"{'Feature':<20}{'Score':<7}{'RF_Importance':<15}{'Methods'}")
print("-" * 65)
for _, row in agg_df.iterrows():
    marker = " << SELECTED" if row['Feature'] in final_features else ""
    print(f"{row['Feature']:<20}{row['Score']:<7}{row['RF_Importance']:<15.6f}{row['Methods']}{marker}")

print(f"\nSelection threshold: score >= {score_threshold} (2 of 4 methods)")
print(f"Final selected features: {len(final_features)}")
print(f"Features: {final_features}")
print(f"Completed in {time.time()-t0:.1f}s")
