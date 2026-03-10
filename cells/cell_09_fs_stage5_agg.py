# ======================================
# FEATURE SELECTION — STAGE 5
# Hybrid Feature Ranking Aggregation
# ======================================
# Combine results from RF importance, correlation filter, RFE-XGBoost,
# and Red Ant selection using multiplicative hybrid scoring.
#
# Hybrid Score = RF_Importance_Weight × RFE_Selection_Weight × RedAnt_Frequency

print("Stage 5: Hybrid Feature Ranking Aggregation")
print("=" * 55)

t0 = time.time()

all_features = list(X_full.columns)

# --- Compute per-method weights for each feature ---

# 1) RF Importance Weight (normalized to [0, 1])
max_rf_imp = rf_importance_df['Importance'].max()
rf_weight_map = {}
for _, row in rf_importance_df.iterrows():
    rf_weight_map[row['Feature']] = row['Importance'] / max_rf_imp if max_rf_imp > 0 else 0

# 2) Correlation Filter Weight: 1.0 if survived, 0.1 if removed
corr_survivors = set(features_after_corr)

# 3) RFE Selection Weight: 1.0 if selected, scaled by (1/rank) if not
rfe_weight_map = {}
max_rfe_rank = max(rfe_ranking.values()) if rfe_ranking else 1
for feat in features_after_corr:
    if feat in rfe_selected_set:
        rfe_weight_map[feat] = 1.0
    else:
        rfe_weight_map[feat] = 1.0 / rfe_ranking.get(feat, max_rfe_rank)

# 4) Red Ant Selection Frequency (normalized pheromone as proxy)
#    Features in the best ant subset get 1.0, others get scaled value
ant_weight_map = {}
max_phero = pheromone.max() if pheromone.max() > 0 else 1.0
for i, feat in enumerate(ant_features):
    ant_weight_map[feat] = pheromone[i] / max_phero

# --- Build hybrid aggregation table ---
agg_records = []
for feat in all_features:
    rf_w = rf_weight_map.get(feat, 0.001)
    corr_w = 1.0 if feat in corr_survivors else 0.1
    rfe_w = rfe_weight_map.get(feat, 0.1)
    ant_w = ant_weight_map.get(feat, 0.1)

    # Multiplicative hybrid score
    hybrid_score = rf_w * corr_w * rfe_w * ant_w

    # Track which methods selected this feature
    methods = []
    if rf_w >= 0.5:
        methods.append("RF")
    if feat in corr_survivors:
        methods.append("Corr")
    if feat in rfe_selected_set:
        methods.append("RFE")
    if feat in ant_selected_set:
        methods.append("Ant")

    agg_records.append({
        'Feature': feat,
        'RF_Weight': rf_w,
        'Corr_Weight': corr_w,
        'RFE_Weight': rfe_w,
        'Ant_Weight': ant_w,
        'Hybrid_Score': hybrid_score,
        'Methods': ', '.join(methods),
        'N_Methods': len(methods)
    })

agg_df = pd.DataFrame(agg_records).sort_values(
    'Hybrid_Score', ascending=False
).reset_index(drop=True)

# --- Select features using hybrid score threshold ---
# Use the "elbow" method: select features with score > mean score
# But ensure at least 10 features
mean_score = agg_df['Hybrid_Score'].mean()
median_score = agg_df['Hybrid_Score'].median()

# Threshold: features above mean score AND selected by >= 2 methods
threshold = max(mean_score, median_score)
final_features = agg_df[
    (agg_df['Hybrid_Score'] > threshold) & (agg_df['N_Methods'] >= 2)
]['Feature'].tolist()

# Fallback: if too few, take top features by hybrid score
MIN_FINAL_FEATURES = 10
if len(final_features) < MIN_FINAL_FEATURES:
    final_features = agg_df.head(MIN_FINAL_FEATURES)['Feature'].tolist()

print("\nHybrid Aggregation Table (all features):")
print(f"{'Feature':<18}{'RF_W':<8}{'Corr':<6}{'RFE_W':<8}{'Ant_W':<8}{'Hybrid':<10}{'Methods'}")
print("-" * 80)
for _, row in agg_df.iterrows():
    marker = " << SELECTED" if row['Feature'] in final_features else ""
    print(f"{row['Feature']:<18}{row['RF_Weight']:<8.4f}{row['Corr_Weight']:<6.1f}"
          f"{row['RFE_Weight']:<8.4f}{row['Ant_Weight']:<8.4f}"
          f"{row['Hybrid_Score']:<10.6f}{row['Methods']}{marker}")

print(f"\nHybrid score threshold: {threshold:.6f}")
print(f"Final selected features: {len(final_features)}")
print(f"Features: {final_features}")
print(f"\nCompleted in {time.time()-t0:.1f}s")
