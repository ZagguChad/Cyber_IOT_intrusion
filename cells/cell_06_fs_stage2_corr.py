# ======================================
# FEATURE SELECTION — STAGE 2
# Correlation-Based Redundancy Removal
# ======================================
# Remove one feature from each highly-correlated pair (|r| >= 0.95),
# keeping the one with higher RF importance.
# NOTE: Threshold raised from 0.90 to 0.95 to preserve more features.
# A 0.90 threshold was too aggressive and removed features critical
# for detecting minority attack classes.

print("Stage 2: Correlation-Based Redundancy Removal")
print("=" * 55)

CORR_THRESHOLD = 0.95  # Raised from 0.90 to preserve more features

t0 = time.time()

corr_matrix = X_full.corr().abs()

# Upper triangle to avoid duplicate pairs
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop_corr = set()
corr_pairs = []

for col in upper.columns:
    for idx_name in upper.index:
        val = upper.loc[idx_name, col]
        if pd.notna(val) and val >= CORR_THRESHOLD:
            # Drop the feature with LOWER RF importance
            imp_col = rf_importance_map.get(col, 0)
            imp_idx = rf_importance_map.get(idx_name, 0)
            drop_feat = col if imp_col < imp_idx else idx_name
            keep_feat = idx_name if drop_feat == col else col
            to_drop_corr.add(drop_feat)
            corr_pairs.append((idx_name, col, val, keep_feat, drop_feat))

print(f"\nHighly correlated pairs found (|r| >= {CORR_THRESHOLD}): {len(corr_pairs)}")
print(f"\n{'Feature A':<18}{'Feature B':<18}{'Corr':<8}{'Keep':<18}{'Drop':<18}")
print("-" * 80)
for a, b, r, keep, drop in corr_pairs[:25]:
    print(f"{a:<18}{b:<18}{r:.4f}  {keep:<18}{drop:<18}")
if len(corr_pairs) > 25:
    print(f"  ... and {len(corr_pairs)-25} more pairs")

features_after_corr = [f for f in X_full.columns if f not in to_drop_corr]
print(f"\nFeatures removed: {len(to_drop_corr)}  ->  {sorted(to_drop_corr)}")
print(f"Features remaining: {len(features_after_corr)}")
print(f"Completed in {time.time()-t0:.1f}s")
