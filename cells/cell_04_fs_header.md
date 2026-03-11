# ═══ FEATURE SELECTION MODULE ═══
> **Leakage-Free Design** — All feature selection operates on TRAINING data only.
> Train-test split is performed in Cell 03b before any feature selection.
>
> | Stage | Method | Purpose |
> |-------|--------|---------|
> | 1 | Random Forest Importance | Preliminary importance ranking |
> | 2 | Correlation Filtering | Redundancy removal (threshold 0.95) |
> | 3 | RFE with XGBoost (GPU) | Recursive subset optimisation |
> | 4 | Red Ant Algorithm | Swarm-intelligence subset search |
> | 5 | Aggregation | Voting-based final feature set |
