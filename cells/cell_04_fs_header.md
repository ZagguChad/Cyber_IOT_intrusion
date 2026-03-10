# ═══ FEATURE SELECTION MODULE ═══
> **New Stage** — Replaces arbitrary column dropping with a multi-strategy,
> scientifically grounded feature selection pipeline.
>
> | Stage | Method | Purpose |
> |-------|--------|---------|
> | 1 | Random Forest Importance | Preliminary importance ranking |
> | 2 | Correlation Filtering | Redundancy removal (threshold 0.90) |
> | 3 | RFE with XGBoost (GPU) | Recursive subset optimisation |
> | 4 | Red Ant Algorithm | Swarm-intelligence subset search |
> | 5 | Aggregation | Voting-based final feature set |
