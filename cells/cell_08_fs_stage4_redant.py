# ======================================
# FEATURE SELECTION — STAGE 4
# Red Ant Feature Selection Algorithm
# ======================================
# Swarm intelligence optimisation inspired by ant colony search.
# Each ant selects a random feature subset, evaluates fitness,
# and the colony converges on the best subset over iterations.
#
# Fitness = Accuracy - lambda * (n_selected / n_total)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time

print("Stage 4: Red Ant Feature Selection Algorithm")
print("=" * 55)

t0 = time.time()

# Parameters
N_ANTS = 20                # number of ants per iteration
N_ITERATIONS = 15          # total iterations
LAMBDA_PENALTY = 0.05      # simplicity penalty
MIN_FEATURES = 5           # minimum features an ant can select
EVAPORATION = 0.3          # pheromone evaporation rate
ALPHA = 1.0                # pheromone influence
BETA = 2.0                 # heuristic (RF importance) influence

# Use features that passed correlation filter
ant_features = features_after_corr.copy()
n_features = len(ant_features)

# Subsample for speed
ANT_SAMPLE = min(50_000, len(X_full))
idx_ant = np.random.choice(len(X_full), ANT_SAMPLE, replace=False)
X_ant = X_full[ant_features].iloc[idx_ant]
y_ant = y_full.iloc[idx_ant]

# Initialize pheromone trails (uniform)
pheromone = np.ones(n_features)

# Heuristic info: RF importance (normalized)
heuristic = np.array([rf_importance_map.get(f, 0.001) for f in ant_features])
heuristic = heuristic / heuristic.sum()

# Quick classifier for fitness evaluation
def evaluate_subset(feature_indices, X_data, y_data):
    """Evaluate a feature subset using RF accuracy (3-fold CV)."""
    if len(feature_indices) == 0:
        return 0.0
    X_sub = X_data.iloc[:, feature_indices]
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    scores = cross_val_score(clf, X_sub, y_data, cv=3, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Track best solution
best_fitness = -np.inf
best_subset_idx = list(range(n_features))
best_accuracy = 0.0

print(f"Configuration: {N_ANTS} ants x {N_ITERATIONS} iterations")
print(f"Feature pool: {n_features} features")
print(f"Lambda penalty: {LAMBDA_PENALTY}")
print()

for iteration in range(N_ITERATIONS):
    iter_best_fitness = -np.inf
    iter_best_subset = None
    iter_best_acc = 0.0

    for ant in range(N_ANTS):
        # Probabilistic feature selection based on pheromone + heuristic
        prob = (pheromone ** ALPHA) * (heuristic ** BETA)
        prob = prob / prob.sum()

        # Each ant selects a random number of features (at least MIN_FEATURES)
        n_select = np.random.randint(MIN_FEATURES, max(MIN_FEATURES+1, n_features))
        selected_idx = np.random.choice(n_features, size=min(n_select, n_features),
                                         replace=False, p=prob)
        selected_idx = sorted(selected_idx)

        # Evaluate fitness
        accuracy = evaluate_subset(selected_idx, X_ant, y_ant)
        n_sel = len(selected_idx)
        fitness = accuracy - LAMBDA_PENALTY * (n_sel / n_features)

        if fitness > iter_best_fitness:
            iter_best_fitness = fitness
            iter_best_subset = selected_idx
            iter_best_acc = accuracy

    # Update pheromone
    pheromone *= (1 - EVAPORATION)  # evaporation
    if iter_best_subset is not None:
        pheromone[iter_best_subset] += iter_best_fitness  # deposit

    # Clamp pheromone
    pheromone = np.clip(pheromone, 0.1, 10.0)

    # Update global best
    if iter_best_fitness > best_fitness:
        best_fitness = iter_best_fitness
        best_subset_idx = iter_best_subset
        best_accuracy = iter_best_acc

    print(f"  Iter {iteration+1:>2}/{N_ITERATIONS}  |  "
          f"Best fitness: {iter_best_fitness:.4f}  |  "
          f"Accuracy: {iter_best_acc:.4f}  |  "
          f"Features: {len(iter_best_subset) if iter_best_subset is not None else 0}")

# Extract best feature names
ant_selected_features = [ant_features[i] for i in best_subset_idx]
ant_selected_set = set(ant_selected_features)

print(f"\n{'='*55}")
print(f"Red Ant Best Solution:")
print(f"  Fitness:        {best_fitness:.4f}")
print(f"  Accuracy:       {best_accuracy:.4f}")
print(f"  Features:       {len(ant_selected_features)}")
print(f"  Selected:       {ant_selected_features}")
print(f"Completed in {time.time()-t0:.1f}s")
