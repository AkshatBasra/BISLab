import numpy as np

# ---------- Graph Definition ----------
# Graph: 1—2, 1—3, 2—4, 3—4, 4—5
edges = [(0,1), (0,2), (1,3), (2,3), (3,4)]
n_vertices = 5

# ---------- Parameters ----------
num_wolves = 8          # population size
max_iter = 30           # iterations
alpha_penalty = 10      # penalty for uncovered edges

# ---------- Fitness Function ----------
def fitness(solution):
    """Compute fitness = |C| + penalty * uncovered_edges."""
    cover_size = np.sum(solution)
    uncovered = 0
    for u, v in edges:
        if solution[u] == 0 and solution[v] == 0:
            uncovered += 1
    return cover_size + alpha_penalty * uncovered

# ---------- Initialize Wolves ----------
wolves = np.random.randint(0, 2, (num_wolves, n_vertices))  # binary 0/1
fitness_vals = np.array([fitness(w) for w in wolves])

# Identify alpha, beta, delta wolves (best three)
def update_leaders(wolves, fitness_vals):
    sorted_idx = np.argsort(fitness_vals)
    alpha = wolves[sorted_idx[0]].copy()
    beta  = wolves[sorted_idx[1]].copy()
    delta = wolves[sorted_idx[2]].copy()
    return alpha, beta, delta

alpha, beta, delta = update_leaders(wolves, fitness_vals)

# ---------- Grey Wolf Optimization Loop ----------
for t in range(max_iter):
    a = 2 - 2 * t / max_iter  # linearly decreases from 2 to 0
    
    for i in range(num_wolves):
        X = wolves[i].astype(float)
        
        for j in range(n_vertices):
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha[j] - X[j])
            X1 = alpha[j] - A1 * D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta[j] - X[j])
            X2 = beta[j] - A2 * D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta[j] - X[j])
            X3 = delta[j] - A3 * D_delta

            # New position for dimension j
            X[j] = (X1 + X2 + X3) / 3.0
        
        # ---------- Binarization ----------
        # Use sigmoid transfer + threshold
        X = 1 / (1 + np.exp(-10*(X - 0.5)))  # steeper sigmoid
        X = np.where(np.random.rand(n_vertices) < X, 1, 0)
        
        # Update wolf position and fitness
        wolves[i] = X
        fitness_vals[i] = fitness(X)
    
    # Update alpha, beta, delta
    alpha, beta, delta = update_leaders(wolves, fitness_vals)

    # print(f"Iteration {t+1}: Best fitness = {fitness(alpha):.2f}")

# ---------- Results ----------
print("\nBest vertex cover found:")
print("Binary vector:", alpha)
best_vertices = [i+1 for i in range(n_vertices) if alpha[i] == 1]
print("Vertices in cover:", best_vertices)
print("Fitness:", fitness(alpha))
