import numpy as np
from scipy.stats import levy

weights = np.array([12, 43, 59, 21, 15])
values = np.array([60, 100, 120, 240, 30])
capacity = 100
n_items = len(values)

def fitness(solution):
    total_value = np.sum(values * solution)
    total_weight = np.sum(weights * solution)
    if total_weight > capacity:
        return 0
    return total_value

n_nests = 20
p_a = 0.25
max_iter = 200
alpha = 0.1

nests = np.random.randint(0, 2, size=(n_nests, n_items))
fitness_values = np.array([fitness(n) for n in nests])

def levy_flight(size):
    return levy.rvs(size=size)

def simple_bounds(sol):
    sol = np.clip(sol, 0, 1)
    return np.where(sol > 0.5, 1, 0)

for t in range(max_iter):
    for i in range(n_nests):
        step = alpha * levy_flight(n_items)
        new_sol = nests[i] + step * (nests[i] - nests[np.random.randint(n_nests)])
        new_sol = simple_bounds(new_sol)
        
        f_new = fitness(new_sol)
        if f_new > fitness_values[i]:
            nests[i] = new_sol
            fitness_values[i] = f_new

    worst_n = int(p_a * n_nests)
    worst_indices = np.argsort(fitness_values)[:worst_n]
    nests[worst_indices] = np.random.randint(0, 2, size=(worst_n, n_items))
    fitness_values[worst_indices] = [fitness(n) for n in nests[worst_indices]]

    best_index = np.argmax(fitness_values)
    best_solution = nests[best_index]
    best_fitness = fitness_values[best_index]

print("\nFinal Best Solution:")
print("Items selected:", best_solution)
print("Total value:", fitness(best_solution))
print("Total weight:", np.sum(weights * best_solution))
