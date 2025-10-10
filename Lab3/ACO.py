import numpy as np

num_cities = 5
num_ants = 10
num_iterations = 100
alpha = 1.0       
beta = 5.0        
rho = 0.5         
Q = 100

cities = np.array([
    [0, 2, 9, 10, 7],
    [1, 0, 6, 4, 3],
    [15, 7, 0, 8, 3],
    [6, 3, 12, 0, 11],
    [9, 7, 5, 6, 0]
])

num_cities = cities.shape[0]


pheromone = np.ones((num_cities, num_cities))


heuristic = 1 / (cities + 1e-10)

best_distance = float('inf')
best_tour = None

for iteration in range(num_iterations):
    all_tours = []
    all_distances = []
    
    for ant in range(num_ants):
        
        start = np.random.randint(num_cities)
        visited = [start]
        tour_distance = 0
        
        while len(visited) < num_cities:
            current = visited[-1]
            allowed = [i for i in range(num_cities) if i not in visited]
            
            prob = []
            for city in allowed:
                prob.append((pheromone[current][city] ** alpha) * (heuristic[current][city] ** beta))
            prob = np.array(prob)
            prob = prob / prob.sum()
            
            next_city = np.random.choice(allowed, p=prob)
            tour_distance += cities[current][next_city]
            visited.append(next_city)
        
        tour_distance += cities[visited[-1]][visited[0]]
        all_tours.append(visited)
        all_distances.append(tour_distance)
        
        if tour_distance < best_distance:
            best_distance = tour_distance
            best_tour = visited.copy()
    
    pheromone = (1 - rho) * pheromone
    for idx, tour in enumerate(all_tours):
        for i in range(num_cities):
            from_city = tour[i]
            to_city = tour[(i + 1) % num_cities]
            pheromone[from_city][to_city] += Q / all_distances[idx]

best_tour = list(map(int, best_tour))
print("Best tour:", best_tour)
print("Best distance:", best_distance)
