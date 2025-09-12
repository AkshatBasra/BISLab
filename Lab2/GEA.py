import random

grid = [
    [0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
end = (4, 4)

moves = ['U', 'D', 'L', 'R']
population_size = 20
chromosome_length = 10
generations = 1000
mutation_rate = 0.3

def move_position(pos, move):
    x, y = pos
    if move == 'U':
        x -= 1
    elif move == 'D':
        x += 1
    elif move == 'L':
        y -= 1
    elif move == 'R':
        y += 1
    return (x, y)

def is_valid(pos):
    x, y = pos
    if 0 <= x < 5 and 0 <= y < 5 and grid[x][y] == 0:
        return True
    return False

def fitness(chromosome):
    pos = start
    for move in chromosome:
        next_pos = move_position(pos, move)
        if not is_valid(next_pos):
            break
        pos = next_pos
        if pos == end:
            break
    dist = abs(pos[0] - end[0]) + abs(pos[1] - end[1])
    return -dist

def create_chromosome():
    return [random.choice(moves) for _ in range(chromosome_length)]

def mutate(chromosome):
    new_chromosome = chromosome[:]
    for i in range(len(new_chromosome)):
        if random.random() < mutation_rate:
            new_chromosome[i] = random.choice(moves)
    return new_chromosome

def crossover(parent1, parent2):
    point = random.randint(1, chromosome_length - 1)
    return parent1[:point] + parent2[point:]

population = [create_chromosome() for _ in range(population_size)]

for gen in range(generations):
    population = sorted(population, key=fitness, reverse=True)
    best_fitness = fitness(population[0])
    
    if best_fitness == 0:
        break

    new_population = population[:2]

    while len(new_population) < population_size:
        parents = random.sample(population[:5], 2)
        child = crossover(parents[0], parents[1])
        child = mutate(child)
        new_population.append(child)

    population = new_population

best = population[0]
pos = start
path = [pos]
used_moves = []

for move in best:
    next_pos = move_position(pos, move)
    if not is_valid(next_pos):
        break
    pos = next_pos
    path.append(pos)
    used_moves.append(move)
    if pos == end:
        break

print("\nBest path found:", path)
print("Moves used to reach goal:", used_moves)
print("Path Length:", len(used_moves))
