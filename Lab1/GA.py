# Knapsack Problem by Genetic Algorithm

import random

# Set input Parameters
weights = [12, 43, 59, 21, 15]
values = [60, 100, 120, 240, 30]
capacity = 100
num_items = len(weights)

population_size = 50
mutation_rate = 0.05
crossover_rate = 0.7
num_generations = 10

def create_individual():
    return ''.join(random.choice(['0', '1']) for _ in range(num_items))

def create_population():
    return [create_individual() for _ in range(population_size)]

def fitness(individual):
    total_weight = 0
    total_value = 0
    for i in range(num_items):
        if individual[i] == '1':
            total_weight += weights[i]
            total_value += values[i]
    if total_weight > capacity:
        return 0  # Penalize overweight solutions
    return total_value

def selection(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        contenders = random.sample(list(zip(population, fitnesses)), 3)
        winner = max(contenders, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, num_items - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    else:
        return parent1, parent2

def mutate(individual):
    mutated = list(individual)
    for i in range(num_items):
        if random.random() < mutation_rate:
            mutated[i] = '1' if mutated[i] == '0' else '0'
    return ''.join(mutated)

def run_ga():
    population = create_population()
    best_solution = None
    best_fitness = 0

    for gen in range(num_generations):
        fitnesses = [fitness(ind) for ind in population]
        max_fitness = max(fitnesses)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_solution = population[fitnesses.index(max_fitness)]
        selected = selection(population, fitnesses)
        next_generation = []

        for i in range(0, population_size, 2):
            parent1 = selected[i]
            parent2 = selected[i+1]
            offspring1, offspring2 = crossover(parent1, parent2)
            next_generation.append(mutate(offspring1))
            next_generation.append(mutate(offspring2))

        population = next_generation

    print("Best solution:", best_solution)
    print("Selected items:", [i for i, bit in enumerate(best_solution) if bit == '1'])
    print("Total value:", best_fitness)
    total_weight = sum(weights[i] for i, bit in enumerate(best_solution) if bit == '1')
    print("Total weight:", total_weight)

if __name__ == "__main__":
    run_ga()
