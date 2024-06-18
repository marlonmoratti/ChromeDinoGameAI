import numpy as np
import time

class GeneticAlgorithm:
    def __init__(self, fitness_fn,
                 chromosome_length,
                 population_size=100,
                 mutation_rate=0.05,
                 mutation_strength=0.1,
                 random_state=None):

        self._fitness = fitness_fn
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.random = np.random.RandomState(random_state)

    def evolve(self, generations, timelimit):
        population = self._initialize_population()
        fitness_values = self._fitness(population)
        best_individual = max(zip(population, fitness_values), key=lambda x: x[1])

        start_time = time.time()
        for generation in range(generations):
            population, fitness_values, current_best = self._next_generation(population, fitness_values)
            best_individual = max([best_individual, current_best], key=lambda x: x[1])

            print(f'Generation {generation + 1}/{generations}: Fitness = {best_individual[1]}')

            curr_time = time.time()
            elapsed_time = curr_time - start_time
            if elapsed_time > timelimit: break
        
        return best_individual

    def _initialize_population(self):
        return [self.random.randn(self.chromosome_length) for _ in range(self.population_size)]

    def _parent_selection(self, population, fitness_values):
        total_fitness = sum(fitness_values)
        selection_probs = fitness_values / total_fitness

        indices = np.arange(self.population_size)
        parent_indices = self.random.choice(indices, size=2, p=selection_probs)
        parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]

        return parent1, parent2

    def _crossover(self, parent1, parent2):
        mask = np.random.rand(self.chromosome_length) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def _mutate(self, individual):
        mask = np.random.rand(self.chromosome_length) < self.mutation_rate
        individual[mask] += np.random.randn(np.sum(mask)) * self.mutation_strength
        return individual

    def _next_generation(self, population, fitness_values):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = self._parent_selection(population, fitness_values)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)

        fitness_values = self._fitness(new_population)
        current_best = max(zip(new_population, fitness_values), key=lambda x: x[1])
        return new_population, fitness_values, current_best
