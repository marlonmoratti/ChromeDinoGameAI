import numpy as np
import time
from tqdm import tqdm

class GeneticAlgorithm:
    def __init__(self, fitness_fn,
                 chromosome_length,
                 population_size=100,
                 mutation_rate=0.05,
                 mutation_strength=0.1,
                 elitism_rate=0.05,
                 random_state=None):

        self._fitness = fitness_fn
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism_rate = elitism_rate
        self.random = np.random.RandomState(random_state)

    def evolve(self, generations, timelimit):
        population = self._initialize_population()
        fitness_values = self._fitness(population)
        best_individuals = self._select_k_best(population, fitness_values,
            k=int(self.elitism_rate*self.population_size))

        start_time = time.time()
        with tqdm(total=generations, desc='[Best Fitness -inf]', ncols=100) as pbar:
            for _ in range(generations):
                population, fitness_values, current_best = self._next_generation(population, fitness_values)
                best_individuals, population, fitness_values = self._elitism(best_individuals, population, fitness_values)

                pbar.set_description(
                    f'[Best Fitness: {best_individuals[0][1]:.2f}'
                    f', Current Fitness: {current_best[1]:.2f}]'
                )
                pbar.update(1)

                curr_time = time.time()
                elapsed_time = curr_time - start_time
                if elapsed_time > timelimit: break
        
        return best_individuals[0]

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

    def _select_k_best(self, population, fitness_values, k):
        return sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)[:k]

    def _elitism(self, best_individuals, population, fitness_values):
        new_population = list(zip(population, fitness_values)) + best_individuals
        new_population.sort(reverse=True, key=lambda x: x[1])
        k = int(self.elitism_rate*self.population_size)

        best_individuals = new_population[:k]
        population, fitness_values = list(zip(*new_population[:-k]))
        return best_individuals, population, fitness_values
