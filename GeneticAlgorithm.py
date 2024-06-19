import itertools as it
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
        self.elitism_size = int(np.ceil(elitism_rate * population_size))
        self.random = np.random.RandomState(random_state)

    def evolve(self, generations, timelimit):
        population, fitness_values = self._initialize_population()
        best_individuals = self._select_k_best(population, fitness_values)

        start_time = time.time()
        with tqdm(total=generations, desc='[Best Fitness -inf]', ncols=100) as pbar:
            for _ in range(generations):
                population, fitness_values = self._next_generation(population, fitness_values)
                curr_fitness = fitness_values[0]

                best_individuals, population, fitness_values = self._elitism(best_individuals, population, fitness_values)

                pbar.set_description(
                    f'[Best Fitness: {best_individuals[0][1]:.2f}'
                    f', Current Fitness: {curr_fitness:.2f}]'
                )
                pbar.update(1)

                curr_time = time.time()
                elapsed_time = curr_time - start_time
                if elapsed_time > timelimit: break
        
        return best_individuals[0]

    def _initialize_population(self):
        population = self.random.randn(self.population_size, self.chromosome_length)
        fitness_values = self._fitness(population)
        indices = np.argsort(-fitness_values)

        return population[indices], fitness_values[indices]

    def _parent_selection(self, population, fitness_values):
        total_fitness = sum(fitness_values)
        selection_probs = fitness_values / total_fitness

        parent_indices = self.random.choice(self.population_size, size=2, p=selection_probs)
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
        new_population = np.asarray(new_population)

        fitness_values = self._fitness(new_population)
        indices = np.argsort(-fitness_values)
        return new_population[indices], fitness_values[indices]

    def _select_k_best(self, population, fitness_values):
        return list(it.islice(zip(population, fitness_values), self.elitism_size))

    def _elitism(self, best_individuals, population, fitness_values):
        best_population, best_fitness_values = (np.asarray(item) for item in zip(*best_individuals))
        new_population = np.concatenate((best_population, population))
        new_fitness_values = np.concatenate((best_fitness_values, fitness_values))

        indices = np.argsort(-new_fitness_values, kind='mergesort')
        new_population = new_population[indices]
        new_fitness_values = new_fitness_values[indices]

        best_individuals = self._select_k_best(new_population, new_fitness_values)
        population = new_population[:len(population)]
        fitness_values = new_fitness_values[:len(fitness_values)]

        return best_individuals, population, fitness_values
