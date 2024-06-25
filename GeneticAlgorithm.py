import itertools as it
import numpy as np
import time
from tqdm import tqdm

class GeneticAlgorithm:
    def __init__(self, fitness_fn,
                 chromosome_length,
                 cut_length,
                 population_size=100,
                 selection_type='tournament',
                 tournament_percent=0.05,
                 crossover_rate=0.9,
                 mutation_rate=0.1,
                 mutation_strength=1,
                 elitism_percent=0.2,
                 init_individual_fn=None,
                 random_state=None):

        self._fitness = fitness_fn
        self.chromosome_length = chromosome_length
        self.cut_length = cut_length
        self.population_size = population_size
        self.selection_type = selection_type
        self.tournament_size = int(np.ceil(tournament_percent * population_size))
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism_size = int(np.ceil(elitism_percent * population_size))
        self._init_individual = init_individual_fn
        self.random = np.random.RandomState(random_state)

    def evolve(self, generations, timelimit):
        population, fitness_values = self._initialize_population()
        best_individuals = self._select_k_best(population, fitness_values)

        start_time = time.time()
        with tqdm(total=generations) as pbar:
            for _ in range(generations):
                population, fitness_values = self._next_generation(population, fitness_values)
                curr_fitness = fitness_values[0]

                best_individuals, population, fitness_values = self._elitism(best_individuals, population, fitness_values)

                pbar.set_description(
                    f'[ Training ][ Best Fitness: {best_individuals[0][1]:.2f}'
                    f', Current Fitness: {curr_fitness:.2f} ]'
                )
                pbar.update(1)

                curr_time = time.time()
                elapsed_time = curr_time - start_time
                if elapsed_time > timelimit:
                    print('[ TLE ] The training has been interrupted.')
                    break
        
        return best_individuals[0]

    def _initialize_population(self):
        population = None

        if self._init_individual:
            population = np.asarray([
                self._init_individual(self.random.randint(self.population_size*1000))
                for _ in range(self.population_size)
            ])
        else:
            population = self.random.randn(self.population_size, self.chromosome_length)

        fitness_values = self._fitness(population)
        indices = np.argsort(-fitness_values)
        return population[indices], fitness_values[indices]

    def _parent_selection(self, population, fitness_values):
        selection = {
            'roulette': self._roulette_wheel_selection,
            'tournament': self._tournament_selection
        }

        return selection[self.selection_type](population, fitness_values)
    
    def _roulette_wheel_selection(self, population, fitness_values):
        total_fitness = sum(fitness_values)
        selection_probs = fitness_values / total_fitness

        parent_indices = self.random.choice(self.population_size, size=2, replace=False, p=selection_probs)
        parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]

        return parent1, parent2
    
    def _tournament_selection(self, population, fitness_values):
        get_tournament_indices = lambda: self.random.choice(self.population_size, size=self.tournament_size, replace=False)

        tournament_indices = get_tournament_indices()
        parent1_idx = min(tournament_indices)

        tournament_indices = get_tournament_indices()
        parent2_idx = min(tournament_indices)

        while parent1_idx == parent2_idx:
            tournament_indices = get_tournament_indices()
            parent2_idx = min(tournament_indices)

        parent1, parent2 = population[parent1_idx], population[parent2_idx]
        return parent1, parent2

    def _crossover(self, parent1, parent2):
        mask = self._get_cut_mask()
        child = np.where(mask, parent1, parent2)
        return child

    def _mutate(self, individual):
        mask = self._get_cut_mask()
        individual[mask] += self.random.randn(np.sum(mask)) * self.mutation_strength
        return individual
    
    def _get_cut_mask(self):
        num_batches = int(np.ceil(self.chromosome_length / self.cut_length))
        batch_values = self.random.choice([True, False], size=num_batches)
        mask = np.repeat(batch_values, self.cut_length)
        return mask[:self.chromosome_length]

    def _next_generation(self, population, fitness_values):
        new_population = set()

        for _ in range(self.population_size):
            parent1, parent2 = self._parent_selection(population, fitness_values)

            while True:
                child = [parent1, parent2][self.random.randint(2)]

                if self.random.rand() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)

                if self.random.rand() < self.mutation_rate:
                    child = self._mutate(child)
                
                if tuple(child) not in new_population: break
            new_population.add(tuple(child))

        new_population = np.asarray(list(new_population))
        fitness_values = self._fitness(new_population)
        indices = np.argsort(-fitness_values)
        return new_population[indices], fitness_values[indices]

    def _elitism(self, best_individuals, population, fitness_values):
        best_population, best_fitness_values = zip(*best_individuals)
        new_population = np.concatenate((best_population, population))
        new_fitness_values = np.concatenate((best_fitness_values, fitness_values))

        indices = np.argsort(-new_fitness_values, kind='mergesort')
        new_population = new_population[indices]
        new_fitness_values = new_fitness_values[indices]

        best_individuals = self._select_k_best(new_population, new_fitness_values)
        population = new_population[:self.population_size]
        fitness_values = new_fitness_values[:self.population_size]

        return best_individuals, population, fitness_values
    
    def _select_k_best(self, population, fitness_values):
        return list(it.islice(zip(population, fitness_values), self.elitism_size))
