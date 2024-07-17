from source.provided_code.dinoAIParallel import manyPlaysResultsTrain, manyPlaysResultsTest
from source.GeneticAlgorithm import GeneticAlgorithm
from source.NeuralNet import NeuralNet
from source.utils import parse_arguments
import numpy as np

def main():
    config = parse_arguments()

    if config['load_state']:
        print('Loading and evaluating the best state of the algorithm...')
        best_state = np.load('results/best_state.npy')

        res, value = manyPlaysResultsTest(30, best_state)
        print(f'\n{{ mean: {np.mean(res):.2f}, std: {np.std(res):.2f}, score: {value:.2f} }}')
        return
    
    print('Starting a new training session...')
    ga = GeneticAlgorithm(
        fitness_fn=lambda x: np.maximum(manyPlaysResultsTrain(10, x), 0),
        chromosome_length=25,
        cut_length=5,
        selection_type='tournament',
        crossover_rate=0.75,
        mutation_rate=0.25,
        init_individual_fn=lambda x: NeuralNet.init_state(4, x),
        random_state=42
    )

    (best_state, best_value), history = ga.evolve(1000, 12*60*60)
    res, value = manyPlaysResultsTest(30, best_state)
    print(f'\n{{ mean: {np.mean(res):.2f}, std: {np.std(res):.2f}, score: {value:.2f} }}')

    np.save('results/best_state.npy', best_state)
    np.save('results/history.npy', history)

if __name__ == '__main__':
    main()
