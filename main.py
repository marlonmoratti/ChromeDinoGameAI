from src.dinoAIParallel import manyPlaysResultsTrain, manyPlaysResultsTest
from src.GeneticAlgorithm import GeneticAlgorithm
from src.NeuralNet import NeuralNet
from src.utils import parse_arguments
import numpy as np

def main():
    config = parse_arguments()

    if config['load_state']:
        best_state = np.load('results/best_state.npy')

        res, value = manyPlaysResultsTest(30, best_state)
        npRes = np.asarray(res)
        print(res, npRes.mean(), npRes.std(), value)
        return

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
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)

    np.save('results/best_state.npy', best_state)
    np.save('results/history.npy', history)

if __name__ == '__main__':
    main()
