import numpy as np
from scipy.special import expit

INPUT_LAYER_SIZE = 4
OUTPUT_LAYER_SIZE = 1
THRESHOLD = 0.55

def parse_object_type(object_type):
    object_types = {
        'SmallCactus': 0,
        'LargeCactus': 1,
        'Bird': 2
    }

    object_type_name = type(object_type).__name__
    return object_types.get(object_type_name, -1)

class NeuralNet:
    def __init__(self, state, hidden_layer_size=4):
        self.hidden_layer_size = hidden_layer_size
        fc1, fc2 = self._parse_state(state)
        self.fc1 = lambda x: np.dot(np.concatenate(([1], x)), fc1)
        self.fc2 = lambda x: np.dot(np.concatenate(([1], x)), fc2)

    @staticmethod
    def init_state(hidden_layer_size, random_state=None):
        random = np.random.RandomState(random_state) if isinstance(random_state, int) else random_state
        
        # He initialization
        fc1 = (random.randn(INPUT_LAYER_SIZE, hidden_layer_size)
            * np.sqrt(2.0 / INPUT_LAYER_SIZE))
        fc1 = np.concatenate((np.zeros((1, hidden_layer_size)), fc1), axis=0)

        # Xavier initialization
        fc2 = (random.randn(hidden_layer_size, OUTPUT_LAYER_SIZE)
            * np.sqrt(2.0 / (hidden_layer_size + OUTPUT_LAYER_SIZE)))
        fc2 = np.concatenate((np.zeros((1, OUTPUT_LAYER_SIZE)), fc2), axis=0)

        state = np.concatenate((fc1.T.flatten(), fc2.T.flatten()))
        return state

    def forward(self, x):
        x = self._relu(self.fc1(x))
        x = self._sigmoid(self.fc2(x))
        return x

    def keySelector(self, distance, obHeight, speed, obType, nextObDistance, nextObHeight, nextObType):
        x = np.array([speed, distance, obHeight, parse_object_type(obType)])
        x = self.forward(x)
        return 'K_DOWN' if x < THRESHOLD else 'K_UP'

    def _parse_state(self, state):
        input_layer_size = INPUT_LAYER_SIZE + 1     # adding bias term
        assert len(state) == ((input_layer_size  * self.hidden_layer_size) + self.hidden_layer_size + 1)

        n_input_weights = input_layer_size * self.hidden_layer_size
        input_layer_weights = state[:n_input_weights].reshape(self.hidden_layer_size, input_layer_size).T
        hidden_layer_weights = state[n_input_weights:]
        return input_layer_weights, hidden_layer_weights

    def _relu(self, x):
        return np.maximum(x, 0)

    def _sigmoid(self, x):
        return expit(x)
