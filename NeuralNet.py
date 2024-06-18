import numpy as np

INPUT_LAYER_SIZE = 7
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

    def forward(self, x):
        x = self._relu(self.fc1(x))
        x = self._sigmoid(self.fc2(x))
        return x

    def keySelector(self, distance, obHeight, speed, obType, nextObDistance, nextObHeight, nextObType):
        x = np.array([speed, distance, obHeight, parse_object_type(obType),
            nextObDistance, nextObHeight, parse_object_type(nextObType)])
        x = self.forward(x)
        return 'K_DOWN' if x < THRESHOLD else 'K_UP'

    def _parse_state(self, state):
        input_layer_size = INPUT_LAYER_SIZE + 1     # adding bias term
        assert len(state) == ((input_layer_size  * self.hidden_layer_size) + self.hidden_layer_size + 1)

        n_input_weights = input_layer_size * self.hidden_layer_size
        input_layer_weights = state[:n_input_weights].reshape(input_layer_size, self.hidden_layer_size)
        hidden_layer_weights = state[n_input_weights:]
        return input_layer_weights, hidden_layer_weights

    def _relu(self, x):
        return np.maximum(x, 0)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
