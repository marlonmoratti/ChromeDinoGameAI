import numpy as np
from dinoAIParallel import KeyClassifier, parse_object_type

INPUT_LAYER_SIZE = 7
THRESHOLD = 0.55

class NeuralNet(KeyClassifier):
    def __init__(self, state, hidden_layer_size=4):
        self.hidden_layer_size = hidden_layer_size
        fc1, fc2 = self._parse_state(state)
        self.fc1 = lambda x: np.dot(fc1, x)
        self.fc2 = lambda x: np.dot(fc2, x)

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
        n_input_weights = INPUT_LAYER_SIZE * self.hidden_layer_size
        input_layer_weights = state[:n_input_weights].reshape(INPUT_LAYER_SIZE, self.hidden_layer_size)
        hidden_layer_weights = state[n_input_weights:]
        return input_layer_weights, hidden_layer_weights

    def _relu(self, x):
        return np.maximum(x, 0)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
