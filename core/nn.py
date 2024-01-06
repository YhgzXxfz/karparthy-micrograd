import typing as tp

import numpy as np

from core.engine import Value


class Neuron:
    def __init__(self, nin: int) -> None:
        self.weights = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(np.random.uniform(-1, 1))

    def __call__(self, x) -> tp.Any:
        out = sum(wi * xi for wi, xi in zip(self.weights, x)) + self.bias
        out = out.tanh()
        return out


class Layer:
    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x) -> tp.Any:
        return [neuron(x) for neuron in self.neurons]
