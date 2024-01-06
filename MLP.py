from typing import Any

import numpy as np

from Values import Value


class Neuron:
    def __init__(self, nin) -> None:
        self.weights = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(np.random.uniform(-1, 1))

    def __call__(self, x) -> Any:
        out = sum(wi * xi for wi, xi in zip(self.weights, x)) + self.bias
        out = out.tanh()
        return out
