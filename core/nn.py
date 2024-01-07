import typing as tp
from typing import Any

import numpy as np

from core.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters():
        return []


class Neuron(Module):
    def __init__(self, nin: int) -> None:
        self.weights = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(np.random.uniform(-1, 1))

    def __call__(self, x) -> tp.Any:
        out = sum(wi * xi for wi, xi in zip(self.weights, x)) + self.bias
        out = out.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]


class Layer(Module):
    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x) -> tp.Any:
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin: int, nouts: tp.List[int]) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
