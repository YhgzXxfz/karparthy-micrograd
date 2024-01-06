import math
import typing as tp


class Value:
    def __init__(self, data, label="", _children: tp.Tuple = (), _op: str = "") -> None:
        self.label = label
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        data = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(data, _children=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - data**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def topo_sort(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    topo_sort(child)
                topo.append(node)

        topo_sort(self)

        self.grad = 1.0
        for n in reversed(topo):
            n._backward()
