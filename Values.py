import typing as tp


class Value:
    def __init__(self, data, label="", _children: tp.Tuple = (), _op: str = "") -> None:
        self.label = label
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, _children=(self, other), _op="+")

    def __mul__(self, other):
        return Value(self.data * other.data, _children=(self, other), _op="*")
