class Value:
    def __init__(self, data) -> None:
        self.data = data

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data)
