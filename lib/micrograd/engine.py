"""
MindMirror - Micrograd: A tiny Autograd engine

This is the foundation of our neural network library. We implement automatic
differentiation from scratch to understand the mathematical foundations
of backpropagation.

Inspired by Andrej Karpathy's micrograd, but implemented step-by-step for
deep understanding.
"""

import math
from typing import Union, Set, List, Tuple, Optional


class Value:
    """
    A Value wraps a single scalar value and tracks operations for autodiff.

    This is the core building block of our autograd engine. Every operation
    creates a new Value that remembers how it was created, enabling automatic
    gradient computation through the chain rule.
    """

    def __init__(
        self,
        data: float,
        children: Tuple["Value", ...] = (),
        op: str = "",
        label: str = "",
    ):
        """
        Initialize a Value node.

        Args:
            data: The scalar value this node holds
            children: The Values that were used to create this Value
            op: String representation of the operation that created this Value
            label: Optional label for debugging/visualization
        """
        self.data = data
        self.grad = 0.0  # Gradient of some scalar output w.r.t. this Value
        self._backward = lambda: None  # Function to compute gradients of children
        self._prev = set(children)  # Set of children nodes
        self._op = op  # Operation that created this node
        self.label = label  # Optional label for visualization

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Union["Value", float, int]) -> "Value":
        """Addition: self + other"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # Gradient of addition: d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: Union["Value", float, int]) -> "Value":
        """Reverse addition: other + self"""
        return self + other

    def __mul__(self, other: Union["Value", float, int]) -> "Value":
        """Multiplication: self * other"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # Gradient of multiplication: d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other: Union["Value", float, int]) -> "Value":
        """Reverse multiplication: other * self"""
        return self * other

    def __truediv__(self, other: Union["Value", float, int]) -> "Value":
        """Division: self / other"""
        return self * other**-1

    def __pow__(self, other: Union[float, int]) -> "Value":
        """Power: self**other (other must be int/float)"""
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            # Gradient of power: d(a^n)/da = n * a^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __neg__(self) -> "Value":
        """Negation: -self"""
        return self * -1

    def __sub__(self, other: Union["Value", float, int]) -> "Value":
        """Subtraction: self - other"""
        return self + (-other)

    def __rsub__(self, other: Union["Value", float, int]) -> "Value":
        """Reverse subtraction: other - self"""
        return other + (-self)

    def exp(self) -> "Value":
        """Exponential function: e^self"""
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # Gradient of exp: d(e^x)/dx = e^x
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def log(self) -> "Value":
        """Natural logarithm: ln(self)"""
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # Gradient of log: d(ln(x))/dx = 1/x
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward

        return out

    def tanh(self) -> "Value":
        """Hyperbolic tangent activation function with numerical stability"""
        x = self.data
        # Use more numerically stable tanh computation
        if x > 20:
            t = 1.0
        elif x < -20:
            t = -1.0
        else:
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        
        out = Value(t, (self,), "tanh")

        def _backward():
            # Gradient of tanh: d(tanh(x))/dx = 1 - tanh^2(x)
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> "Value":
        """ReLU activation function: max(0, x)"""
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            # Gradient of ReLU: 1 if x > 0, 0 if x <= 0
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self) -> "Value":
        """Sigmoid activation function: 1 / (1 + e^(-x))"""
        exp_neg_x = (-self).exp()
        out = 1.0 / (1.0 + exp_neg_x)
        return out

    def backward(self):
        """
        Compute gradients for all nodes in the computational graph.

        This implements reverse-mode automatic differentiation (backpropagation).
        We traverse the graph in topological order (reverse of forward pass)
        and apply the chain rule at each node.
        """
        # Build topological ordering of all nodes in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Initialize gradient of output to 1
        self.grad = 1.0

        # Traverse in reverse topological order and apply chain rule
        for node in reversed(topo):
            node._backward()


def trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    """
    Build a set of all nodes and edges in the computational graph.
    Useful for visualization and debugging.
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


# Example usage and testing
if __name__ == "__main__":
    # Test basic operations
    print("Testing basic operations...")

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    # Forward pass: f = (a*b + c)
    d = a * b  # d = -6
    e = d + c  # e = 4
    f = e.tanh()  # f = tanh(4) ≈ 0.999

    print(f"Forward pass result: {f}")

    # Backward pass
    f.backward()

    print(f"Gradients:")
    print(f"df/da = {a.grad}")  # Should be b.data * (1 - f.data^2) = -3 * (1 - 0.999^2)
    print(f"df/db = {b.grad}")  # Should be a.data * (1 - f.data^2) = 2 * (1 - 0.999^2)
    print(f"df/dc = {c.grad}")  # Should be (1 - f.data^2) = (1 - 0.999^2)

    print("\nMicrograd engine implemented successfully! ✅")
