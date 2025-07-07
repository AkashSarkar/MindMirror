"""
MindMirror Library

Core neural network components built from scratch for educational purposes.
This library contains the fundamental building blocks for AI development.
"""

from .micrograd.engine import Value
from .micrograd.nn import Neuron, Layer, MLP
from .micrograd.optim import SGD, Adam

__version__ = "0.1.0"
__all__ = ["Value", "Neuron", "Layer", "MLP", "SGD", "Adam"]
