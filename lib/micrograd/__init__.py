"""
MindMirror - Micrograd Package

A minimal automatic differentiation engine and neural network library
built from scratch for educational purposes.
"""

from .engine import Value, trace
from .nn import Neuron, Layer, MLP, mse_loss, cross_entropy_loss
from .optim import SGD, Momentum, AdaGrad, Adam, StepLR, ExponentialLR

__version__ = "0.1.0"
__author__ = "MindMirror Project"

__all__ = [
    # Core autograd
    "Value",
    "trace",
    # Neural network components
    "Neuron",
    "Layer",
    "MLP",
    # Loss functions
    "mse_loss",
    "cross_entropy_loss",
    # Optimizers
    "SGD",
    "Momentum",
    "AdaGrad",
    "Adam",
    # Learning rate schedulers
    "StepLR",
    "ExponentialLR",
]
