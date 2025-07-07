"""
MindMirror - Neural Network Layers

Building neural network components on top of our micrograd engine.
Each layer is implemented from scratch to understand the mathematical
foundations of deep learning.
"""

import random
from typing import List
from .engine import Value


class Neuron:
    """
    A single neuron with weights, bias, and an activation function.
    
    This is the fundamental building block of neural networks.
    A neuron computes: activation(sum(wi * xi) + b)
    """
    
    def __init__(self, nin: int, activation: str = 'tanh'):
        """
        Initialize a neuron with random weights and bias.
        
        Args:
            nin: Number of input connections
            activation: Activation function ('tanh', 'relu', 'sigmoid', 'linear')
        """
        # Initialize weights randomly between -1 and 1
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))  # Bias term
        self.activation = activation
    
    def __call__(self, x: List[Value]) -> Value:
        """
        Forward pass through the neuron.
        
        Args:
            x: List of input Values
            
        Returns:
            Output Value after applying weights, bias, and activation
        """
        # Compute weighted sum: sum(wi * xi) + b
        weighted_sum = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        
        # Apply activation function
        if self.activation == 'tanh':
            return weighted_sum.tanh()
        elif self.activation == 'relu':
            return weighted_sum.relu()
        elif self.activation == 'sigmoid':
            return weighted_sum.sigmoid()
        elif self.activation == 'linear':
            return weighted_sum
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def parameters(self) -> List[Value]:
        """Return all trainable parameters (weights + bias)"""
        return self.w + [self.b]


class Layer:
    """
    A layer of neurons - the basic building block of multi-layer networks.
    """
    
    def __init__(self, nin: int, nout: int, activation: str = 'tanh'):
        """
        Initialize a layer with multiple neurons.
        
        Args:
            nin: Number of inputs to each neuron
            nout: Number of neurons in this layer (outputs)
            activation: Activation function for all neurons in layer
        """
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x: List[Value]) -> List[Value]:
        """
        Forward pass through the layer.
        
        Args:
            x: List of input Values
            
        Returns:
            List of output Values (one per neuron)
        """
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def parameters(self) -> List[Value]:
        """Return all trainable parameters in the layer"""
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    Multi-Layer Perceptron - a feedforward neural network.
    
    This is our first complete neural network implementation.
    It consists of multiple layers stacked together.
    """
    
    def __init__(self, nin: int, nouts: List[int], activations: List[str] = None):
        """
        Initialize an MLP with specified architecture.
        
        Args:
            nin: Number of input features
            nouts: List of output sizes for each layer
            activations: List of activation functions for each layer
        """
        if activations is None:
            # Default: tanh for hidden layers, linear for output
            activations = ['tanh'] * (len(nouts) - 1) + ['linear']
        
        assert len(activations) == len(nouts), "Need activation for each layer"
        
        # Build layers
        sizes = [nin] + nouts
        self.layers = []
        
        for i in range(len(nouts)):
            layer = Layer(sizes[i], sizes[i+1], activations[i])
            self.layers.append(layer)
    
    def __call__(self, x: List[Value]) -> List[Value]:
        """
        Forward pass through the entire network.
        
        Args:
            x: Input values
            
        Returns:
            Output values from final layer
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[Value]:
        """Return all trainable parameters in the network"""
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        """Reset gradients to zero (important before each backward pass)"""
        for p in self.parameters():
            p.grad = 0.0


# Loss functions
def mse_loss(predictions: List[Value], targets: List[float]) -> Value:
    """
    Mean Squared Error loss function.
    
    Args:
        predictions: Model outputs
        targets: True target values
        
    Returns:
        MSE loss as a Value (enables automatic differentiation)
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    losses = [(pred - target)**2 for pred, target in zip(predictions, targets)]
    total_loss = sum(losses, Value(0.0))  # Sum all losses
    return total_loss * (1.0 / len(losses))  # Average


def cross_entropy_loss(logits: List[Value], targets: List[int]) -> Value:
    """
    Cross-entropy loss for classification.
    
    Args:
        logits: Raw model outputs (before softmax)
        targets: True class indices
        
    Returns:
        Cross-entropy loss
    """
    # Softmax: convert logits to probabilities
    exp_logits = [logit.exp() for logit in logits]
    sum_exp = sum(exp_logits, Value(0.0))
    probs = [exp_logit / sum_exp for exp_logit in exp_logits]
    
    # Cross-entropy: -log(prob_correct_class)
    losses = [-probs[target].log() for target in targets]
    return sum(losses, Value(0.0)) * (1.0 / len(losses))


# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Network components...")
    
    # Test a simple 2-layer MLP
    model = MLP(3, [4, 2, 1])  # 3 inputs -> 4 hidden -> 2 hidden -> 1 output
    
    print(f"Model has {len(model.parameters())} parameters")
    
    # Test forward pass
    x = [Value(1.0), Value(2.0), Value(-1.0)]
    y_pred = model(x)
    
    print(f"Forward pass output: {y_pred}")
    
    # Test backward pass with MSE loss
    y_target = [0.5]  # Target value
    loss = mse_loss(y_pred, y_target)
    
    print(f"Loss: {loss}")
    
    # Compute gradients
    model.zero_grad()  # Clear previous gradients
    loss.backward()
    
    print(f"First few parameter gradients: {[p.grad for p in model.parameters()[:5]]}")
    
    print("\nNeural network layers implemented successfully! ✅")
