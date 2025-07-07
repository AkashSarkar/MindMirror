"""
MindMirror - Optimization Algorithms

Implementing gradient-based optimization algorithms from scratch.
Understanding how neural networks learn through gradient descent
and its variants.
"""

from typing import List
from .engine import Value


class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, parameters: List[Value]):
        self.parameters = parameters
    
    def step(self):
        """Update parameters - to be implemented by subclasses"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Reset gradients to zero"""
        for p in self.parameters:
            p.grad = 0.0


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    The most fundamental optimization algorithm for neural networks.
    Updates parameters in the direction of negative gradient.
    """
    
    def __init__(self, parameters: List[Value], lr: float = 0.01):
        """
        Initialize SGD optimizer.
        
        Args:
            parameters: List of Values to optimize
            lr: Learning rate (step size)
        """
        super().__init__(parameters)
        self.lr = lr
    
    def step(self):
        """
        Perform one optimization step.
        
        Updates each parameter: p = p - lr * grad
        """
        for p in self.parameters:
            p.data -= self.lr * p.grad


class Momentum(Optimizer):
    """
    SGD with Momentum optimizer.
    
    Adds momentum to help accelerate SGD in relevant directions
    and dampen oscillations. Particularly useful for navigating
    ravines and areas of high curvature.
    """
    
    def __init__(self, parameters: List[Value], lr: float = 0.01, momentum: float = 0.9):
        """
        Initialize Momentum optimizer.
        
        Args:
            parameters: List of Values to optimize
            lr: Learning rate
            momentum: Momentum coefficient (typically 0.9)
        """
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        # Initialize velocity for each parameter
        self.velocities = [0.0 for _ in parameters]
    
    def step(self):
        """
        Perform one optimization step with momentum.
        
        v = momentum * v + lr * grad
        p = p - v
        """
        for i, p in enumerate(self.parameters):
            # Update velocity
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * p.grad
            # Update parameter
            p.data -= self.velocities[i]


class AdaGrad(Optimizer):
    """
    Adaptive Gradient Algorithm (AdaGrad).
    
    Adapts the learning rate for each parameter based on historical
    gradients. Parameters with large gradients get smaller learning rates.
    """
    
    def __init__(self, parameters: List[Value], lr: float = 0.01, eps: float = 1e-8):
        """
        Initialize AdaGrad optimizer.
        
        Args:
            parameters: List of Values to optimize
            lr: Initial learning rate
            eps: Small constant for numerical stability
        """
        super().__init__(parameters)
        self.lr = lr
        self.eps = eps
        # Accumulate squared gradients
        self.squared_grads = [0.0 for _ in parameters]
    
    def step(self):
        """
        Perform one optimization step with adaptive learning rates.
        
        G += grad^2
        p = p - lr * grad / (sqrt(G) + eps)
        """
        for i, p in enumerate(self.parameters):
            # Accumulate squared gradient
            self.squared_grads[i] += p.grad ** 2
            # Adaptive learning rate
            adapted_lr = self.lr / (self.squared_grads[i] ** 0.5 + self.eps)
            # Update parameter
            p.data -= adapted_lr * p.grad


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    
    Combines the advantages of momentum and adaptive learning rates.
    Maintains moving averages of both gradients and squared gradients.
    Currently the most popular optimizer for deep learning.
    """
    
    def __init__(self, parameters: List[Value], lr: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            parameters: List of Values to optimize
            lr: Learning rate
            beta1: Decay rate for first moment estimate (momentum)
            beta2: Decay rate for second moment estimate (squared gradients)
            eps: Small constant for numerical stability
        """
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialize moment estimates
        self.m = [0.0 for _ in parameters]  # First moment (momentum)
        self.v = [0.0 for _ in parameters]  # Second moment (squared grads)
        self.t = 0  # Time step
    
    def step(self):
        """
        Perform one optimization step with Adam.
        
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)  # Bias correction
        v_hat = v / (1 - beta2^t)  # Bias correction
        p = p - lr * m_hat / (sqrt(v_hat) + eps)
        """
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameter
            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)


# Learning rate schedulers
class LRScheduler:
    """Base class for learning rate schedulers"""
    
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.lr
    
    def step(self, epoch: int):
        """Update learning rate - to be implemented by subclasses"""
        raise NotImplementedError


class StepLR(LRScheduler):
    """
    Multiply learning rate by gamma every step_size epochs.
    """
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self, epoch: int):
        """Update learning rate every step_size epochs"""
        if epoch > 0 and epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class ExponentialLR(LRScheduler):
    """
    Multiply learning rate by gamma every epoch.
    """
    
    def __init__(self, optimizer: Optimizer, gamma: float):
        super().__init__(optimizer)
        self.gamma = gamma
    
    def step(self, epoch: int):
        """Exponentially decay learning rate"""
        self.optimizer.lr = self.initial_lr * (self.gamma ** epoch)


# Example usage and testing
if __name__ == "__main__":
    print("Testing optimization algorithms...")
    
    # Create some dummy parameters
    params = [Value(1.0), Value(2.0), Value(-0.5)]
    
    # Set some dummy gradients
    params[0].grad = 0.1
    params[1].grad = -0.2
    params[2].grad = 0.05
    
    print("Initial parameters:", [p.data for p in params])
    print("Gradients:", [p.grad for p in params])
    
    # Test SGD
    sgd = SGD(params, lr=0.1)
    sgd.step()
    print("After SGD step:", [p.data for p in params])
    
    # Reset and test Adam
    params = [Value(1.0), Value(2.0), Value(-0.5)]
    params[0].grad = 0.1
    params[1].grad = -0.2  
    params[2].grad = 0.05
    
    adam = Adam(params, lr=0.1)
    adam.step()
    print("After Adam step:", [p.data for p in params])
    
    print("\nOptimization algorithms implemented successfully! ✅")
