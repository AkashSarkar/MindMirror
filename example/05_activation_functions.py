import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.micrograd.engine import Value
from lib.micrograd.nn import Neuron
import math


def demonstrate_activation_functions():
    """
    Show what activation functions are and why they're essential
    This is the key to understanding why neural networks work!
    """
    print("=== Understanding Activation Functions ===\n")
    
    print("🤔 WHAT IS AN ACTIVATION FUNCTION?")
    print("   An activation function decides whether a neuron should be 'activated' or not")
    print("   It's like a decision gate that transforms the input signal")
    print("   Without it, neural networks would just be linear algebra!")
    print()
    
    # Show what happens without activation
    print("⚠️  WITHOUT ACTIVATION FUNCTIONS:")
    print("   Neuron output = w1*x1 + w2*x2 + bias")
    print("   This is just a LINEAR equation (like y = mx + b)")
    print("   No matter how many layers you stack, it's still linear!")
    print("   Linear functions can only learn straight lines 📏")
    print()
    
    print("✨ WITH ACTIVATION FUNCTIONS:")
    print("   Neuron output = activation(w1*x1 + w2*x2 + bias)")
    print("   Now we can learn curves, circles, complex patterns! 🌊")
    print()


def compare_activation_functions():
    """
    Compare different activation functions with examples
    """
    print("🔍 COMMON ACTIVATION FUNCTIONS:\n")
    
    # Test inputs
    test_inputs = [-3, -1, 0, 1, 3]
    
    print("📊 INPUT VALUES:", test_inputs)
    print("=" * 60)
    
    # Linear (no activation)
    print("📏 LINEAR (No Activation):")
    print("   Formula: f(x) = x")
    print("   Output: ", [x for x in test_inputs])
    print("   Problem: Always linear, can't learn complex patterns")
    print()
    
    # Tanh activation
    print("🌊 TANH (Hyperbolic Tangent):")
    print("   Formula: f(x) = (e^2x - 1) / (e^2x + 1)")
    tanh_outputs = []
    for x in test_inputs:
        if x > 20:
            tanh_val = 1.0
        elif x < -20:
            tanh_val = -1.0
        else:
            tanh_val = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        tanh_outputs.append(round(tanh_val, 3))
    print("   Output: ", tanh_outputs)
    print("   Range: -1 to +1")
    print("   Shape: S-curve (sigmoid)")
    print("   ✅ Good for: Hidden layers, centered outputs")
    print()
    
    # ReLU activation
    print("⚡ RELU (Rectified Linear Unit):")
    print("   Formula: f(x) = max(0, x)")
    relu_outputs = [max(0, x) for x in test_inputs]
    print("   Output: ", relu_outputs)
    print("   Range: 0 to +∞")
    print("   Shape: Ramp function")
    print("   ✅ Good for: Modern deep networks, fast training")
    print()
    
    # Sigmoid activation
    print("🔄 SIGMOID:")
    print("   Formula: f(x) = 1 / (1 + e^(-x))")
    sigmoid_outputs = []
    for x in test_inputs:
        sigmoid_val = 1 / (1 + math.exp(-x))
        sigmoid_outputs.append(round(sigmoid_val, 3))
    print("   Output: ", sigmoid_outputs)
    print("   Range: 0 to 1")
    print("   Shape: S-curve")
    print("   ✅ Good for: Output layer (probabilities)")
    print()


def show_why_nonlinearity_matters():
    """
    Demonstrate why we need non-linear activation functions
    """
    print("=" * 60)
    print("🧠 WHY NON-LINEARITY IS ESSENTIAL")
    print("=" * 60)
    print()
    
    print("🔍 EXPERIMENT: Linear vs Non-linear Networks")
    print()
    
    # Create two neurons - one linear, one with tanh
    linear_neuron = Neuron(2)
    nonlinear_neuron = Neuron(2)
    
    # Override the call method for linear neuron
    class LinearNeuron:
        def __init__(self, nin):
            self.w = [Value((2 * (i / nin) - 1)) for i in range(nin)]
            self.b = Value(0.0)
        
        def __call__(self, x):
            # Linear activation (no tanh)
            weighted_sum = sum(wi * xi for wi, xi in zip(self.w, x))
            return weighted_sum + self.b
        
        def parameters(self):
            return self.w + [self.b]
    
    linear_neuron = LinearNeuron(2)
    
    print("📊 TESTING PATTERN RECOGNITION:")
    print("   Task: Classify points inside vs outside a circle")
    print()
    
    # Test points
    test_points = [
        [0.0, 0.0],   # Center (should be inside)
        [0.8, 0.8],   # Corner (should be outside)
        [0.3, 0.3],   # Medium (should be inside)
        [0.1, 0.9],   # Edge case
    ]
    
    print("🔲 LINEAR NEURON PREDICTIONS:")
    for point in test_points:
        inputs = [Value(x) for x in point]
        prediction = linear_neuron(inputs)
        inside_circle = (point[0]**2 + point[1]**2) < 0.5
        expected = "Inside" if inside_circle else "Outside"
        print(f"   Point {point}: prediction={prediction.data:.3f}, expected={expected}")
    print("   ❌ Linear can only create straight line boundaries!")
    print()
    
    print("🌊 NON-LINEAR NEURON PREDICTIONS:")
    for point in test_points:
        inputs = [Value(x) for x in point]
        prediction = nonlinear_neuron(inputs)
        inside_circle = (point[0]**2 + point[1]**2) < 0.5
        expected = "Inside" if inside_circle else "Outside"
        print(f"   Point {point}: prediction={prediction.data:.3f}, expected={expected}")
    print("   ✅ Non-linear can learn curved boundaries!")
    print()


def visualize_activation_curves():
    """
    Show what activation functions look like as curves
    """
    print("📈 ACTIVATION FUNCTION SHAPES:")
    print()
    
    print("📏 LINEAR: f(x) = x")
    print("    ╱")
    print("   ╱ ")
    print("  ╱  ")
    print(" ╱   ")
    print("╱────")
    print("Always a straight line - no curves!")
    print()
    
    print("🌊 TANH: f(x) = tanh(x)")
    print("     ─────")
    print("   ╱       ")
    print("  ╱        ")
    print(" ╱         ")
    print("╱          ")
    print("│          ")
    print("╲          ")
    print(" ╲         ")
    print("  ╲        ")
    print("   ╲       ")
    print("     ─────")
    print("S-shaped curve - smooth transitions!")
    print()
    
    print("⚡ RELU: f(x) = max(0, x)")
    print("    ╱")
    print("   ╱ ")
    print("  ╱  ")
    print(" ╱   ")
    print("╱────")
    print("Ramp function - simple but effective!")
    print()


def show_real_world_analogy():
    """
    Explain activation functions with real-world analogies
    """
    print("=" * 60)
    print("🌍 REAL-WORLD ANALOGIES")
    print("=" * 60)
    print()
    
    print("🧠 THINK OF A NEURON LIKE A PERSON MAKING A DECISION:")
    print()
    
    print("📊 INPUT SIGNALS = Information you receive")
    print("   - Friend says 'let's go to movie' (+0.8)")
    print("   - Weather is rainy (-0.3)")
    print("   - You have money (+0.5)")
    print("   Weighted sum = 0.8 - 0.3 + 0.5 = +1.0")
    print()
    
    print("🔄 ACTIVATION FUNCTION = How you process the decision")
    print()
    
    print("📏 LINEAR PERSON:")
    print("   'My decision strength is exactly 1.0'")
    print("   Problem: Too mechanical, no real personality!")
    print()
    
    print("🌊 TANH PERSON:")
    print("   'I'm pretty excited (+0.76) but not overly so'")
    print("   Benefit: Realistic human-like responses!")
    print()
    
    print("⚡ RELU PERSON:")
    print("   'Either I'm not interested (0) or I am (1.0)'")
    print("   Benefit: Clear decisions, fast thinking!")
    print()
    
    print("💡 THE KEY INSIGHT:")
    print("   Without activation functions, all neurons would think exactly alike")
    print("   With activation functions, each neuron can have a unique 'personality'")
    print("   This diversity is what makes neural networks intelligent!")


if __name__ == "__main__":
    demonstrate_activation_functions()
    compare_activation_functions()
    show_why_nonlinearity_matters()
    visualize_activation_curves()
    show_real_world_analogy()
    
    print("\n🎯 SUMMARY:")
    print("   ✅ Activation functions add non-linearity")
    print("   ✅ Non-linearity enables learning complex patterns")
    print("   ✅ Different activations have different strengths")
    print("   ✅ Tanh: Good for hidden layers (-1 to +1)")
    print("   ✅ ReLU: Modern choice, fast training (0 to +∞)")
    print("   ✅ Sigmoid: Good for probabilities (0 to 1)")
    print("\n🚀 Ready for text generation? That's where it gets exciting!")
