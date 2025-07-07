import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.micrograd.engine import Value
from lib.micrograd.nn import Neuron


def demonstrate_backward_step_by_step():
    """
    Show exactly what happens when we call loss.backward()
    This reveals the magic behind neural network learning!
    """
    print("=== Understanding loss.backward() ===\n")
    
    # Create a simple neuron
    neuron = Neuron(2)
    
    print("🧠 INITIAL NEURON STATE:")
    print(f"   Weight 1: {neuron.w[0].data:.3f}")
    print(f"   Weight 2: {neuron.w[1].data:.3f}")
    print(f"   Bias: {neuron.b.data:.3f}")
    print()
    
    # Create input and target
    x = [Value(0.5), Value(0.3)]
    target = Value(1.0)
    
    print("📊 FORWARD PASS:")
    print(f"   Input: [{x[0].data}, {x[1].data}]")
    print(f"   Target: {target.data}")
    
    # Forward pass - let's trace each step
    print("\n🔢 STEP-BY-STEP FORWARD CALCULATION:")
    
    # Step 1: Weighted sum
    w1_times_x1 = neuron.w[0] * x[0]
    print(f"   1. w1 * x1 = {neuron.w[0].data:.3f} * {x[0].data} = {w1_times_x1.data:.3f}")
    
    w2_times_x2 = neuron.w[1] * x[1]
    print(f"   2. w2 * x2 = {neuron.w[1].data:.3f} * {x[1].data} = {w2_times_x2.data:.3f}")
    
    sum_weighted = w1_times_x1 + w2_times_x2 + neuron.b
    print(f"   3. sum + bias = {w1_times_x1.data:.3f} + {w2_times_x2.data:.3f} + {neuron.b.data:.3f} = {sum_weighted.data:.3f}")
    
    # Step 2: Activation
    output = sum_weighted.tanh()
    print(f"   4. tanh({sum_weighted.data:.3f}) = {output.data:.3f}")
    
    # Step 3: Loss
    error = output - target
    loss = error * error
    print(f"   5. error = {output.data:.3f} - {target.data} = {error.data:.3f}")
    print(f"   6. loss = error² = {loss.data:.3f}")
    
    print(f"\n🎯 PREDICTION vs TARGET:")
    print(f"   Predicted: {output.data:.3f}")
    print(f"   Target: {target.data}")
    print(f"   Loss: {loss.data:.3f}")
    
    print(f"\n📊 GRADIENTS BEFORE BACKWARD (all should be 0):")
    print(f"   Weight 1 gradient: {neuron.w[0].grad}")
    print(f"   Weight 2 gradient: {neuron.w[1].grad}")
    print(f"   Bias gradient: {neuron.b.grad}")
    
    print(f"\n⚡ CALLING loss.backward() - THE MAGIC MOMENT!")
    loss.backward()
    
    print(f"\n📊 GRADIENTS AFTER BACKWARD:")
    print(f"   Weight 1 gradient: {neuron.w[0].grad:.6f}")
    print(f"   Weight 2 gradient: {neuron.w[1].grad:.6f}")
    print(f"   Bias gradient: {neuron.b.grad:.6f}")
    
    print(f"\n💡 WHAT THESE GRADIENTS MEAN:")
    print(f"   Weight 1: {'↗️ Increase' if neuron.w[0].grad < 0 else '↘️ Decrease'} to reduce loss")
    print(f"   Weight 2: {'↗️ Increase' if neuron.w[1].grad < 0 else '↘️ Decrease'} to reduce loss")
    print(f"   Bias: {'↗️ Increase' if neuron.b.grad < 0 else '↘️ Decrease'} to reduce loss")
    
    print(f"\n🔄 GRADIENT DESCENT UPDATE:")
    learning_rate = 0.1
    print(f"   Learning rate: {learning_rate}")
    
    old_w1 = neuron.w[0].data
    old_w2 = neuron.w[1].data
    old_b = neuron.b.data
    
    new_w1 = old_w1 - learning_rate * neuron.w[0].grad
    new_w2 = old_w2 - learning_rate * neuron.w[1].grad
    new_b = old_b - learning_rate * neuron.b.grad
    
    print(f"   New Weight 1: {old_w1:.3f} - {learning_rate} * {neuron.w[0].grad:.3f} = {new_w1:.3f}")
    print(f"   New Weight 2: {old_w2:.3f} - {learning_rate} * {neuron.w[1].grad:.3f} = {new_w2:.3f}")
    print(f"   New Bias: {old_b:.3f} - {learning_rate} * {neuron.b.grad:.3f} = {new_b:.3f}")
    
    # Actually update the weights
    neuron.w[0].data = new_w1
    neuron.w[1].data = new_w2
    neuron.b.data = new_b
    
    print(f"\n🎯 TESTING THE UPDATED NEURON:")
    # Reset gradients
    neuron.w[0].grad = 0
    neuron.w[1].grad = 0
    neuron.b.grad = 0
    
    # Forward pass with updated weights
    new_output = neuron(x)
    new_loss = (new_output - target) ** 2
    
    print(f"   Old prediction: {output.data:.3f}, Old loss: {loss.data:.3f}")
    print(f"   New prediction: {new_output.data:.3f}, New loss: {new_loss.data:.3f}")
    print(f"   Loss improvement: {loss.data - new_loss.data:.6f} {'✅' if new_loss.data < loss.data else '❌'}")
    
    return neuron


def show_computational_graph():
    """
    Visualize the computational graph that backward() traverses
    """
    print("\n" + "="*60)
    print("🔗 THE COMPUTATIONAL GRAPH")
    print("="*60)
    print()
    
    print("📈 FORWARD PASS (building the graph):")
    print("   x1 ──┐")
    print("       │ × w1 ──┐")
    print("   x2 ──┘        │ + ──> tanh ──> output ──> (output - target)² ──> LOSS")
    print("               bias")
    print()
    
    print("📉 BACKWARD PASS (traversing the graph):")
    print("   ∂loss/∂w1 ←──┐")
    print("               │ chain rule")
    print("   ∂loss/∂w2 ←──┤ ←── ∂loss/∂tanh ←── ∂loss/∂output ←── LOSS")
    print("               │")
    print("   ∂loss/∂bias ←┘")
    print()
    
    print("🧮 THE CHAIN RULE IN ACTION:")
    print("   ∂loss/∂w1 = ∂loss/∂output × ∂output/∂tanh × ∂tanh/∂sum × ∂sum/∂w1")
    print("              ↑             ↑              ↑            ↑")
    print("           2×error      1-tanh²         1           x1")
    print()
    
    print("💡 WHY THIS MATTERS:")
    print("   - Each operation in your network stores how to compute its gradient")
    print("   - backward() visits each operation in reverse order")
    print("   - Uses chain rule to combine gradients from 'downstream' operations")
    print("   - This is how your network 'learns' - by following the gradient!")


if __name__ == "__main__":
    neuron = demonstrate_backward_step_by_step()
    show_computational_graph()
