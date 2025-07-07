import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.micrograd.engine import Value
from lib.micrograd.nn import MLP
import random


def demonstrate_multi_layer_network():
    """
    Build a multi-layer neural network to solve a more complex problem
    This shows how connecting neurons creates intelligence!
    """
    print("=== Multi-Layer Neural Network ===\n")

    # Create a network: 2 inputs -> 4 hidden neurons -> 2 hidden neurons -> 1 output
    network = MLP(2, [4, 2, 1])

    print("🏗️ NETWORK ARCHITECTURE:")
    print("   Input Layer: 2 neurons (for x, y coordinates)")
    print("   Hidden Layer 1: 4 neurons")
    print("   Hidden Layer 2: 2 neurons")
    print("   Output Layer: 1 neuron (classification)")
    print(f"   Total parameters: {len(network.parameters())}")
    print()

    # Complex problem: classify points as inside or outside a circle
    # Circle: x² + y² < 0.5 (inside = 1, outside = -1)
    def generate_data_point():
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        inside_circle = 1.0 if (x * x + y * y) < 0.5 else -1.0
        return [x, y], inside_circle

    # Generate training data
    training_data = [generate_data_point() for _ in range(100)]

    print("📊 PROBLEM: Classify points as inside/outside a circle")
    print("   Circle equation: x² + y² < 0.5")
    print("   Inside circle = +1, Outside circle = -1")
    print(f"   Training on {len(training_data)} examples")
    print()

    # Show some examples
    print("📝 SAMPLE TRAINING DATA:")
    for i in range(5):
        point, label = training_data[i]
        distance = (point[0] ** 2 + point[1] ** 2) ** 0.5
        print(
            f"   Point ({point[0]:.2f}, {point[1]:.2f}), distance={distance:.2f}, label={label}"
        )
    print()

    # Test before training
    print("🔍 BEFORE TRAINING (random predictions):")
    test_points = [
        ([0.0, 0.0], 1.0),  # Center - should be inside
        ([0.8, 0.8], -1.0),  # Far corner - should be outside
        ([0.3, 0.3], 1.0),  # Close to center - should be inside
    ]

    for point, expected in test_points:
        inputs = [Value(x) for x in point]
        prediction = network(inputs)
        # Handle both single Value and list of Values
        if isinstance(prediction, list):
            prediction = prediction[0]
        print(f"   Point {point}: predicted {prediction.data:.3f}, expected {expected}")
    print()

    # Training loop
    learning_rate = 0.001  # Much smaller learning rate
    print(f"🏃‍♂️ TRAINING (learning rate = {learning_rate}):")

    for epoch in range(100):  # Fewer epochs
        # Shuffle data each epoch
        random.shuffle(training_data)
        total_loss = Value(0.0)

        for inputs, target in training_data:
            # Forward pass
            input_values = [Value(x) for x in inputs]
            outputs = network(input_values)
            # Handle both single Value and list of Values
            prediction = outputs[0] if isinstance(outputs, list) else outputs

            # Loss calculation
            loss = (prediction - Value(target)) ** 2
            total_loss = total_loss + loss

            # Backward pass
            loss.backward()

        # Update all parameters with gradient clipping
        for param in network.parameters():
            # Clip gradients to prevent explosion
            if param.grad > 10:
                param.grad = 10
            elif param.grad < -10:
                param.grad = -10

            param.data -= learning_rate * param.grad
            param.grad = 0.0

        # Print progress
        if epoch % 20 == 0:
            avg_loss = total_loss.data / len(training_data)
            print(f"   Epoch {epoch}: Average Loss = {avg_loss:.4f}")

    print()

    # Test after training
    print("🎯 AFTER TRAINING:")
    correct = 0
    total = 0

    for point, expected in test_points:
        inputs = [Value(x) for x in point]
        prediction = network(inputs)
        # Handle both single Value and list of Values
        if isinstance(prediction, list):
            prediction = prediction[0]
        predicted_class = 1.0 if prediction.data > 0 else -1.0
        is_correct = predicted_class == expected
        correct += is_correct
        total += 1

        print(
            f"   Point {point}: predicted {prediction.data:.3f} → {predicted_class}, expected {expected} {'✓' if is_correct else '✗'}"
        )

    accuracy = correct / total * 100
    print(f"   Accuracy: {accuracy:.1f}%")
    print()

    # Test on more points to see the pattern
    print("✨ TESTING ON GRID OF POINTS:")
    print("   Visualizing what the network learned...")
    test_grid = []
    for x in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        for y in [-0.8, -0.4, 0.0, 0.4, 0.8]:
            inputs = [Value(x), Value(y)]
            prediction = network(inputs)
            # Handle both single Value and list of Values
            if isinstance(prediction, list):
                prediction = prediction[0]
            predicted_class = "🔵" if prediction.data > 0 else "🔴"
            actual_class = "🔵" if (x * x + y * y) < 0.5 else "🔴"
            test_grid.append((x, y, predicted_class, actual_class))

    # Display as a grid
    for i, (x, y, pred, actual) in enumerate(test_grid):
        if i % 5 == 0:
            print()
        symbol = pred if pred == actual else "❌"
        print(f"{symbol}", end=" ")

    print("\n")
    print("   Legend: 🔵 = Inside circle, 🔴 = Outside circle, ❌ = Wrong prediction")
    print()

    return network


def compare_with_single_neuron():
    """
    Show why multiple layers are more powerful than a single neuron
    """
    print("=" * 60)
    print("🧠 WHY MULTIPLE LAYERS MATTER")
    print("=" * 60)
    print()

    print("🔍 SINGLE NEURON LIMITATION:")
    print("   - Can only learn LINEAR patterns (straight lines)")
    print("   - Cannot learn circles, curves, or complex shapes")
    print("   - Limited to problems like: 'is x + y > threshold?'")
    print()

    print("✨ MULTI-LAYER NETWORK POWER:")
    print("   - Can learn ANY pattern (circles, spirals, complex boundaries)")
    print("   - Each layer transforms the data into a new representation")
    print("   - Hidden layers discover useful features automatically")
    print()

    print("🎯 WHAT YOU JUST BUILT:")
    print("   - Layer 1: Detects basic features (edges, corners)")
    print("   - Layer 2: Combines features into patterns")
    print("   - Layer 3: Makes final decision based on patterns")
    print("   - This is the SAME principle used in image recognition!")
    print()

    print("🚀 NEXT STEPS:")
    print("   - Text generation (character-by-character)")
    print("   - Sequence modeling (understanding order)")
    print("   - Attention mechanisms (focusing on important parts)")
    print("   - Transformers (the architecture behind ChatGPT)")


if __name__ == "__main__":
    network = demonstrate_multi_layer_network()
    compare_with_single_neuron()
