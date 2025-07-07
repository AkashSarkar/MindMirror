import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.micrograd.engine import Value
from lib.micrograd.nn import Neuron
import random
import sys
import os


# Single neuron: takes multiple inputs, produces one output
# This is like a tiny "decision maker" in your AI


def test_single_neuron():
    print("=== Single Neuron Example ===")

    # Create a neuron with 3 inputs
    neuron = Neuron(3)

    # Example: encoding "cat" as numbers [0.1, 0.5, 0.8]
    inputs = [Value(0.1), Value(0.5), Value(0.8)]

    # Neuron processes the input
    output = neuron(inputs)

    print(f"Input 'cat' as numbers: {[x.data for x in inputs]}")
    print(f"Neuron output: {output.data}")
    print(f"Neuron has {len(neuron.w)} weights and 1 bias")

    return neuron, inputs, output


if __name__ == "__main__":
    test_single_neuron()
