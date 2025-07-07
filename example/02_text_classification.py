import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.micrograd.engine import Value
from lib.micrograd.nn import Neuron
import random


def simple_text_classifier():
    """
    Build a neuron that learns to classify text as positive/negative
    This is a micro-version of sentiment analysis!
    """
    print("=== Text Classification with Neural Network ===")

    # Simple word encoding (normally you'd use better methods)
    word_to_number = {
        "good": 0.8,
        "great": 0.9,
        "awesome": 1.0,
        "bad": 0.1,
        "terrible": 0.0,
        "awful": 0.05,
    }

    # Training data: [word_encoding] -> sentiment (1.0 = positive, 0.0 = negative)
    training_data = [
        ([word_to_number["good"]], 1.0),  # "good" -> positive
        ([word_to_number["bad"]], 0.0),  # "bad" -> negative
        ([word_to_number["great"]], 1.0),  # "great" -> positive
        ([word_to_number["awful"]], 0.0),  # "awful" -> negative
    ]

    # Create classifier neuron
    classifier = Neuron(1)  # 1 input (word encoding)

    print("Training the classifier...")

    # Training loop
    for epoch in range(100):
        total_loss = Value(0.0)

        for word_encoding, target_sentiment in training_data:
            # Forward pass
            inputs = [Value(word_encoding[0])]
            prediction = classifier(inputs)
            target = Value(target_sentiment)

            # Loss calculation
            loss = (prediction - target) ** 2
            total_loss = total_loss + loss

        # Backward pass
        total_loss.backward()

        # Update weights (simple gradient descent)
        learning_rate = 0.01
        for param in classifier.parameters():
            param.data -= learning_rate * param.grad
            param.grad = 0.0  # Reset gradient

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.data:.4f}")

    # Test the trained classifier
    print("\n=== Testing Trained Classifier ===")
    test_words = ["good", "bad", "awesome", "terrible"]

    for word in test_words:
        if word in word_to_number:
            inputs = [Value(word_to_number[word])]
            prediction = classifier(inputs)
            print(
                f"'{word}' -> {prediction.data:.3f} ({'Positive' if prediction.data > 0.5 else 'Negative'})"
            )


if __name__ == "__main__":
    simple_text_classifier()
