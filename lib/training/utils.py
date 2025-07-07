"""
MindMirror - Training Utilities

Helper functions for data processing, evaluation, and visualization.
"""

import random
import string
from typing import List, Dict, Tuple, Any


def clean_text(text: str) -> str:
    """
    Clean and normalize text for training.

    Args:
        text: Raw text string

    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()

    # Keep only letters, spaces, and basic punctuation
    allowed_chars = set(string.ascii_lowercase + " .,!?;:-'\"")
    text = "".join(c for c in text if c in allowed_chars)

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def load_text_data(file_paths: List[str]) -> List[str]:
    """
    Load text data from files.

    Args:
        file_paths: List of file paths to load

    Returns:
        List of text strings
    """
    texts = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                cleaned = clean_text(content)
                if cleaned.strip():  # Only add non-empty texts
                    texts.append(cleaned)
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")

    return texts


def create_sample_dataset() -> List[str]:
    """
    Create a sample text dataset for testing and experimentation.

    Returns:
        List of sample text strings
    """
    texts = [
        # Programming and AI related
        "python is a powerful programming language for machine learning",
        "neural networks learn patterns from data using backpropagation",
        "deep learning models can solve complex artificial intelligence problems",
        "transformers use attention mechanisms to process sequential data",
        "gradient descent optimizes neural network parameters iteratively",
        # General knowledge
        "the quick brown fox jumps over the lazy dog",
        "machine learning algorithms can recognize patterns in large datasets",
        "artificial intelligence research focuses on creating intelligent systems",
        "computer science involves algorithms, data structures, and programming",
        "software engineering requires careful design and implementation",
        # Conversational patterns
        "hello, how are you doing today?",
        "i am working on building an ai assistant from scratch",
        "this project helps me understand neural networks deeply",
        "learning by implementing everything from first principles",
        "coding every component helps build intuition for ai systems",
        # Technical concepts
        "backpropagation computes gradients using the chain rule",
        "attention mechanisms allow models to focus on relevant information",
        "embedding layers map discrete tokens to continuous vectors",
        "loss functions measure how well models perform on tasks",
        "optimization algorithms update parameters to minimize loss",
        # More varied content
        "natural language processing enables computers to understand text",
        "recurrent neural networks can model sequential patterns",
        "convolutional networks excel at processing image data",
        "reinforcement learning trains agents through trial and error",
        "unsupervised learning discovers hidden patterns in data",
    ]

    return texts


def analyze_text_statistics(texts: List[str]) -> Dict[str, Any]:
    """
    Analyze statistics of text data.

    Args:
        texts: List of text strings

    Returns:
        Dictionary with text statistics
    """
    # Combine all texts
    combined_text = " ".join(texts)

    # Character statistics
    char_counts = {}
    for char in combined_text:
        char_counts[char] = char_counts.get(char, 0) + 1

    # Word statistics
    words = combined_text.split()
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    stats = {
        "num_texts": len(texts),
        "total_chars": len(combined_text),
        "unique_chars": len(char_counts),
        "total_words": len(words),
        "unique_words": len(word_counts),
        "avg_text_length": sum(len(text) for text in texts) / len(texts),
        "char_frequencies": char_counts,
        "word_frequencies": word_counts,
        "vocabulary": sorted(char_counts.keys()),
        "most_common_chars": sorted(
            char_counts.items(), key=lambda x: x[1], reverse=True
        )[:10],
        "most_common_words": sorted(
            word_counts.items(), key=lambda x: x[1], reverse=True
        )[:10],
    }

    return stats


def print_text_statistics(stats: Dict[str, Any]):
    """Print text statistics in a readable format."""
    print("Text Dataset Statistics:")
    print("=" * 50)
    print(f"Number of texts: {stats['num_texts']}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Unique characters: {stats['unique_chars']}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"Unique words: {stats['unique_words']:,}")
    print(f"Average text length: {stats['avg_text_length']:.1f} characters")

    print(f"\nVocabulary: {repr(''.join(stats['vocabulary']))}")

    print(f"\nMost common characters:")
    for char, count in stats["most_common_chars"]:
        if char == " ":
            char_repr = "'space'"
        elif char == "\n":
            char_repr = "'newline'"
        else:
            char_repr = f"'{char}'"
        print(f"  {char_repr}: {count}")

    print(f"\nMost common words:")
    for word, count in stats["most_common_words"]:
        print(f"  '{word}': {count}")


def generate_training_curves_data(
    losses: List[float], smoothing_window: int = 10
) -> Tuple[List[float], List[float]]:
    """
    Generate smoothed training curves for visualization.

    Args:
        losses: List of loss values
        smoothing_window: Window size for moving average

    Returns:
        (original_losses, smoothed_losses)
    """
    if len(losses) < smoothing_window:
        return losses, losses

    smoothed = []
    for i in range(len(losses)):
        start_idx = max(0, i - smoothing_window // 2)
        end_idx = min(len(losses), i + smoothing_window // 2 + 1)
        window_losses = losses[start_idx:end_idx]
        smoothed_loss = sum(window_losses) / len(window_losses)
        smoothed.append(smoothed_loss)

    return losses, smoothed


# Evaluation utilities
def compute_model_size(model) -> int:
    """
    Compute total number of parameters in a model.

    Args:
        model: Model with get_parameters() method

    Returns:
        Number of parameters
    """
    if hasattr(model, "get_parameters"):
        return len(model.get_parameters())
    elif hasattr(model, "parameters"):
        return len(model.parameters())
    else:
        return 0


def print_model_info(model, model_name: str = "Model"):
    """
    Print information about a model.

    Args:
        model: Model to analyze
        model_name: Name for display
    """
    num_params = compute_model_size(model)

    print(f"{model_name} Information:")
    print("=" * 40)
    print(f"Total parameters: {num_params:,}")

    if hasattr(model, "vocab_size"):
        print(f"Vocabulary size: {model.vocab_size}")

    if hasattr(model, "embedding_dim"):
        print(f"Embedding dimension: {model.embedding_dim}")

    if hasattr(model, "hidden_dim"):
        print(f"Hidden dimension: {model.hidden_dim}")


# Example usage
if __name__ == "__main__":
    print("Testing training utilities...")

    # Create sample dataset
    texts = create_sample_dataset()
    print(f"Created sample dataset with {len(texts)} texts")

    # Analyze statistics
    stats = analyze_text_statistics(texts)
    print_text_statistics(stats)

    # Test text cleaning
    dirty_text = "  HELLO!!! This is a Test... with WEIRD   spacing!  "
    clean = clean_text(dirty_text)
    print(f"\nText cleaning example:")
    print(f"Original: {repr(dirty_text)}")
    print(f"Cleaned:  {repr(clean)}")

    print("\nTraining utilities implemented successfully! ✅")
