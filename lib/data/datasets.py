"""
MindMirror - Data Loading and Processing

Utilities for loading and processing text data for language model training.
"""

import os
import random
from typing import List, Iterator, Tuple, Dict, Optional
import pickle


class TextDataset:
    """
    A dataset class for text data that can be used for training language models.
    """

    def __init__(self, texts: List[str], sequence_length: int = 32, stride: int = 1):
        """
        Initialize the text dataset.

        Args:
            texts: List of text strings
            sequence_length: Length of text sequences for training
            stride: Stride for creating sequences (1 = every character)
        """
        self.texts = texts
        self.sequence_length = sequence_length
        self.stride = stride

        # Build character vocabulary
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self._build_vocabulary()

        # Create training sequences
        self.sequences = []
        self._create_sequences()

    def _build_vocabulary(self):
        """Build character vocabulary from all texts."""
        chars = set()

        # Add special tokens
        chars.add("<START>")
        chars.add("<END>")

        # Add all characters from texts
        for text in self.texts:
            chars.update(text.lower())

        # Create mappings
        sorted_chars = sorted(chars)
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(sorted_chars)

        print(f"Built vocabulary with {self.vocab_size} characters")
        print(f"Vocabulary: {''.join(sorted_chars)}")

    def _create_sequences(self):
        """Create training sequences from texts."""
        for text in self.texts:
            # Add start and end tokens
            text = "<START>" + text.lower() + "<END>"

            # Create sequences of specified length
            for i in range(0, len(text) - self.sequence_length, self.stride):
                sequence = text[i : i + self.sequence_length + 1]  # +1 for target
                self.sequences.append(sequence)

        print(
            f"Created {len(self.sequences)} sequences of length {self.sequence_length}"
        )

    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to list of character indices."""
        return [self.char_to_idx.get(char, 0) for char in text]

    def indices_to_text(self, indices: List[int]) -> str:
        """Convert list of indices to text."""
        return "".join(self.idx_to_char.get(idx, "?") for idx in indices)

    def get_batch(self, batch_size: int) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Get a random batch of training data.

        Args:
            batch_size: Size of the batch

        Returns:
            (inputs, targets) where each is a list of sequences as index lists
        """
        batch_sequences = random.sample(
            self.sequences, min(batch_size, len(self.sequences))
        )

        inputs = []
        targets = []

        for sequence in batch_sequences:
            input_text = sequence[:-1]  # All but last character
            target_text = sequence[1:]  # All but first character

            input_indices = self.text_to_indices(input_text)
            target_indices = self.text_to_indices(target_text)

            inputs.append(input_indices)
            targets.append(target_indices)

        return inputs, targets

    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """Get a single training example."""
        sequence = self.sequences[idx]
        input_text = sequence[:-1]
        target_text = sequence[1:]

        input_indices = self.text_to_indices(input_text)
        target_indices = self.text_to_indices(target_text)

        return input_indices, target_indices


def load_text_files(
    directory: str, extensions: List[str] = [".txt", ".md"]
) -> List[str]:
    """
    Load text files from a directory.

    Args:
        directory: Directory path to load files from
        extensions: List of file extensions to include

    Returns:
        List of text strings
    """
    texts = []

    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return texts

    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in extensions):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                        print(f"Loaded {filename}: {len(content)} characters")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return texts


def save_dataset(dataset: TextDataset, filepath: str):
    """Save dataset to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {filepath}")


def load_dataset(filepath: str) -> TextDataset:
    """Load dataset from disk."""
    with open(filepath, "rb") as f:
        dataset = pickle.load(f)
    print(f"Dataset loaded from {filepath}")
    return dataset


# Sample text data for experimentation
SAMPLE_TEXTS = [
    "The art of programming is the art of organizing complexity.",
    "Any sufficiently advanced technology is indistinguishable from magic.",
    "The best way to predict the future is to invent it.",
    "Programs must be written for people to read, and only incidentally for machines to execute.",
    "The computer was born to solve problems that did not exist before.",
    "Software is a great combination between artistry and engineering.",
    "Code is like humor. When you have to explain it, it's bad.",
    "Programming isn't about what you know; it's about what you can figure out.",
    "The most important thing in programming is to write code that humans can understand.",
    "Debugging is twice as hard as writing the code in the first place.",
]


# Example usage
if __name__ == "__main__":
    print("Testing TextDataset...")

    # Create dataset from sample texts
    dataset = TextDataset(SAMPLE_TEXTS, sequence_length=16, stride=4)

    print(f"Dataset has {len(dataset)} sequences")
    print(f"Vocabulary size: {dataset.vocab_size}")

    # Get a batch
    inputs, targets = dataset.get_batch(batch_size=3)

    print("\nSample batch:")
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        inp_text = dataset.indices_to_text(inp)
        tgt_text = dataset.indices_to_text(tgt)
        print(f"Sample {i+1}:")
        print(f"  Input:  '{inp_text}'")
        print(f"  Target: '{tgt_text}'")

    print("\nTextDataset implemented successfully! ✅")
