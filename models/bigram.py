"""
MindMirror - Character-Level Bigram Model

Our first language model! This implements a simple bigram model that predicts
the next character based on the current character. While simple, this teaches
the fundamentals of language modeling.

A bigram model predicts P(char_next | char_current) by counting character pairs
in training data. This is the foundation for understanding more complex models.
"""

import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import pickle
import os


class CharBigramModel:
    """
    A character-level bigram language model.
    
    This model learns to predict the next character given the current character
    by counting character pairs in the training data and computing probabilities.
    """
    
    def __init__(self, smoothing: float = 0.01):
        """
        Initialize the bigram model.
        
        Args:
            smoothing: Laplace smoothing parameter to handle unseen bigrams
        """
        self.smoothing = smoothing
        self.char_to_idx = {}  # Character to index mapping
        self.idx_to_char = {}  # Index to character mapping
        self.vocab_size = 0
        
        # Bigram counts: bigram_counts[char1][char2] = count
        self.bigram_counts = defaultdict(Counter)
        self.char_counts = Counter()  # Total count for each character
        
        # Special tokens
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        
        self.is_trained = False
    
    def _build_vocab(self, texts: List[str]):
        """Build character vocabulary from training texts."""
        chars = set()
        chars.add(self.START_TOKEN)
        chars.add(self.END_TOKEN)
        
        for text in texts:
            chars.update(text.lower())
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        
        print(f"Built vocabulary with {self.vocab_size} characters")
        print(f"Characters: {sorted(chars)}")
    
    def _count_bigrams(self, texts: List[str]):
        """Count character bigrams in training texts."""
        for text in texts:
            # Add start and end tokens
            text = self.START_TOKEN + text.lower() + self.END_TOKEN
            
            # Count bigrams
            for i in range(len(text) - 1):
                char1, char2 = text[i], text[i + 1]
                self.bigram_counts[char1][char2] += 1
                self.char_counts[char1] += 1
    
    def train(self, texts: List[str]):
        """
        Train the bigram model on text data.
        
        Args:
            texts: List of training text strings
        """
        print("Training character-level bigram model...")
        
        # Build vocabulary
        self._build_vocab(texts)
        
        # Count bigrams
        self._count_bigrams(texts)
        
        self.is_trained = True
        print(f"Training complete! Found {sum(sum(counts.values()) for counts in self.bigram_counts.values())} bigrams")
    
    def get_prob(self, char1: str, char2: str) -> float:
        """
        Get probability P(char2 | char1) using Laplace smoothing.
        
        Args:
            char1: Current character
            char2: Next character
            
        Returns:
            Probability of char2 following char1
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Laplace smoothing: (count + α) / (total + α * vocab_size)
        bigram_count = self.bigram_counts[char1][char2]
        total_count = self.char_counts[char1]
        
        prob = (bigram_count + self.smoothing) / (total_count + self.smoothing * self.vocab_size)
        return prob
    
    def get_next_char_probs(self, char: str) -> Dict[str, float]:
        """
        Get probability distribution for next character given current character.
        
        Args:
            char: Current character
            
        Returns:
            Dictionary mapping characters to probabilities
        """
        probs = {}
        for next_char in self.char_to_idx.keys():
            probs[next_char] = self.get_prob(char, next_char)
        return probs
    
    def sample_next_char(self, char: str, temperature: float = 1.0) -> str:
        """
        Sample next character from probability distribution.
        
        Args:
            char: Current character
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Sampled next character
        """
        probs = self.get_next_char_probs(char)
        
        # Apply temperature scaling
        if temperature != 1.0:
            for c in probs:
                probs[c] = probs[c] ** (1.0 / temperature)
        
        # Normalize probabilities
        total = sum(probs.values())
        probs = {c: p / total for c, p in probs.items()}
        
        # Sample from distribution
        rand = random.random()
        cumsum = 0.0
        for char, prob in probs.items():
            cumsum += prob
            if rand <= cumsum:
                return char
        
        # Fallback (shouldn't happen)
        return random.choice(list(probs.keys()))
    
    def generate(self, max_length: int = 100, temperature: float = 1.0, 
                 seed: Optional[str] = None) -> str:
        """
        Generate text using the bigram model.
        
        Args:
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            seed: Optional seed character to start generation
            
        Returns:
            Generated text string
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        if seed is None:
            current_char = self.START_TOKEN
        else:
            current_char = seed.lower()
        
        generated = []
        
        for _ in range(max_length):
            next_char = self.sample_next_char(current_char, temperature)
            
            if next_char == self.END_TOKEN:
                break
            
            if next_char != self.START_TOKEN:  # Don't include start token in output
                generated.append(next_char)
            
            current_char = next_char
        
        return ''.join(generated)
    
    def evaluate_perplexity(self, test_texts: List[str]) -> float:
        """
        Evaluate model perplexity on test data.
        Lower perplexity indicates better model.
        
        Args:
            test_texts: Test text strings
            
        Returns:
            Perplexity score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        total_log_prob = 0.0
        total_chars = 0
        
        for text in test_texts:
            text = self.START_TOKEN + text.lower() + self.END_TOKEN
            
            for i in range(len(text) - 1):
                char1, char2 = text[i], text[i + 1]
                prob = self.get_prob(char1, char2)
                
                if prob > 0:  # Avoid log(0)
                    total_log_prob += math.log(prob)
                    total_chars += 1
        
        if total_chars == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_chars
        perplexity = math.exp(-avg_log_prob)
        return perplexity
    
    def save(self, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'bigram_counts': dict(self.bigram_counts),
            'char_counts': dict(self.char_counts),
            'smoothing': self.smoothing,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.char_to_idx = model_data['char_to_idx']
        self.idx_to_char = model_data['idx_to_char']
        self.vocab_size = model_data['vocab_size']
        self.bigram_counts = defaultdict(Counter, model_data['bigram_counts'])
        self.char_counts = Counter(model_data['char_counts'])
        self.smoothing = model_data['smoothing']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    import math
    
    print("Testing Character-Level Bigram Model...")
    
    # Sample training data
    training_texts = [
        "hello world",
        "this is a simple test",
        "machine learning is fascinating", 
        "neural networks are powerful",
        "building ai from scratch",
        "understanding deep learning",
        "python programming language",
        "artificial intelligence research"
    ]
    
    # Create and train model
    model = CharBigramModel(smoothing=0.01)
    model.train(training_texts)
    
    # Test probability computation
    print(f"\nP('l' | 'e') = {model.get_prob('e', 'l'):.4f}")
    print(f"P('o' | 'l') = {model.get_prob('l', 'o'):.4f}")
    
    # Generate some text
    print("\nGenerated text samples:")
    for i in range(5):
        generated = model.generate(max_length=50, temperature=0.8)
        print(f"Sample {i+1}: {generated}")
    
    # Test with different temperatures
    print(f"\nHigh temperature (random): {model.generate(30, temperature=2.0)}")
    print(f"Low temperature (conservative): {model.generate(30, temperature=0.5)}")
    
    print("\nCharacter-level bigram model implemented successfully! ✅")
    print("This is our first language model - the foundation for more complex models!")
