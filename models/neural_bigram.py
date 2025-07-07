"""
MindMirror - Neural Bigram Model

This implements a neural network version of the bigram model using our 
micrograd engine. Instead of counting frequencies, we learn character 
embeddings and predict the next character using a neural network.

This bridges the gap between statistical models and neural networks.
"""

import random
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter
import sys
import os

# Add micrograd to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from micrograd import Value, MLP, cross_entropy_loss, Adam


class NeuralBigramModel:
    """
    A neural network bigram model that learns character embeddings.
    
    Architecture:
    - Character embedding layer (maps characters to dense vectors)
    - Small neural network to predict next character probabilities
    - Trained using cross-entropy loss and backpropagation
    """
    
    def __init__(self, embedding_dim: int = 8, hidden_dim: int = 16):
        """
        Initialize neural bigram model.
        
        Args:
            embedding_dim: Dimension of character embeddings
            hidden_dim: Hidden layer dimension
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Vocabulary
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        # Model components
        self.embeddings = {}  # char_idx -> embedding vector (list of Values)
        self.network = None   # MLP to predict next character
        
        # Special tokens
        self.START_TOKEN = '^'  # Start token
        self.END_TOKEN = '$'    # End token
        
        self.is_trained = False
    
    def _build_vocab(self, texts: List[str]):
        """Build character vocabulary from texts."""
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
    
    def _initialize_model(self):
        """Initialize embeddings and neural network."""
        # Initialize character embeddings randomly
        self.embeddings = {}
        for char_idx in range(self.vocab_size):
            embedding = [Value(random.uniform(-0.5, 0.5)) for _ in range(self.embedding_dim)]
            self.embeddings[char_idx] = embedding
        
        # Create neural network: embedding -> hidden -> vocab_size
        self.network = MLP(
            nin=self.embedding_dim,
            nouts=[self.hidden_dim, self.vocab_size],
            activations=['tanh', 'linear']  # Linear output for logits
        )
        
        print(f"Initialized model with {len(self.get_parameters())} parameters")
    
    def get_parameters(self) -> List[Value]:
        """Get all trainable parameters."""
        params = []
        
        # Embedding parameters
        for embedding in self.embeddings.values():
            params.extend(embedding)
        
        # Network parameters
        if self.network:
            params.extend(self.network.parameters())
        
        return params
    
    def get_embedding(self, char_idx: int) -> List[Value]:
        """Get embedding for a character index."""
        return self.embeddings[char_idx]
    
    def forward(self, char_idx: int) -> List[Value]:
        """
        Forward pass: character -> embedding -> network -> logits
        
        Args:
            char_idx: Input character index
            
        Returns:
            Logits for next character (before softmax)
        """
        # Get character embedding
        embedding = self.get_embedding(char_idx)
        
        # Pass through network
        logits = self.network(embedding)
        
        return logits if isinstance(logits, list) else [logits]
    
    def compute_loss(self, char_pairs: List[Tuple[int, int]]) -> Value:
        """
        Compute cross-entropy loss for character pairs.
        
        Args:
            char_pairs: List of (current_char_idx, next_char_idx) pairs
            
        Returns:
            Average cross-entropy loss
        """
        total_loss = Value(0.0)
        
        for curr_idx, next_idx in char_pairs:
            # Forward pass
            logits = self.forward(curr_idx)
            
            # Compute softmax probabilities manually
            exp_logits = [logit.exp() for logit in logits]
            sum_exp = sum(exp_logits, Value(0.0))
            probs = [exp_logit / sum_exp for exp_logit in exp_logits]
            
            # Cross-entropy loss: -log(prob_correct)
            correct_prob = probs[next_idx]
            loss = -correct_prob.log()
            total_loss = total_loss + loss
        
        # Return average loss
        return total_loss * (1.0 / len(char_pairs))
    
    def prepare_training_data(self, texts: List[str]) -> List[Tuple[int, int]]:
        """
        Prepare character pairs for training.
        
        Args:
            texts: Training text strings
            
        Returns:
            List of (current_char_idx, next_char_idx) pairs
        """
        char_pairs = []
        
        for text in texts:
            # Add start and end tokens
            text = self.START_TOKEN + text.lower() + self.END_TOKEN
            
            # Create character pairs
            for i in range(len(text) - 1):
                curr_char = text[i]
                next_char = text[i + 1]
                
                curr_idx = self.char_to_idx[curr_char]
                next_idx = self.char_to_idx[next_char]
                
                char_pairs.append((curr_idx, next_idx))
        
        return char_pairs
    
    def train(self, texts: List[str], epochs: int = 100, lr: float = 0.1, 
              batch_size: int = 32, verbose: bool = True):
        """
        Train the neural bigram model.
        
        Args:
            texts: Training text strings
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Mini-batch size
            verbose: Print training progress
        """
        print("Training Neural Bigram Model...")
        
        # Build vocabulary and initialize model
        self._build_vocab(texts)
        self._initialize_model()
        
        # Prepare training data
        char_pairs = self.prepare_training_data(texts)
        print(f"Training on {len(char_pairs)} character pairs")
        
        # Initialize optimizer
        optimizer = Adam(self.get_parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle training data
            random.shuffle(char_pairs)
            
            epoch_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(char_pairs), batch_size):
                batch = char_pairs[i:i + batch_size]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Compute loss
                loss = self.compute_loss(batch)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                epoch_loss += loss.data
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        print("Training completed!")
    
    def get_next_char_probs(self, char: str) -> Dict[str, float]:
        """
        Get probability distribution for next character.
        
        Args:
            char: Current character
            
        Returns:
            Dictionary mapping characters to probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        char_idx = self.char_to_idx.get(char.lower())
        if char_idx is None:
            # Unknown character - return uniform distribution
            uniform_prob = 1.0 / self.vocab_size
            return {c: uniform_prob for c in self.char_to_idx.keys()}
        
        # Forward pass
        logits = self.forward(char_idx)
        
        # Convert to probabilities using softmax
        exp_logits = [math.exp(logit.data) for logit in logits]
        sum_exp = sum(exp_logits)
        probs = [exp_logit / sum_exp for exp_logit in exp_logits]
        
        # Create character -> probability mapping
        char_probs = {}
        for idx, prob in enumerate(probs):
            char = self.idx_to_char[idx]
            char_probs[char] = prob
        
        return char_probs
    
    def sample_next_char(self, char: str, temperature: float = 1.0) -> str:
        """
        Sample next character from model distribution.
        
        Args:
            char: Current character
            temperature: Sampling temperature
            
        Returns:
            Sampled next character
        """
        probs = self.get_next_char_probs(char)
        
        # Apply temperature scaling
        if temperature != 1.0:
            for c in probs:
                probs[c] = probs[c] ** (1.0 / temperature)
        
        # Normalize
        total = sum(probs.values())
        probs = {c: p / total for c, p in probs.items()}
        
        # Sample
        rand = random.random()
        cumsum = 0.0
        for char, prob in probs.items():
            cumsum += prob
            if rand <= cumsum:
                return char
        
        return random.choice(list(probs.keys()))
    
    def generate(self, max_length: int = 100, temperature: float = 1.0, 
                 seed: Optional[str] = None) -> str:
        """
        Generate text using the neural bigram model.
        
        Args:
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            seed: Optional seed character
            
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
            
            if next_char != self.START_TOKEN:
                generated.append(next_char)
            
            current_char = next_char
        
        return ''.join(generated)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Bigram Model...")
    
    # Sample training data
    training_texts = [
        "hello world",
        "this is a simple test",
        "machine learning is fascinating",
        "neural networks are powerful", 
        "building ai from scratch",
        "understanding deep learning",
        "python programming language",
        "artificial intelligence research",
        "backpropagation algorithm",
        "gradient descent optimization"
    ]
    
    # Create and train model
    model = NeuralBigramModel(embedding_dim=8, hidden_dim=16)
    model.train(training_texts, epochs=50, lr=0.1, batch_size=16)
    
    print(f"\nModel has {len(model.get_parameters())} trainable parameters")
    
    # Generate some text
    print("\nGenerated text samples:")
    for i in range(5):
        generated = model.generate(max_length=30, temperature=0.8)
        print(f"Sample {i+1}: {generated}")
    
    # Test different temperatures
    print(f"\nHigh temperature (random): {model.generate(25, temperature=2.0)}")
    print(f"Low temperature (conservative): {model.generate(25, temperature=0.3)}")
    
    # Show character probabilities for specific contexts
    print(f"\nNext char probabilities after 'th':")
    probs = model.get_next_char_probs('h')
    top_chars = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for char, prob in top_chars:
        print(f"  '{char}': {prob:.3f}")
    
    print("\nNeural bigram model implemented successfully! ✅")
    print("This demonstrates how neural networks can learn language patterns!")
