"""
MindMirror - Training Framework

A generic training framework for our neural networks. This provides
utilities for training loops, data handling, and model evaluation.
"""

import random
import time
import math
from typing import List, Dict, Any, Callable, Optional, Tuple
from collections import defaultdict
import sys
import os

# Add micrograd to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from micrograd import Value


class Trainer:
    """
    Generic trainer for neural networks built with micrograd.
    
    Handles training loops, logging, and basic evaluation metrics.
    """
    
    def __init__(self, model, optimizer, loss_fn: Callable):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimizer (SGD, Adam, etc.)
            loss_fn: Loss function
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.metrics = defaultdict(list)
    
    def train_epoch(self, train_data: List[Any], batch_size: int = 32) -> float:
        """
        Train for one epoch.
        
        Args:
            train_data: Training data
            batch_size: Mini-batch size
            
        Returns:
            Average training loss for the epoch
        """
        # Shuffle training data
        random.shuffle(train_data)
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss for batch
            loss = self.compute_batch_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            epoch_loss += loss.data
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def compute_batch_loss(self, batch: List[Any]) -> Value:
        """
        Compute loss for a batch - to be implemented by specific trainers.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Loss value
        """
        raise NotImplementedError("Subclasses must implement compute_batch_loss")
    
    def evaluate(self, val_data: List[Any]) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        
        Args:
            val_data: Validation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        total_loss = 0.0
        num_samples = 0
        
        for sample in val_data:
            loss = self.compute_sample_loss(sample)
            total_loss += loss.data
            num_samples += 1
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        return {'loss': avg_loss}
    
    def compute_sample_loss(self, sample: Any) -> Value:
        """
        Compute loss for a single sample - to be implemented by specific trainers.
        """
        raise NotImplementedError("Subclasses must implement compute_sample_loss")
    
    def train(self, train_data: List[Any], val_data: Optional[List[Any]] = None,
              epochs: int = 100, batch_size: int = 32, 
              eval_every: int = 10, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_data: Training data
            val_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Mini-batch size
            eval_every: Evaluate every N epochs
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_data, batch_size)
            self.train_losses.append(train_loss)
            
            # Evaluate on validation data
            if val_data and (epoch + 1) % eval_every == 0:
                val_metrics = self.evaluate(val_data)
                val_loss = val_metrics['loss']
                self.val_losses.append(val_loss)
                
                # Store additional metrics
                for metric, value in val_metrics.items():
                    self.metrics[metric].append(value)
            
            # Print progress
            if verbose and (epoch + 1) % eval_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f}")
                
                if val_data:
                    print(f"                    | Val Loss: {val_loss:.4f}")
                
                print(f"                    | Time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        
        # Return training history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            **self.metrics
        }
        
        return history


class LanguageModelTrainer(Trainer):
    """
    Specialized trainer for language models.
    """
    
    def compute_batch_loss(self, batch: List[Tuple[int, int]]) -> Value:
        """
        Compute loss for a batch of (input, target) pairs.
        """
        return self.model.compute_loss(batch)
    
    def compute_sample_loss(self, sample: Tuple[int, int]) -> Value:
        """
        Compute loss for a single (input, target) pair.
        """
        return self.model.compute_loss([sample])
    
    def compute_perplexity(self, data: List[Tuple[int, int]]) -> float:
        """
        Compute perplexity on data.
        
        Args:
            data: List of (input, target) pairs
            
        Returns:
            Perplexity score
        """
        total_log_prob = 0.0
        num_samples = len(data)
        
        for input_idx, target_idx in data:
            # Get model probabilities
            logits = self.model.forward(input_idx)
            
            # Convert to probabilities using softmax
            exp_logits = [math.exp(logit.data) for logit in logits]
            sum_exp = sum(exp_logits)
            probs = [exp_logit / sum_exp for exp_logit in exp_logits]
            
            # Get probability of target
            target_prob = probs[target_idx]
            
            if target_prob > 0:
                total_log_prob += math.log(target_prob)
        
        avg_log_prob = total_log_prob / num_samples
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def evaluate(self, val_data: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        Evaluate language model with loss and perplexity.
        """
        # Compute loss
        metrics = super().evaluate(val_data)
        
        # Compute perplexity
        try:
            perplexity = self.compute_perplexity(val_data)
            metrics['perplexity'] = perplexity
        except:
            metrics['perplexity'] = float('inf')
        
        return metrics


def split_data(data: List[Any], train_ratio: float = 0.8, 
               val_ratio: float = 0.1) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (remainder goes to test)
        
    Returns:
        (train_data, val_data, test_data)
    """
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Training history from trainer
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
        if 'val_loss' in history:
            # Validation is evaluated less frequently, so we need to interpolate x-axis
            val_epochs = list(range(0, len(history['train_loss']), 
                                  len(history['train_loss']) // len(history['val_loss'])))
            axes[0].plot(val_epochs, history['val_loss'], label='Val Loss', alpha=0.7)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot perplexity if available
        if 'perplexity' in history:
            axes[1].plot(val_epochs, history['perplexity'], label='Perplexity', alpha=0.7)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Perplexity')
            axes[1].set_title('Validation Perplexity')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")


# Example usage
if __name__ == "__main__":
    print("Training framework implemented successfully! ✅")
    print("This provides the foundation for training all our neural networks.")
    print("\nKey features:")
    print("- Generic trainer with mini-batch support")
    print("- Language model specific trainer with perplexity")
    print("- Training history tracking and visualization")
    print("- Data splitting utilities")
