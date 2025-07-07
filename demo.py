#!/usr/bin/env python3
"""
MindMirror - Complete Demonstration Script

This script demonstrates the entire MindMirror project:
1. Micrograd autograd engine
2. Neural network layers  
3. Character-level language models
4. Text generation and evaluation

Run this to see everything working together!
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from micrograd import Value, MLP, mse_loss, Adam
from models.bigram import CharBigramModel
from models.neural_bigram import NeuralBigramModel
from training.utils import create_sample_dataset, analyze_text_statistics, print_text_statistics


def demo_micrograd():
    """Demonstrate the micrograd autograd engine."""
    print("🔧 MICROGRAD AUTOGRAD ENGINE DEMO")
    print("=" * 50)
    
    # Basic operations
    print("1. Basic automatic differentiation:")
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b') 
    c = Value(10.0, label='c')
    
    # Forward pass: f = (a*b + c).tanh()
    d = a * b  # d = -6
    e = d + c  # e = 4
    f = e.tanh()  # f = tanh(4) ≈ 0.999
    
    print(f"   a = {a.data}")
    print(f"   b = {b.data}")
    print(f"   c = {c.data}")
    print(f"   f = (a*b + c).tanh() = {f.data:.6f}")
    
    # Backward pass
    f.backward()
    
    print(f"\n2. Computed gradients:")
    print(f"   df/da = {a.grad:.6f}")
    print(f"   df/db = {b.grad:.6f}")
    print(f"   df/dc = {c.grad:.6f}")
    
    # Neural network demonstration
    print(f"\n3. Simple neural network:")
    model = MLP(3, [4, 2, 1])  # 3 inputs -> 4 -> 2 -> 1 output
    print(f"   Model has {len(model.parameters())} parameters")
    
    # Forward pass
    x = [Value(1.0), Value(2.0), Value(-1.0)]
    y_pred = model(x)
    print(f"   Input: [1.0, 2.0, -1.0]")
    print(f"   Output: {y_pred.data:.6f}")
    
    # Training step
    y_target = [0.5]
    loss = mse_loss(y_pred, y_target)
    print(f"   Target: {y_target[0]}")
    print(f"   Loss: {loss.data:.6f}")
    
    model.zero_grad()
    loss.backward()
    
    # Show some gradients
    grads = [p.grad for p in model.parameters()[:5]]
    print(f"   Sample gradients: {[f'{g:.4f}' for g in grads]}")
    
    print("   ✅ Micrograd engine working perfectly!\n")


def demo_statistical_bigram():
    """Demonstrate statistical bigram model."""
    print("📊 STATISTICAL BIGRAM MODEL DEMO")
    print("=" * 50)
    
    # Create model and training data
    model = CharBigramModel(smoothing=0.01)
    texts = create_sample_dataset()
    
    print(f"1. Training on {len(texts)} text samples...")
    model.train(texts[:5])  # Use subset for demo
    
    print(f"\n2. Model statistics:")
    print(f"   Vocabulary size: {model.vocab_size}")
    print(f"   Total bigrams: {sum(sum(counts.values()) for counts in model.bigram_counts.values())}")
    
    print(f"\n3. Character probabilities after 'th':")
    probs = model.get_next_char_probs('h')
    top_chars = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for char, prob in top_chars:
        char_repr = 'space' if char == ' ' else char
        print(f"   P('{char_repr}' | 'h') = {prob:.4f}")
    
    print(f"\n4. Generated text samples:")
    for i in range(3):
        generated = model.generate(max_length=40, temperature=0.8)
        print(f"   Sample {i+1}: {generated}")
    
    print("   ✅ Statistical bigram model working!\n")


def demo_neural_bigram():
    """Demonstrate neural bigram model."""
    print("🧠 NEURAL BIGRAM MODEL DEMO") 
    print("=" * 50)
    
    # Create model and training data
    model = NeuralBigramModel(embedding_dim=8, hidden_dim=16)
    texts = create_sample_dataset()
    
    print(f"1. Training neural bigram model...")
    model.train(texts[:10], epochs=30, lr=0.1, verbose=False)
    
    print(f"\n2. Model architecture:")
    print(f"   Embedding dimension: {model.embedding_dim}")
    print(f"   Hidden dimension: {model.hidden_dim}")
    print(f"   Vocabulary size: {model.vocab_size}")
    print(f"   Total parameters: {len(model.get_parameters())}")
    
    print(f"\n3. Character probabilities after 'th':")
    probs = model.get_next_char_probs('h')
    top_chars = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for char, prob in top_chars:
        char_repr = 'space' if char == ' ' else char
        print(f"   P('{char_repr}' | 'h') = {prob:.4f}")
    
    print(f"\n4. Generated text samples:")
    for i in range(3):
        generated = model.generate(max_length=50, temperature=0.8)
        print(f"   Sample {i+1}: {generated}")
    
    print(f"\n5. Different temperature sampling:")
    print(f"   Conservative (T=0.3): {model.generate(30, temperature=0.3)}")
    print(f"   Balanced (T=1.0): {model.generate(30, temperature=1.0)}")
    print(f"   Creative (T=1.5): {model.generate(30, temperature=1.5)}")
    
    print("   ✅ Neural bigram model working!\n")


def demo_data_analysis():
    """Demonstrate data analysis utilities."""
    print("📈 DATA ANALYSIS DEMO")
    print("=" * 50)
    
    texts = create_sample_dataset()
    stats = analyze_text_statistics(texts)
    
    print_text_statistics(stats)
    print("   ✅ Data analysis working!\n")


def main():
    """Run the complete MindMirror demonstration."""
    print("🎯 MINDMIRROR PROJECT DEMONSTRATION")
    print("Building a Personal AI Assistant from Scratch")
    print("=" * 60)
    print()
    
    try:
        # Run all demonstrations
        demo_micrograd()
        demo_statistical_bigram()
        demo_neural_bigram()
        demo_data_analysis()
        
        # Summary
        print("🎉 DEMONSTRATION COMPLETE!")
        print("=" * 40)
        print("✅ Micrograd autograd engine - Working")
        print("✅ Neural network layers - Working")
        print("✅ Statistical language model - Working") 
        print("✅ Neural language model - Working")
        print("✅ Text generation - Working")
        print("✅ Data analysis utilities - Working")
        print()
        print("🚀 Next steps:")
        print("   1. Implement RNN architecture")
        print("   2. Build attention mechanisms")
        print("   3. Create transformer model")
        print("   4. Develop conversational interface")
        print()
        print("💡 The foundation is solid! Ready to build more advanced models.")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
