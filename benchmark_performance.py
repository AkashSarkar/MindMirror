"""
Performance Comparison: Educational vs Optimized Implementation

This script demonstrates the massive performance difference between
the educational implementation and the optimized TensorFlow version.
"""

import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_educational():
    """Test the educational implementation"""
    print("📚 Testing Educational Implementation...")
    
    try:
        from lib.micrograd.engine import Value
        from lib.micrograd.nn import MLP
        
        # Create small network
        network = MLP(8, [32, 16, 10])
        
        # Simple training simulation
        start_time = time.time()
        
        for epoch in range(10):
            # Simulate training on 100 samples
            for _ in range(100):
                inputs = [Value(0.1) for _ in range(8)]
                outputs = network(inputs)
                
                # Simple loss
                if isinstance(outputs, list):
                    loss = sum(out ** 2 for out in outputs)
                else:
                    loss = outputs ** 2
                
                loss.backward()
                
                # Update parameters
                for param in network.parameters():
                    param.data -= 0.01 * param.grad
                    param.grad = 0.0
        
        training_time = time.time() - start_time
        print(f"   ✅ Completed in {training_time:.2f} seconds")
        print(f"   📊 Parameters: {len(network.parameters())}")
        return training_time
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def test_optimized():
    """Test the optimized TensorFlow implementation"""
    print("🚀 Testing Optimized Implementation...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Create equivalent network
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='tanh', input_shape=(8,)),
            tf.keras.layers.Dense(16, activation='tanh'),
            tf.keras.layers.Dense(10)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Generate training data
        X = np.random.randn(1000, 8)
        y = np.random.randn(1000, 10)
        
        start_time = time.time()
        
        # Train
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        training_time = time.time() - start_time
        print(f"   ✅ Completed in {training_time:.2f} seconds")
        print(f"   📊 Parameters: {model.count_params()}")
        print(f"   🚀 GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        return training_time
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def main():
    print("⚡ MindMirror AI Performance Comparison")
    print("=" * 50)
    print("Testing equivalent networks with similar parameter counts\n")
    
    # Test educational implementation
    edu_time = test_educational()
    print()
    
    # Test optimized implementation
    opt_time = test_optimized()
    print()
    
    # Compare results
    if edu_time and opt_time:
        speedup = edu_time / opt_time
        print("📈 PERFORMANCE COMPARISON:")
        print(f"   Educational: {edu_time:.2f} seconds")
        print(f"   Optimized:   {opt_time:.2f} seconds")
        print(f"   Speedup:     {speedup:.1f}x faster! 🚀")
        
        if speedup > 10:
            print("\n💡 The optimized version is MUCH faster!")
            print("   This is why we use TensorFlow for production AI.")
        elif speedup > 2:
            print("\n💡 The optimized version shows good improvement!")
        else:
            print("\n🤔 Results may vary based on your hardware.")
    
    print("\n🎯 RECOMMENDATION:")
    print("   📚 Use examples/ for learning concepts")
    print("   🚀 Use src/main_optimized.py for real AI applications")


if __name__ == "__main__":
    main()
