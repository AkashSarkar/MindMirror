import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.micrograd.engine import Value
from lib.micrograd.nn import MLP
import random


def build_character_level_text_generator():
    """
    Build a neural network that learns to generate text character by character!
    This is the foundation of how ChatGPT works - just at a tiny scale.
    """
    print("=== Character-Level Text Generator ===\n")
    
    # Sample text to learn from
    text_data = """hello world this is amazing neural networks can learn to generate text character by character"""
    
    print("📚 TRAINING TEXT:")
    print(f"   '{text_data}'")
    print(f"   Length: {len(text_data)} characters")
    print()
    
    # Create character vocabulary
    chars = sorted(list(set(text_data)))
    vocab_size = len(chars)
    
    print("🔤 CHARACTER VOCABULARY:")
    print(f"   Characters: {chars}")
    print(f"   Vocabulary size: {vocab_size}")
    print()
    
    # Create mappings between characters and indices
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print("📊 CHARACTER ENCODING:")
    for i, char in enumerate(chars[:10]):  # Show first 10
        print(f"   '{char}' → {i}")
    if len(chars) > 10:
        print("   ...")
    print()
    
    # Create training data: sequences of characters
    sequence_length = 3  # Use 3 characters to predict the next one
    X, Y = [], []
    
    for i in range(len(text_data) - sequence_length):
        # Input: sequence of 3 characters
        input_sequence = text_data[i:i + sequence_length]
        # Target: next character
        target_char = text_data[i + sequence_length]
        
        # Convert to indices
        input_indices = [char_to_idx[ch] for ch in input_sequence]
        target_index = char_to_idx[target_char]
        
        X.append(input_indices)
        Y.append(target_index)
    
    print("📝 TRAINING EXAMPLES:")
    print("   Input sequence → Target character")
    for i in range(min(10, len(X))):
        input_chars = [idx_to_char[idx] for idx in X[i]]
        target_char = idx_to_char[Y[i]]
        print(f"   {input_chars} → '{target_char}'")
    print(f"   Total training examples: {len(X)}")
    print()
    
    return text_data, chars, char_to_idx, idx_to_char, X, Y, sequence_length


def create_and_train_network(X, Y, chars, char_to_idx, idx_to_char, sequence_length):
    """
    Create and train a neural network for text generation
    """
    vocab_size = len(chars)
    
    print("🏗️ NETWORK ARCHITECTURE:")
    print(f"   Input: {sequence_length} characters (each encoded as index)")
    print(f"   Hidden Layer 1: {vocab_size * 2} neurons")
    print(f"   Hidden Layer 2: {vocab_size} neurons")
    print(f"   Output: {vocab_size} neurons (probability for each character)")
    print()
    
    # Create the network
    # Input: sequence_length integers, Output: vocab_size probabilities
    network = MLP(sequence_length, [vocab_size * 2, vocab_size, vocab_size])
    
    print(f"   Total parameters: {len(network.parameters())}")
    print()
    
    print("🎯 TRAINING PROCESS:")
    print("   The network learns: given 3 characters, what's the next one?")
    print()
    
    # Training loop
    learning_rate = 0.01
    epochs = 100
    
    print(f"🏃‍♂️ TRAINING (learning rate = {learning_rate}, epochs = {epochs}):")
    
    for epoch in range(epochs):
        total_loss = Value(0.0)
        correct_predictions = 0
        
        # Shuffle training data
        training_pairs = list(zip(X, Y))
        random.shuffle(training_pairs)
        
        for input_seq, target_idx in training_pairs:
            # Convert input to Values
            inputs = [Value(float(idx)) for idx in input_seq]
            
            # Forward pass
            outputs = network(inputs)
            if not isinstance(outputs, list):
                outputs = [outputs]
            
            # Find predicted character (highest output)
            predicted_idx = 0
            max_output = outputs[0].data
            for i, output in enumerate(outputs):
                if output.data > max_output:
                    max_output = output.data
                    predicted_idx = i
            
            # Check if prediction is correct
            if predicted_idx == target_idx:
                correct_predictions += 1
            
            # Loss: try to maximize the output for the correct character
            # We'll use a simple approach: minimize distance to target
            target_output = Value(1.0) if predicted_idx == target_idx else Value(-1.0)
            
            # Use the output corresponding to target character
            if target_idx < len(outputs):
                prediction = outputs[target_idx]
                loss = (prediction - target_output) ** 2
                total_loss = total_loss + loss
                
                # Backward pass
                loss.backward()
        
        # Update parameters
        for param in network.parameters():
            # Gradient clipping
            if param.grad > 5:
                param.grad = 5
            elif param.grad < -5:
                param.grad = -5
            
            param.data -= learning_rate * param.grad
            param.grad = 0.0
        
        # Print progress
        if epoch % 20 == 0:
            accuracy = correct_predictions / len(X) * 100
            avg_loss = total_loss.data / len(X)
            print(f"   Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.1f}%")
    
    return network


def generate_text(network, chars, char_to_idx, idx_to_char, sequence_length, seed_text, length=50):
    """
    Generate text using the trained network
    """
    print(f"\n📖 GENERATING TEXT:")
    print(f"   Seed: '{seed_text}'")
    print(f"   Length: {length} characters")
    print()
    
    generated = seed_text
    current_sequence = seed_text[-sequence_length:]  # Last 3 characters
    
    for _ in range(length):
        # Convert current sequence to indices
        try:
            input_indices = [char_to_idx[ch] for ch in current_sequence]
        except KeyError:
            # If character not in vocabulary, use space
            input_indices = [char_to_idx.get(ch, char_to_idx[' ']) for ch in current_sequence]
        
        # Get network prediction
        inputs = [Value(float(idx)) for idx in input_indices]
        outputs = network(inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        # Find character with highest probability
        best_idx = 0
        best_score = outputs[0].data if outputs else 0
        for i, output in enumerate(outputs):
            if output.data > best_score:
                best_score = output.data
                best_idx = i
        
        # Add some randomness for more interesting text
        if random.random() < 0.3:  # 30% chance of random choice
            best_idx = random.randint(0, len(chars) - 1)
        
        # Generate next character
        if best_idx < len(idx_to_char):
            next_char = idx_to_char[best_idx]
        else:
            next_char = ' '  # Default to space
        
        generated += next_char
        
        # Update sequence (sliding window)
        current_sequence = current_sequence[1:] + next_char
    
    return generated


def demonstrate_learning_process():
    """
    Show how the network learns patterns in text
    """
    print("=" * 60)
    print("🧠 WHAT THE NETWORK IS LEARNING")
    print("=" * 60)
    print()
    
    print("🔍 PATTERN RECOGNITION:")
    print("   'hel' → 'l'  (hello)")
    print("   'ell' → 'o'  (hello)")
    print("   'wor' → 'l'  (world)")
    print("   'orl' → 'd'  (world)")
    print("   'thi' → 's'  (this)")
    print()
    
    print("🧮 HOW IT WORKS:")
    print("   1. Network sees 3 characters as input")
    print("   2. Each character becomes a number (encoding)")
    print("   3. Network processes through hidden layers")
    print("   4. Output layer has one neuron per possible character")
    print("   5. Highest output = most likely next character")
    print()
    
    print("🎯 THIS IS THE FOUNDATION OF:")
    print("   ✅ Auto-complete")
    print("   ✅ Text prediction on phones")
    print("   ✅ ChatGPT (but with billions of parameters!)")
    print("   ✅ Code completion in IDEs")
    print()
    
    print("🚀 SCALING UP:")
    print("   - More text → better patterns")
    print("   - Longer sequences → better context")
    print("   - More layers → more complex understanding")
    print("   - Attention mechanism → focus on important parts")


if __name__ == "__main__":
    # Step 1: Prepare data
    text_data, chars, char_to_idx, idx_to_char, X, Y, sequence_length = build_character_level_text_generator()
    
    # Step 2: Train network
    network = create_and_train_network(X, Y, chars, char_to_idx, idx_to_char, sequence_length)
    
    # Step 3: Generate text
    print("\n🎨 TEXT GENERATION EXAMPLES:")
    
    # Test with different seed texts
    seeds = ["hel", "wor", "thi", "net"]
    
    for seed in seeds:
        if all(c in char_to_idx for c in seed):
            generated = generate_text(network, chars, char_to_idx, idx_to_char, sequence_length, seed, 30)
            print(f"   Seed '{seed}' → '{generated}'")
    
    # Step 4: Explain what happened
    demonstrate_learning_process()
    
    print("\n💡 CONGRATULATIONS!")
    print("   You just built a neural network that generates text!")
    print("   This is the same principle behind ChatGPT - just much smaller.")
    print("   Next: we'll add attention mechanisms to make it even smarter! 🤖")
