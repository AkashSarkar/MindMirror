"""
MindMirror AI - Core Components with TensorFlow Optimization

This module contains the main AI components optimized for performance
while maintaining educational clarity.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Try TensorFlow first, fallback to educational implementation
try:
    import tensorflow as tf
    import numpy as np
    HAS_TENSORFLOW = True
    
    # Enable Metal GPU acceleration on Mac M2
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
    except:
        pass
        
except ImportError:
    HAS_TENSORFLOW = False
    # Fallback imports
    from lib.micrograd.engine import Value
    from lib.micrograd.nn import MLP

from typing import List, Dict, Optional
import random
import time


class TextProcessor:
    """Handles text preprocessing and tokenization"""
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
    
    def build_vocabulary(self, text: str) -> None:
        """Build character-level vocabulary from text"""
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.reverse_vocab = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        print(f"📚 Built vocabulary with {self.vocab_size} characters")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to list of indices"""
        return [self.vocab.get(ch, 0) for ch in text]
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        return ''.join([self.reverse_vocab.get(idx, '?') for idx in indices])


if HAS_TENSORFLOW:
    class LanguageModel(tf.keras.Model):
        """High-performance language model using TensorFlow"""
        
        def __init__(self, vocab_size: int, context_length: int = 32, 
                     embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
            super().__init__()
            
            self.vocab_size = vocab_size
            self.context_length = context_length
            
            # Embedding layer
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            
            # LSTM layers
            self.lstm_layers = []
            for i in range(num_layers):
                return_sequences = i < num_layers - 1
                self.lstm_layers.append(
                    tf.keras.layers.LSTM(
                        hidden_dim, 
                        return_sequences=return_sequences,
                        dropout=0.1
                    )
                )
            
            # Output layers
            self.dense = tf.keras.layers.Dense(hidden_dim, activation='relu')
            self.dropout = tf.keras.layers.Dropout(0.2)
            self.output_layer = tf.keras.layers.Dense(vocab_size)
        
        def call(self, inputs, training=None):
            x = self.embedding(inputs)
            
            for lstm_layer in self.lstm_layers:
                x = lstm_layer(x, training=training)
            
            x = self.dense(x)
            x = self.dropout(x, training=training)
            return self.output_layer(x)

else:
    # Fallback to educational implementation
    class LanguageModel:
        """Fallback language model using educational implementation"""
        
        def __init__(self, vocab_size: int, context_length: int = 8, hidden_size: int = 64):
            self.vocab_size = vocab_size
            self.context_length = context_length
            self.network = MLP(context_length, [hidden_size, hidden_size, vocab_size])
            print("📚 Using educational implementation (install TensorFlow for better performance)")
        
        def forward(self, context: List[int]) -> List[Value]:
            if len(context) < self.context_length:
                context = [0] * (self.context_length - len(context)) + context
            elif len(context) > self.context_length:
                context = context[-self.context_length:]
            
            inputs = [Value(float(idx)) for idx in context]
            outputs = self.network(inputs)
            return outputs if isinstance(outputs, list) else [outputs]


class MindMirrorAI:
    """Main AI class with automatic optimization detection"""
    
    def __init__(self, context_length: int = None, hidden_size: int = None):
        # Set defaults based on available libraries
        if HAS_TENSORFLOW:
            self.context_length = context_length or 32
            self.hidden_size = hidden_size or 256
            print("🚀 MindMirror AI initialized with TensorFlow optimization")
            print(f"   GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        else:
            self.context_length = context_length or 8
            self.hidden_size = hidden_size or 64
            print("📚 MindMirror AI initialized with educational implementation")
        
        self.processor = TextProcessor()
        self.model = None
        self.training_data = []
    
    def learn_from_text(self, text: str) -> None:
        """Learn patterns from text"""
        print(f"\n📖 Learning from text ({len(text)} characters)...")
        
        self.processor.build_vocabulary(text)
        
        if HAS_TENSORFLOW:
            self.model = LanguageModel(
                vocab_size=self.processor.vocab_size,
                context_length=self.context_length,
                embedding_dim=self.hidden_size//2,
                hidden_dim=self.hidden_size,
                num_layers=2
            )
            
            # Prepare training data for TensorFlow
            encoded_text = self.processor.encode(text)
            sequences, targets = [], []
            
            for i in range(0, len(encoded_text) - self.context_length, self.context_length//4):
                if i + self.context_length < len(encoded_text):
                    seq = encoded_text[i:i + self.context_length]
                    target = encoded_text[i + 1:i + self.context_length + 1]
                    sequences.append(seq)
                    targets.append(target[-1])
            
            self.X = np.array(sequences)
            self.y = tf.keras.utils.to_categorical(targets, self.processor.vocab_size)
            
        else:
            self.model = LanguageModel(
                vocab_size=self.processor.vocab_size,
                context_length=self.context_length,
                hidden_size=self.hidden_size
            )
            
            # Prepare training data for educational version
            encoded_text = self.processor.encode(text)
            for i in range(len(encoded_text) - self.context_length):
                context = encoded_text[i:i + self.context_length]
                target = encoded_text[i + self.context_length]
                self.training_data.append((context, target))
        
        print(f"📝 Prepared training data")
    
    def train(self, epochs: int = None, learning_rate: float = 0.001) -> None:
        """Train the model"""
        if self.model is None:
            print("❌ No model created. Call learn_from_text() first.")
            return
        
        # Set appropriate defaults
        if epochs is None:
            epochs = 20 if HAS_TENSORFLOW else 50
        
        print(f"\n🏃‍♂️ Training for {epochs} epochs...")
        start_time = time.time()
        
        if HAS_TENSORFLOW:
            # TensorFlow training
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5),
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            ]
            
            history = self.model.fit(
                self.X, self.y,
                epochs=epochs,
                batch_size=16,
                callbacks=callbacks,
                verbose=1
            )
            
        else:
            # Educational training
            for epoch in range(epochs):
                total_loss = Value(0.0)
                correct_predictions = 0
                
                random.shuffle(self.training_data)
                
                for context, target in self.training_data:
                    outputs = self.model.forward(context)
                    
                    predicted = 0
                    max_output = outputs[0].data if outputs else 0
                    for i, output in enumerate(outputs):
                        if output.data > max_output:
                            max_output = output.data
                            predicted = i
                    
                    if predicted == target:
                        correct_predictions += 1
                    
                    if target < len(outputs):
                        target_output = Value(1.0) if predicted == target else Value(-1.0)
                        prediction = outputs[target]
                        loss = (prediction - target_output) ** 2
                        total_loss = total_loss + loss
                        loss.backward()
                
                # Update parameters
                for param in self.model.network.parameters():
                    if param.grad > 5:
                        param.grad = 5
                    elif param.grad < -5:
                        param.grad = -5
                    param.data -= learning_rate * param.grad
                    param.grad = 0.0
                
                if epoch % (epochs // 5) == 0:
                    accuracy = correct_predictions / len(self.training_data) * 100
                    avg_loss = total_loss.data / len(self.training_data)
                    print(f"   Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.1f}%")
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds!")
    
    def generate(self, prompt: str = "", max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text using the trained model"""
        if self.model is None:
            return "❌ Model not trained yet."
        
        if prompt:
            context = self.processor.encode(prompt)
        else:
            context = [0] * self.context_length
        
        if len(context) > self.context_length:
            context = context[-self.context_length:]
        elif len(context) < self.context_length:
            context = [0] * (self.context_length - len(context)) + context
        
        generated = prompt
        
        for _ in range(max_length):
            if HAS_TENSORFLOW:
                input_seq = np.array([context])
                logits = self.model(input_seq, training=False).numpy()[0]
                
                if temperature == 0:
                    next_id = np.argmax(logits)
                else:
                    logits = logits / temperature
                    probabilities = tf.nn.softmax(logits).numpy()
                    next_id = np.random.choice(len(probabilities), p=probabilities)
            else:
                outputs = self.model.forward(context)
                
                if temperature == 0:
                    next_id = 0
                    max_output = outputs[0].data if outputs else 0
                    for i, output in enumerate(outputs):
                        if output.data > max_output:
                            max_output = output.data
                            next_id = i
                else:
                    # Simple random choice with noise
                    scores = []
                    for output in outputs:
                        noise = random.uniform(-temperature, temperature)
                        scores.append(output.data + noise)
                    next_id = scores.index(max(scores))
            
            next_char = self.processor.reverse_vocab.get(next_id, '?')
            generated += next_char
            context = context[1:] + [next_id]
        
        return generated
    
    def chat_response(self, user_input: str) -> str:
        """Generate a response to user input"""
        context_prompt = user_input[-self.context_length:]
        response = self.generate(prompt=context_prompt, max_length=100, temperature=0.7)
        return response[len(context_prompt):]
