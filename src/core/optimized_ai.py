"""
Optimized MindMirror AI using TensorFlow with Metal GPU acceleration for Mac M2

This version uses TensorFlow for performance while maintaining the educational
structure of the project. The examples still use your custom implementation
for learning, but the production AI uses optimized libraries.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional
import random
import time

# Enable Metal GPU acceleration on Mac M2
try:
    # Check for Metal GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU acceleration enabled: {len(gpus)} GPU(s) detected")
    else:
        print("⚠️  No GPU detected, using CPU with optimizations")
except Exception as e:
    print(f"⚠️  GPU setup error: {e}, falling back to CPU")

# Configure TensorFlow for M2 optimization
tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores


class OptimizedTextProcessor:
    """Fast text processing optimized for neural networks"""
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        self.char_to_id = None
        self.id_to_char = None
    
    def build_vocabulary(self, text: str) -> None:
        """Build vocabulary with TensorFlow lookup tables for speed"""
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.reverse_vocab = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # Create TensorFlow lookup tables for fast encoding/decoding
        keys = tf.constant(list(self.vocab.keys()))
        values = tf.constant(list(self.vocab.values()))
        self.char_to_id = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=0
        )
        
        keys_reverse = tf.constant(list(self.reverse_vocab.keys()))
        values_reverse = tf.constant(list(self.reverse_vocab.values()))
        self.id_to_char = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_reverse, values_reverse),
            default_value='?'
        )
        
        print(f"📚 Built vocabulary: {self.vocab_size} characters")
        print(f"   Characters: {''.join(chars)}")
    
    def encode_fast(self, text: str) -> tf.Tensor:
        """Fast encoding using TensorFlow operations"""
        chars = tf.strings.bytes_split(text)
        return self.char_to_id.lookup(chars)
    
    def decode_fast(self, ids: tf.Tensor) -> str:
        """Fast decoding using TensorFlow operations"""
        chars = self.id_to_char.lookup(ids)
        return tf.strings.reduce_join(chars).numpy().decode('utf-8')
    
    def encode(self, text: str) -> List[int]:
        """Fallback encoding for compatibility"""
        return [self.vocab.get(ch, 0) for ch in text]
    
    def decode(self, ids: List[int]) -> str:
        """Fallback decoding for compatibility"""
        return ''.join([self.reverse_vocab.get(id, '?') for id in ids])


class OptimizedLanguageModel(tf.keras.Model):
    """High-performance language model using TensorFlow/Keras"""
    
    def __init__(self, vocab_size: int, context_length: int = 32, 
                 embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        
        # Embedding layer - converts character IDs to dense vectors
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=context_length
        )
        
        # Positional encoding to help model understand sequence order
        self.pos_encoding = tf.keras.layers.Embedding(
            context_length, embedding_dim
        )
        
        # LSTM layers for sequence modeling
        self.lstm_layers = []
        for i in range(num_layers):
            return_sequences = i < num_layers - 1  # Only last layer doesn't return sequences
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    hidden_dim, 
                    return_sequences=return_sequences,
                    dropout=0.1,
                    recurrent_dropout=0.1
                )
            )
        
        # Output layers
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(vocab_size)
        
        print(f"🏗️  Created optimized language model:")
        print(f"   Vocabulary: {vocab_size}")
        print(f"   Context length: {context_length}")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Layers: {num_layers}")
    
    def call(self, inputs, training=None):
        """Forward pass through the model"""
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Create position indices
        positions = tf.range(seq_length)[tf.newaxis, :]
        positions = tf.tile(positions, [batch_size, 1])
        
        # Embedding + positional encoding
        token_embeddings = self.embedding(inputs)
        pos_embeddings = self.pos_encoding(positions)
        x = token_embeddings + pos_embeddings
        
        # LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # Output layers
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        logits = self.output_layer(x)
        
        return logits


class OptimizedMindMirrorAI:
    """High-performance MindMirror AI using TensorFlow with GPU acceleration"""
    
    def __init__(self, context_length: int = 32, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 3):
        self.processor = OptimizedTextProcessor()
        self.model = None
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        print("🚀 Optimized MindMirror AI initialized!")
        print(f"   TensorFlow version: {tf.__version__}")
        print(f"   GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        print(f"   Context length: {context_length}")
    
    def learn_from_text(self, text: str) -> None:
        """Learn from text data with optimized preprocessing"""
        print(f"\n📖 Learning from text ({len(text)} characters)...")
        
        # Build vocabulary
        self.processor.build_vocabulary(text)
        
        # Create model
        self.model = OptimizedLanguageModel(
            vocab_size=self.processor.vocab_size,
            context_length=self.context_length,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Prepare training data efficiently
        encoded_text = self.processor.encode(text)
        
        # Create training sequences
        sequences = []
        targets = []
        
        for i in range(0, len(encoded_text) - self.context_length, self.context_length // 4):
            if i + self.context_length < len(encoded_text):
                seq = encoded_text[i:i + self.context_length]
                target = encoded_text[i + 1:i + self.context_length + 1]
                sequences.append(seq)
                targets.append(target[-1])  # Predict last character
        
        self.X = np.array(sequences)
        self.y = tf.keras.utils.to_categorical(targets, self.processor.vocab_size)
        
        print(f"📝 Prepared {len(sequences)} training sequences")
        print(f"   Input shape: {self.X.shape}")
        print(f"   Target shape: {self.y.shape}")
    
    def train(self, epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001) -> None:
        """Train the model with GPU acceleration"""
        if self.model is None:
            print("❌ No model created. Call learn_from_text() first.")
            return
        
        print(f"\n🏃‍♂️ Training with GPU acceleration...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        # Compile model with optimized settings
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for better training
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.8, patience=5, min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=10, restore_best_weights=True
            )
        ]
        
        # Train with timing
        start_time = time.time()
        
        history = self.model.fit(
            self.X, self.y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds!")
        print(f"   Final loss: {history.history['loss'][-1]:.4f}")
        print(f"   Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    def generate(self, prompt: str = "", max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text using the trained model"""
        if self.model is None:
            return "❌ Model not trained yet."
        
        # Prepare initial context
        if prompt:
            context = self.processor.encode(prompt)
        else:
            context = [0] * self.context_length
        
        # Ensure context is the right length
        if len(context) > self.context_length:
            context = context[-self.context_length:]
        elif len(context) < self.context_length:
            context = [0] * (self.context_length - len(context)) + context
        
        generated = prompt
        
        for _ in range(max_length):
            # Prepare input
            input_seq = np.array([context])
            
            # Get prediction
            logits = self.model(input_seq, training=False)
            logits = logits.numpy()[0]
            
            # Apply temperature sampling
            if temperature == 0:
                next_id = np.argmax(logits)
            else:
                logits = logits / temperature
                probabilities = tf.nn.softmax(logits).numpy()
                next_id = np.random.choice(len(probabilities), p=probabilities)
            
            # Add to generated text
            next_char = self.processor.reverse_vocab.get(next_id, '?')
            generated += next_char
            
            # Update context
            context = context[1:] + [next_id]
        
        return generated
    
    def chat_response(self, user_input: str) -> str:
        """Generate a response to user input"""
        # Use the last part of user input as context
        context_prompt = user_input[-self.context_length:]
        response = self.generate(
            prompt=context_prompt, 
            max_length=100, 
            temperature=0.7
        )
        # Return only the new part
        return response[len(context_prompt):]
    
    def save_model(self, path: str) -> None:
        """Save the trained model"""
        if self.model:
            self.model.save_weights(path)
            print(f"💾 Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model"""
        if self.model and os.path.exists(path):
            self.model.load_weights(path)
            print(f"📁 Model loaded from {path}")
