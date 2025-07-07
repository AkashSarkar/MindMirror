"""
MindMirror AI - Optimized Main Application

High-performance version using TensorFlow with GPU acceleration for Mac M2.
Fallback to educational implementation if TensorFlow is not available.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time

# Import the optimized AI (which has built-in fallback)
from core.ai import MindMirrorAI


def load_training_data(file_path: str = None) -> str:
    """Load training data from file or use default"""
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"📁 Loaded {len(content)} characters from {file_path}")
        return content
    else:
        # Use our new conversational AI training data as default
        default_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "lib",
            "data",
            "text_samples",
            "conversational_ai.txt",
        )
        if os.path.exists(default_file):
            with open(default_file, "r", encoding="utf-8") as f:
                content = f.read()
            print(
                f"📁 Using default conversational training data ({len(content)} characters)"
            )
            return content
        else:
            # Fallback to basic training data if file not found
            print("📁 Using fallback training data")
            return """Hello! How are you?
I'm doing well, thank you for asking! I'm an AI assistant ready to help you with various tasks.

What can you help me with?
I can help you with programming questions, explain concepts, write code, and have conversations about technology and learning.

Tell me about machine learning.
Machine learning is a field of artificial intelligence where computers learn patterns from data. Instead of being explicitly programmed, they improve their performance through experience.

How do neural networks work?
Neural networks are inspired by how the human brain works. They consist of interconnected nodes called neurons that process information. Each connection has a weight that determines how much influence one neuron has on another.

What is Python?
Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, machine learning, and automation.

Can you write code?
Yes, I can help write code in various programming languages. What kind of program would you like me to help you create?

Good morning!
Good morning! I hope you're having a great start to your day. What would you like to work on today?

Thank you for your help!
You're very welcome! I'm glad I could help. Feel free to ask if you have any more questions.

Goodbye!
Goodbye! It was great chatting with you. Have a wonderful day and happy coding!"""


def train_ai(ai: MindMirrorAI, data_file: str = None, epochs: int = 35):
    """Train the advanced neural AI on conversation data"""
    print("🤖 Starting ADVANCED Neural AI Training...")
    print("🧠 This AI will learn to generate human-like responses from neural networks!")

    # Load training data
    training_text = load_training_data(data_file)
    print(f"📚 Loaded {len(training_text)} characters of training data")

    # Prepare training data
    print("⚙️  Preparing advanced conversation training data...")
    ai.learn_from_text(training_text)

    # Train the neural model with optimal parameters
    print(f"🎯 Training advanced neural model for {epochs} epochs...")
    start_time = time.time()

    # Use optimal learning rate for the extended training
    ai.train(epochs=epochs, learning_rate=0.0003)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Advanced neural training completed in {training_time:.2f} seconds!")
    print("🧠 The AI can now generate human-like responses with advanced understanding!")

    return ai


def interactive_mode(ai: MindMirrorAI):
    """Run interactive chat mode with enhanced commands"""
    print("\n🎯 MindMirror AI - Interactive Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 50)

    # Set initial temperature for generation
    temperature = 0.7

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "help":
                print("\n📋 Available Commands:")
                print("  help - Show this help")
                print("  generate <length> - Generate text (e.g., 'generate 100')")
                print("  temp <value> - Set temperature 0.1-2.0 (e.g., 'temp 0.8')")
                print("  info - Show AI information")
                print("  stats - Show generation statistics")
                print("  quit - Exit")
                continue
            elif user_input.lower().startswith("generate"):
                parts = user_input.split()
                length = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 100
                print(f"🎲 Generating {length} characters...")
                start_time = time.time()
                response = ai.generate(max_length=length, temperature=temperature)
                gen_time = time.time() - start_time
                print(f"🤖 Generated ({gen_time:.2f}s): {response}")
                continue
            elif user_input.lower().startswith("temp"):
                parts = user_input.split()
                if len(parts) > 1:
                    try:
                        new_temp = float(parts[1])
                        if 0.1 <= new_temp <= 2.0:
                            temperature = new_temp
                            print(f"🌡️  Temperature set to {temperature}")
                        else:
                            print("❌ Temperature must be between 0.1 and 2.0")
                    except ValueError:
                        print("❌ Invalid temperature value")
                else:
                    print(f"🌡️  Current temperature: {temperature}")
                continue
            elif user_input.lower() == "info":
                print(f"\n🤖 Neural AI Information:")
                print(f"   Implementation: Real Neural Conversational AI")
                print(f"   Embedding Dimension: {getattr(ai, 'embedding_dim', 'Unknown')}")
                print(f"   Hidden Dimension: {getattr(ai, 'hidden_dim', 'Unknown')}")
                print(f"   Vocabulary Size: {getattr(ai.tokenizer, 'vocab_size', 'Unknown') if hasattr(ai, 'tokenizer') else 'Unknown'}")
                print(f"   Model Status: {'Trained' if ai.model else 'Not trained'}")
                continue
            elif user_input.lower() == "stats":
                print("📊 Generation statistics not implemented yet")
                continue
            elif not user_input:
                continue

            # Generate response to user input
            print("🤔 Thinking...")
            start_time = time.time()
            response = ai.chat_response(user_input)
            response_time = time.time() - start_time
            print(f"🤖 MindMirror ({response_time:.2f}s): {response}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()


def demo_mode(ai: MindMirrorAI):
    """Run demonstration mode showing neural response generation"""
    print("\n🎭 MindMirror AI - Neural Demo Mode")
    print("=" * 50)
    print("🧠 All responses generated DIRECTLY by neural networks!")
    print("=" * 50)

    # Use the neural AI's built-in demo function
    if hasattr(ai, 'demo_neural_generation'):
        ai.demo_neural_generation()
    else:
        # Fallback demo
        test_questions = [
            "Hello",
            "What is Python?", 
            "How do I learn programming?",
            "What is AI?",
            "Thank you"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. 👤 User: {question}")
            try:
                start_time = time.time()
                response = ai.chat_response(question)
                response_time = time.time() - start_time
                print(f"   🤖 Neural AI ({response_time:.2f}s): {response}")
            except Exception as e:
                print(f"   ❌ Error: {e}")

    print("\n" + "=" * 50)
    print("✅ All responses were generated by PURE NEURAL NETWORKS!")
    print("🧠 No rule-based responses - only learned neural patterns!")


def benchmark_mode(ai: MindMirrorAI):
    """Run performance benchmarks"""
    print("\n⚡ Performance Benchmark Mode")
    print("=" * 50)

    # Test generation speed
    print("🚀 Testing generation speed...")
    test_lengths = [50, 100, 200]

    for length in test_lengths:
        print(f"\n📏 Generating {length} characters:")
        times = []
        for i in range(3):  # 3 runs for average
            start_time = time.time()
            ai.generate(max_length=length, temperature=0.7)
            times.append(time.time() - start_time)

        avg_time = sum(times) / len(times)
        chars_per_sec = length / avg_time
        print(f"   Average time: {avg_time:.3f}s ({chars_per_sec:.1f} chars/sec)")

    print(
        f"\n🎯 Implementation: {'TensorFlow (GPU-accelerated)' if hasattr(ai, 'using_tensorflow') and ai.using_tensorflow else 'Educational (CPU-only)'}"
    )


def main():
    """Main application entry point with enhanced features"""
    parser = argparse.ArgumentParser(
        description="MindMirror AI - Personal AI Assistant"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "chat", "demo", "benchmark"],
        default="demo",
        help="Run mode: train, chat, demo, or benchmark",
    )
    parser.add_argument("--data", type=str, help="Training data file path")
    parser.add_argument(
        "--epochs", type=int, default=35, help="Training epochs (default: 35)"
    )
    parser.add_argument(
        "--context", type=int, default=128, help="Embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--hidden", type=int, default=256, help="Hidden layer size (default: 256)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training (use pre-trained weights)",
    )

    args = parser.parse_args()

    print("🌟 Welcome to MindMirror AI!")
    print("Your personal AI assistant built from scratch")
    print("=" * 50)

    # Show system information
    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        print(
            f"🖥️  System: {len(gpus)} GPU(s) detected"
            if gpus
            else "🖥️  System: CPU-only mode"
        )
    except:
        print("🖥️  System: CPU-only mode (TensorFlow not available)")

    # Initialize AI with specified parameters
    print(f"⚙️  Initializing AI (embedding: {args.context}, hidden: {args.hidden})...")
    ai = MindMirrorAI(embedding_dim=args.context, hidden_dim=args.hidden)

    if args.mode == "train":
        train_ai(ai, args.data, args.epochs)
    elif args.mode == "chat":
        if not args.no_train:
            train_ai(ai, args.data, args.epochs)
        interactive_mode(ai)
    elif args.mode == "demo":
        if not args.no_train:
            train_ai(ai, args.data, args.epochs)
        demo_mode(ai)

        # Ask if user wants to continue
        print("\n" + "=" * 50)
        response = (
            input(
                "🤔 What would you like to do next?\n   (c)hat, (b)enchmark, or (q)uit: "
            )
            .strip()
            .lower()
        )
        if response in ["c", "chat"]:
            interactive_mode(ai)
        elif response in ["b", "bench", "benchmark"]:
            benchmark_mode(ai)
        elif response in ["q", "quit"]:
            print("👋 Goodbye!")
        else:
            print("👋 Goodbye!")
    elif args.mode == "benchmark":
        if not args.no_train:
            train_ai(ai, args.data, args.epochs)
        benchmark_mode(ai)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
