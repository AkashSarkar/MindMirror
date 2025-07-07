# MindMirror - Personal AI Assistant Built From Scratch

A complete neural network-based AI assistant built from the ground up to understand the fundamentals of artificial intelligence and machine learning.

## 🎯 Project Goals

- **Deep Understanding**: Build AI from scratch to understand every component
- **Educational Journey**: Progress from single neurons to transformer-like architectures
- **Practical Application**: Create a working AI assistant you can actually use
- **Foundation Building**: Prepare for advanced AI concepts and modern architectures

## 🏗️ Project Structure

```
MindMirror/
├── lib/                    # Core neural network library
├── example/               # Step-by-step learning examples
├── src/                   # Your working AI assistant
└── notebooks/             # Jupyter exploration notebooks
```

## 🚀 Quick Start

### 📚 Learn the Fundamentals

```bash
# Start with understanding a single neuron
python3 example/01_single_neuron.py

# See how networks learn patterns
python3 example/03_multi_layer_network.py

# Understand the learning process
python3 example/04_understanding_backward.py
```

### 🤖 Try Your AI Assistant

```bash
cd src

# Quick demo (automatically uses TensorFlow if available)
python3 main.py --mode demo

# Interactive chat
python3 main.py --mode chat

# Performance benchmark
python3 main.py --mode benchmark

# Train on custom data
python3 main.py --mode train --data path/to/your/text.txt
```

### ⚡ For Maximum Performance (Mac M1/M2)

```bash
# Install optimized dependencies
./setup_optimized.sh

# Run with GPU acceleration
python3 src/main.py --mode demo
```

## 📖 Learning Path

1. **Single Neuron** → Understand basic building blocks
2. **Multi-Layer Networks** → Learn pattern recognition
3. **Text Generation** → See language emerge
4. **AI Assistant** → Build a complete system
5. **Advanced Concepts** → Attention, transformers, and beyond

## 🧠 What Makes This Special

- **Hybrid Architecture**: Uses TensorFlow for speed when available, falls back to educational code for learning
- **No Black Boxes**: Every line of code is yours and understandable
- **Educational Focus**: Each example teaches core concepts clearly
- **Real Implementation**: Working AI you can chat with and customize
- **Hardware Optimized**: Automatic GPU acceleration on Mac M1/M2
- **Scalable Foundation**: Built to grow into more advanced architectures

## ⚡ Performance Features

- **Automatic Optimization**: Detects TensorFlow and uses GPU acceleration when available
- **Educational Fallback**: Falls back to from-scratch implementation for learning
- **Mac M2 Optimized**: Special Metal GPU acceleration for Apple Silicon
- **Performance Monitoring**: Built-in benchmarks to track generation speed
- **Flexible Training**: Adjustable context length, hidden sizes, and learning rates

## 🎓 Perfect For

- Software engineers wanting to understand AI deeply
- Students learning machine learning fundamentals
- Anyone curious about how ChatGPT actually works
- Developers building custom AI applications

## 🛠️ Command Line Options

```bash
# Basic usage
python3 src/main.py --mode [demo|chat|train|benchmark]

# Customization options
python3 src/main.py --mode chat \
    --epochs 100 \
    --context 16 \
    --hidden 128 \
    --learning-rate 0.005 \
    --data custom_training.txt

# Skip training with pre-trained weights
python3 src/main.py --mode chat --no-train
```

**Available Options:**

- `--mode`: Choose demo, chat, train, or benchmark mode
- `--data`: Path to custom training text file
- `--epochs`: Number of training epochs (default: 50)
- `--context`: Context window length (default: 8)
- `--hidden`: Hidden layer size (default: 64)
- `--learning-rate`: Training learning rate (default: 0.01)
- `--no-train`: Skip training and use existing weights

## 📊 Development Roadmap

### ✅ Phase 1: Foundations (Completed)

- [x] Scalar-valued neural networks (micrograd)
- [x] Basic optimization algorithms
- [x] Multi-layer perceptrons
- [x] Text processing and generation

### ✅ Phase 2: Character-Level Models (Completed)

- [x] Character-level language models
- [x] Multi-layer networks for text
- [x] Working AI assistant prototype

### 🚧 Phase 3: Optimization (In Progress)

- [x] TensorFlow integration for speed
- [x] GPU acceleration (Mac M1/M2)
- [x] Performance benchmarking
- [ ] Advanced training techniques

### 🎯 Phase 4: Advanced Features (Planned)

- [ ] Attention mechanisms from scratch
- [ ] Transformer architecture
- [ ] Memory and context management
- [ ] Knowledge integration

## 📁 Project Structure

```
MindMirror/
├── lib/                    # Core neural network library
│   ├── micrograd.py       # Autograd engine from scratch
│   ├── models.py          # Neural network layers and models
│   ├── training.py        # Training utilities
│   └── data.py            # Data processing
├── example/               # Step-by-step learning examples
│   ├── 01_single_neuron.py    # Understanding neurons
│   ├── 02_text_classification.py
│   ├── 03_multi_layer_network.py
│   ├── 04_understanding_backward.py
│   ├── 05_activation_functions.py
│   └── 06_text_generator.py
├── src/                   # Working AI assistant
│   ├── core/             # AI implementation
│   ├── main.py           # Main application
│   └── models/           # Advanced model architectures
├── setup_optimized.sh    # Mac M1/M2 optimization setup
├── benchmark_performance.py  # Performance testing
└── requirements.txt      # Python dependencies
```

## 🚀 Getting Started

1. **Clone and setup**:

   ```bash
   git clone <repo>
   cd MindMirror
   pip install -r requirements.txt
   ```

2. **For maximum performance** (Mac M1/M2):

   ```bash
   ./setup_optimized.sh
   ```

3. **Start learning**:

   ```bash
   python3 example/01_single_neuron.py
   ```

4. **Try your AI**:
   ```bash
   python3 src/main.py --mode demo
   ```

## 🎯 Next Steps

After completing this foundation, you'll be ready to:

- Implement attention mechanisms from scratch
- Build transformer architectures
- Create domain-specific AI applications
- Understand modern LLM architectures like GPT, BERT, etc.
- Contribute to open-source AI projects with deep understanding
- Develop skills to innovate and create novel architectures

## Current Status

🚧 **In Development** - Starting with micrograd implementation

---

_"The best way to understand something is to build it from scratch"_
