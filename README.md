# MindMirror - Personal AI Assistant from Scratch

A journey to build a conversational AI assistant from first principles, focusing on deep understanding of neural networks, transformers, and AI architectures.

## Project Philosophy

This project is built entirely from scratch to achieve deep understanding of:
- Neural network fundamentals and backpropagation
- Attention mechanisms and transformer architecture
- Language modeling from character-level to full conversational AI
- Mathematical foundations of modern AI systems

**No pre-trained models, no high-level frameworks - everything coded from first principles.**

## Learning Progression

### Phase 1: Foundation (micrograd)
- [ ] Autograd engine implementation
- [ ] Scalar-valued neural networks
- [ ] Basic optimization algorithms

### Phase 2: Character-Level Models
- [ ] Bigram character models
- [ ] Multi-layer perceptrons for language
- [ ] Character-level RNNs

### Phase 3: Advanced Architectures
- [ ] Attention mechanisms from scratch
- [ ] Transformer implementation
- [ ] GPT-style language model

### Phase 4: Personal Assistant
- [ ] Conversational interface
- [ ] Knowledge integration
- [ ] Coding assistance capabilities

## Project Structure

```
micrograd/          # Autograd engine from scratch
├── engine.py       # Core autograd implementation
├── nn.py          # Neural network layers
└── optim.py       # Optimization algorithms

models/            # Neural network implementations
├── bigram.py      # Character-level bigram model
├── mlp.py         # Multi-layer perceptron
├── rnn.py         # Recurrent neural networks
├── attention.py   # Attention mechanisms
└── transformer.py # Full transformer implementation

data/              # Training datasets
├── datasets.py    # Data loading utilities
└── text_samples/  # Text data for training

training/          # Training loops and utilities
├── trainer.py     # Generic training framework
└── utils.py       # Training utilities

assistant/         # Conversational AI interface
├── chat.py        # Chat interface
└── knowledge.py   # Knowledge integration

notebooks/         # Jupyter notebooks for experimentation
├── experiments/   # Various experiments
└── analysis/      # Model analysis and visualization

tests/             # Unit tests
└── requirements.txt
```

## Getting Started

1. **Set up environment**: `pip install -r requirements.txt`
2. **Start with micrograd**: Begin in `/micrograd` to understand automatic differentiation
3. **Progress through models**: Follow the learning progression in `/models`
4. **Experiment**: Use Jupyter notebooks for interactive development

## Goals

- Build a personal AI that understands my coding patterns
- Achieve deep mathematical understanding of every component
- Create a foundation for AI/ML career transition
- Develop skills to innovate and create novel architectures

## Current Status

🚧 **In Development** - Starting with micrograd implementation

---

*"The best way to understand something is to build it from scratch"*