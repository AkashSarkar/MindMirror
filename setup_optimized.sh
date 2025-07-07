#!/bin/bash

# MindMirror AI Setup Script for Mac M1/M2
# This script installs optimized dependencies for maximum performance

echo "🚀 Setting up MindMirror AI for Mac M1/M2..."
echo "=" * 50

# Check if we're on Mac
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This setup script is optimized for macOS. You may need to modify it for other systems."
fi

# Check for Python 3.8+
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Install basic requirements first
echo "📦 Installing basic requirements..."
pip3 install matplotlib jupyter ipykernel

# Install TensorFlow with Metal support for Mac M1/M2
echo "🔥 Installing TensorFlow with Metal GPU acceleration..."
pip3 install tensorflow-macos>=2.13.0
pip3 install tensorflow-metal>=1.0.0

# Install scientific computing packages
echo "🧮 Installing scientific computing packages..."
pip3 install numpy>=1.21.0 scikit-learn>=1.3.0 seaborn>=0.11.0

# Install optional PyTorch (alternative to TensorFlow)
echo "🔷 Installing PyTorch (optional)..."
pip3 install torch torchvision

# Install text processing libraries
echo "📝 Installing text processing libraries..."
pip3 install nltk spacy

# Install system monitoring
echo "📊 Installing performance monitoring..."
pip3 install psutil

# Test TensorFlow installation
echo "🧪 Testing TensorFlow installation..."
python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
if len(tf.config.list_physical_devices('GPU')) > 0:
    print('🚀 GPU acceleration is working!')
else:
    print('⚠️  GPU not detected, but CPU will still be optimized')
"

echo "✅ Setup complete! You can now run:"
echo "   cd src"
echo "   python3 main_optimized.py --mode demo --fast"
