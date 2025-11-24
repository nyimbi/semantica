#!/bin/bash
# Development environment setup script
# Sets up Python virtual environment and installs dependencies

set -e

echo "🚀 Setting up development environment..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10 or higher"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📦 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install project in editable mode with dev dependencies
echo "📥 Installing package with dev dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks if available
if command -v pre-commit &> /dev/null; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install || echo "⚠️  Pre-commit installation skipped"
else
    echo "⚠️  pre-commit not found, skipping hooks installation"
fi

# Verify installation
echo "✅ Verifying installation..."
if python -c "import semantica; print(f'Semantica version: {semantica.__version__}')" 2>/dev/null; then
    echo "✅ Installation verified"
else
    echo "❌ Installation verification failed"
    exit 1
fi

echo ""
echo "✨ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  source venv/bin/activate    # Activate virtual environment"
echo "  pytest                      # Run tests"
echo "  black semantica/            # Format code"
echo "  .github/scripts/check-code.sh  # Run all checks"

