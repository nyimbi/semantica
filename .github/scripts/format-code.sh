#!/bin/bash
# Code formatting script
# Formats code with black and sorts imports with isort

set -e

echo "🎨 Formatting code..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if directories exist
if [ ! -d "semantica" ]; then
    echo "❌ semantica/ directory not found"
    exit 1
fi

# Format with black
echo "📝 Formatting with black..."
black semantica/
echo "✅ Black formatting complete"

# Sort imports with isort
echo "📦 Sorting imports with isort..."
isort semantica/
echo "✅ Import sorting complete"

echo ""
echo "✅ Code formatting complete!"

