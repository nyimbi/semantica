#!/bin/bash
# Test runner script
# Runs tests with coverage reporting

set -e

echo "🧪 Running tests..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if tests directory exists
if [ ! -d "tests" ]; then
    echo "⚠️  No tests directory found"
    echo "   Create tests/ directory and add test files"
    exit 0
fi

# Check if test files exist
if [ -z "$(find tests -name 'test_*.py' -o -name '*_test.py' 2>/dev/null)" ]; then
    echo "⚠️  No test files found"
    echo "   Add test files to tests/ directory"
    exit 0
fi

# Run tests with coverage
echo "📊 Running pytest with coverage..."
pytest \
    --cov=semantica \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml \
    -v \
    tests/

echo ""
echo "✅ Tests complete!"
echo "📊 Coverage report: htmlcov/index.html"

