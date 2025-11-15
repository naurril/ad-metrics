#!/bin/bash
# Build API documentation using Sphinx

set -e

echo "============================================"
echo "Building AD-Metrics API Documentation"
echo "============================================"

# Check if we're in the docs directory
if [ ! -f "conf.py" ]; then
    echo "Error: Please run this script from the docs/ directory"
    exit 1
fi

# Install documentation dependencies
echo ""
echo "Installing documentation dependencies..."
pip install -q -r requirements.txt

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
make clean

# Build HTML documentation
echo ""
echo "Building HTML documentation..."
make html

# Check if build succeeded
if [ -d "_build/html" ]; then
    echo ""
    echo "============================================"
    echo "Documentation built successfully!"
    echo "============================================"
    echo ""
    echo "Open: _build/html/index.html"
    echo ""
    echo "To view locally, run:"
    echo "  cd _build/html && python -m http.server 8000"
    echo "  Then open: http://localhost:8000"
    echo ""
else
    echo ""
    echo "Error: Build failed!"
    exit 1
fi
