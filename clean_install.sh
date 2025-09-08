#!/bin/bash
# Clean installation script for MatterVial

echo "MatterVial Clean Installation Script"
echo "===================================="

# Step 1: Clean all Python cache files
echo "Step 1: Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
echo "✓ Python cache files cleaned"

# Step 2: Clean build artifacts
echo "Step 2: Cleaning build artifacts..."
rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
echo "✓ Build artifacts cleaned"

# Step 3: Clean any temporary files
echo "Step 3: Cleaning temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*~" -delete 2>/dev/null || true
echo "✓ Temporary files cleaned"

# Step 4: Install the package
echo "Step 4: Installing MatterVial..."
if python setup.py install; then
    echo "✓ MatterVial installed successfully!"
else
    echo "✗ Installation failed. Trying alternative method..."
    
    # Alternative: pip install in development mode
    echo "Trying pip install in development mode..."
    if pip install -e .; then
        echo "✓ MatterVial installed successfully in development mode!"
    else
        echo "✗ Both installation methods failed."
        echo "Please check the error messages above."
        exit 1
    fi
fi

echo ""
echo "Installation completed!"
echo "You can now import mattervial in Python:"
echo "  import mattervial.featurizers"
echo ""
echo "To test the installation, run:"
echo "  python -c 'import mattervial.featurizers; print(\"✓ MatterVial imported successfully\")'"
