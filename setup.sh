#!/bin/bash

# Exit on error
set -e

# Function to check if running on Windows
is_windows() {
    if [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ] || \
       [ "$(expr substr $(uname -s) 1 4)" == "MSYS" ] || \
       [ "$(expr substr $(uname -s) 1 5)" == "CYGWI" ]; then
        return 0
    else
        return 1
    fi
}

# Function to clean up on error
cleanup() {
    if [ $? -ne 0 ]; then
        echo "❌ Error occurred during setup"
        echo "🧹 Cleaning up..."
        # Deactivate virtual environment if active
        if [ -n "$VIRTUAL_ENV" ]; then
            deactivate 2>/dev/null || true
        fi
        # Remove virtual environment if it exists
        [ -d "venv" ] && rm -rf venv
        echo "🔄 Please try running the script again"
        exit 1
    fi
}

# Register cleanup function
trap cleanup EXIT

echo "🚀 Setting up Next Step AI development environment..."

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 8 ]); then
    echo "❌ Error: Python 3.8 or higher is required (found $python_version)"
    exit 1
fi

echo "✅ Python version $python_version detected"

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "🧹 Removing existing virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment based on OS
if is_windows; then
    echo "🪟 Detected Windows OS"
    source venv/Scripts/activate
else
    echo "🐧 Detected Unix-like OS"
    source venv/bin/activate
fi

# Upgrade pip
echo "🔄 Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete! Virtual environment is activated and dependencies are installed."
echo "💡 To activate the virtual environment in the future:"
if is_windows; then
    echo "   Run: venv\\Scripts\\activate"
else
    echo "   Run: source venv/bin/activate"
fi

# Verify key packages
echo "🔍 Verifying installation..."
python -c "import lightgbm as lgb; import numpy as np; import pandas as pd; import sklearn; print(f'LightGBM: {lgb.__version__}\nNumPy: {np.__version__}\nPandas: {pd.__version__}\nscikit-learn: {sklearn.__version__}')"
