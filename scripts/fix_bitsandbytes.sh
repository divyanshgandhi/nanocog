#!/bin/bash
# Script to fix bitsandbytes library for CUDA 12.6

echo "=== Fixing bitsandbytes for CUDA 12.6 ==="
echo "This script will compile bitsandbytes from source for CUDA 12.6"

# Check if CUDA is installed and accessible
if ! command -v nvcc &> /dev/null; then
    echo "CUDA toolkit not found. Please install CUDA toolkit first."
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | sed 's/,//')
echo "Detected CUDA version: $CUDA_VERSION"

# Set environment variable for CUDA library path
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
echo "Set LD_LIBRARY_PATH to include CUDA libraries"

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR
echo "Working in temporary directory: $TEMP_DIR"

# Clone bitsandbytes repository
echo "Cloning bitsandbytes repository..."
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes

# Determine CUDA version format for make
CUDA_VERSION_SHORT=$(echo $CUDA_VERSION | cut -d. -f1-2 | tr -d '.')
echo "CUDA version for make: $CUDA_VERSION_SHORT"

# Compile bitsandbytes for the detected CUDA version
echo "Compiling bitsandbytes for CUDA $CUDA_VERSION_SHORT..."
CUDA_VERSION=$CUDA_VERSION_SHORT make

# Install the compiled package
echo "Installing compiled bitsandbytes..."
pip uninstall -y bitsandbytes
pip install -e .

echo "=== bitsandbytes installation complete ==="
echo "You can now run your model with 4-bit quantization on CUDA 12.6"
echo "If you still encounter issues, try running: python -m bitsandbytes" 