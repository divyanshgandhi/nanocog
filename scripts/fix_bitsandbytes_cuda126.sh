#!/bin/bash
# Improved script to fix bitsandbytes for CUDA 12.6
set -e  # Exit on any error

echo "=== Fixing bitsandbytes for CUDA 12.6 ==="

# Check if CUDA is installed and accessible
if ! command -v nvcc &> /dev/null; then
    echo "CUDA toolkit not found. Please install CUDA toolkit first."
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | sed 's/,//')
echo "Detected CUDA version: $CUDA_VERSION"

# Set proper environment variables
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
echo "Set environment variables for CUDA"

# Uninstall existing bitsandbytes
echo "Removing existing bitsandbytes installation..."
pip uninstall -y bitsandbytes

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
echo "Working in temporary directory: $TEMP_DIR"
cd $TEMP_DIR

# Clone bitsandbytes repository
echo "Cloning bitsandbytes repository..."
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes

# Ensure we have the right branch
git checkout main

# Show available CUDA libraries for diagnosis
echo "Available CUDA libraries:"
ls -la $CUDA_HOME/lib64/libcudart*

# Create correct symlinks if needed
if [ ! -f "$CUDA_HOME/lib64/libcudart.so.12.0" ] && [ -f "$CUDA_HOME/lib64/libcudart.so.12" ]; then
    echo "Creating symlink for libcudart.so.12.0"
    sudo ln -sf $CUDA_HOME/lib64/libcudart.so.12 $CUDA_HOME/lib64/libcudart.so.12.0
fi

# Set CUDA_VERSION=126 for the build
export CUDA_VERSION=126
echo "Set CUDA_VERSION=$CUDA_VERSION for build"

# Clean any previous build artifacts
make clean

# Compile with specific CUDA version
echo "Compiling bitsandbytes for CUDA $CUDA_VERSION..."
make CUDA_VERSION=$CUDA_VERSION

# Verify the library was built
if [ ! -f "bitsandbytes/libbitsandbytes_cuda126.so" ]; then
    echo "ERROR: Failed to build libbitsandbytes_cuda126.so"
    echo "Checking what files were built:"
    find . -name "libbitsandbytes_*.so"
    exit 1
fi

echo "Successfully built libbitsandbytes_cuda126.so"

# Install the compiled package
echo "Installing the compiled bitsandbytes package..."
pip install -e .

# Verify installation and set BNB_CUDA_VERSION
echo "export BNB_CUDA_VERSION=126" >> ~/.bashrc
export BNB_CUDA_VERSION=126

echo "=== bitsandbytes installation complete ==="
echo "Testing bitsandbytes with CUDA 12.6..."

# Test installation
python -c "
import torch
import bitsandbytes as bnb
print('PyTorch version:', torch.__version__)
print('bitsandbytes version:', bnb.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Current CUDA device:', torch.cuda.current_device())
print('Current CUDA device name:', torch.cuda.get_device_name(0))

# Simple test to verify bitsandbytes works
if torch.cuda.is_available():
    # Create a small model and convert to 4-bit
    linear = bnb.nn.Linear4bit(10, 10).cuda()
    print('4-bit linear layer created successfully')
    # Test with some data
    input_data = torch.randn(1, 10).cuda()
    output = linear(input_data)
    print('Forward pass successful:', output.shape)
"

echo "If the test ran successfully, bitsandbytes is now properly installed with CUDA 12.6 support"
echo "You can now run your model with 4-bit quantization" 