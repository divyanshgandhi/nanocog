#!/bin/bash
# Script to run Nano-Cog model without 4-bit quantization

echo "=== Running Nano-Cog without quantization ==="

# Check if config file exists
CONFIG_FILE="src/configs/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "Found config file: $CONFIG_FILE"
    
    # Create a temporary backup
    cp $CONFIG_FILE ${CONFIG_FILE}.bak
    echo "Created backup at ${CONFIG_FILE}.bak"
    
    # Modify quantization setting in config file
    sed -i 's/quantization: "4bit"/quantization: "none"/g' $CONFIG_FILE
    echo "Changed quantization setting to 'none' in config"
    
    # Set environment variable to skip bitsandbytes
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
    export BITSANDBYTES_NOWELCOME=1
    
    # Run the model
    echo "Running model without quantization..."
    python main.py "$@"
    
    # Restore backup
    mv ${CONFIG_FILE}.bak $CONFIG_FILE
    echo "Restored original config"
else
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "=== Completed run without quantization ===" 