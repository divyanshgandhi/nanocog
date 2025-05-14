#!/bin/bash
# Minimal run script for Nano-Cog with maximum compatibility

echo "=== Running Nano-Cog in minimal compatibility mode ==="

# Set environment variables to disable all optimization features
export NANO_COG_NO_QUANTIZATION=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export BITSANDBYTES_NOWELCOME=1
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1

# Disable Python warnings
export PYTHONWARNINGS="ignore"

# Create a minimal config
CONFIG_FILE="src/configs/minimal_config.yaml"

# Create a minimal configuration
cat > $CONFIG_FILE << EOL
# Minimal configuration for maximum compatibility
model:
  backbone:
    checkpoint: "state-spaces/mamba-130m"
    hidden_size: 768
    quantization: "none"
  lora:
    rank: 8
    target_modules: ["mixer.out_proj", "mixer.x_proj"]
  moe:
    hidden_size: 256
    num_experts: 2
    temperature: 0.1
  dynamic_symbol_engine:
    max_tokens: 16
    grammar_tokens: ["<def>", "</def>"]

inference:
  max_length: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repetition_penalty: 1.1

tools:
  calc:
    token: "<calc>"
  search:
    token: "<search>"
  code:
    token: "<code>"
  weather:
    token: "<weather>"

logging:
  level: "INFO"
  file: "logs/nanocog.log"
EOL

echo "Created minimal configuration at $CONFIG_FILE"

# Run with minimal configuration
echo "Running model in minimal configuration..."
python main.py --config $CONFIG_FILE "$@"

echo "=== Completed minimal run ===" 