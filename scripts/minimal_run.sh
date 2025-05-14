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
  max_length: 4096
  max_new_tokens: 1024
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repetition_penalty: 1.1

tools:
  calc:
    token: "<<calc>>"
    timeout_ms: 200
  search:
    token: "<<search>>"
    timeout_ms: 200
  code:
    token: "<<code>>"
    timeout_ms: 200
  weather:
    token: "<<weather>>"
    timeout_ms: 200
  python:
    token: "<<py>>"
    timeout_ms: 200
    allowed_modules: ["math", "datetime", "re", "collections", "itertools"]
  bash:
    token: "<<bash>>"
    timeout_ms: 200
    allowed_commands: ["ls", "cat", "grep", "find", "echo", "wc"]

logging:
  level: "INFO"
  file: "logs/nanocog.log"
EOL

echo "Created minimal configuration at $CONFIG_FILE"

# Run with minimal configuration
echo "Running model in minimal configuration..."
python main.py --config $CONFIG_FILE "$@"

echo "=== Completed minimal run ===" 