"""
Configuration utilities for Nano-Cog
"""

import os
import yaml
import toml


def load_config(config_path=None):
    """
    Load configuration from YAML or TOML file

    Args:
        config_path (str, optional): Path to config file. If None, uses default path.

    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            os.path.join("src", "configs", "config.yaml"),
            os.path.join("configs", "config.yaml"),
            "config.yaml",
            os.path.join("src", "configs", "config.toml"),
            os.path.join("configs", "config.toml"),
            "config.toml",
        ]

        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break

        if config_path is None:
            raise FileNotFoundError("No configuration file found")

    # Load based on file extension
    if config_path.endswith((".yaml", ".yml")):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.endswith(".toml"):
        with open(config_path, "r") as f:
            config = toml.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")

    # Fill in any missing values with defaults
    config = _apply_default_config(config)

    return config


def _apply_default_config(config):
    """
    Apply default configuration values for missing entries

    Args:
        config (dict): User configuration

    Returns:
        dict: Configuration with defaults applied
    """
    # Default model configuration
    if "model" not in config:
        config["model"] = {}

    model_defaults = {
        "backbone": {
            "name": "mamba-130m",
            "checkpoint": "state-spaces/mamba-130m",
            "hidden_size": 768,
            "quantization": "4bit",
        },
        "lora": {
            "rank": 64,
            "target_modules": ["qkv", "ffn"],
            "insertion_blocks": [22, 23],
        },
        "moe": {
            "num_experts": 2,
            "insertion_block": 12,
            "hidden_size": 32,
            "temperature": 0.7,
            "routing": "top-1",
        },
        "dynamic_symbol_engine": {
            "max_tokens": 50,
            "grammar_tokens": ["<define symbol=", ":>"],
        },
    }

    # Apply model defaults
    for key, value in model_defaults.items():
        if key not in config["model"]:
            config["model"][key] = value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey not in config["model"][key]:
                    config["model"][key][subkey] = subvalue

    # Default inference configuration
    if "inference" not in config:
        config["inference"] = {}

    inference_defaults = {
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }

    for key, value in inference_defaults.items():
        if key not in config["inference"]:
            config["inference"][key] = value

    return config


def save_config(config, path):
    """
    Save configuration to file

    Args:
        config (dict): Configuration dictionary
        path (str): Path to save file
    """
    # Determine format based on file extension
    if path.endswith((".yaml", ".yml")):
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    elif path.endswith(".toml"):
        with open(path, "w") as f:
            toml.dump(config, f)
    else:
        raise ValueError(f"Unsupported config file format: {path}")

    print(f"Configuration saved to {path}")


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully")
    print(f"Model backbone: {config['model']['backbone']['name']}")
    print(f"LoRA rank: {config['model']['lora']['rank']}")
    print(f"MoE experts: {config['model']['moe']['num_experts']}")
    print(f"Max generation length: {config['inference']['max_length']}")
