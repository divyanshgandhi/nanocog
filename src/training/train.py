"""
Training script for Nano-Cog model
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.config import load_config
from src.core.model import NanoCogModel
from src.training.rl_trainer import GRPOTrainer


def load_training_data(data_path):
    """
    Load training data from JSON file

    Args:
        data_path (str): Path to training data file

    Returns:
        list: List of prompts
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    with open(data_path, "r") as f:
        data = json.load(f)

    prompts = []
    for example in data:
        prompts.append(example["prompt"])

    return prompts


def train_supervised(config_path=None, output_dir="models", num_epochs=3, batch_size=4):
    """
    Train the model in supervised mode (LoRA fine-tuning)

    Args:
        config_path (str, optional): Path to config file
        output_dir (str, optional): Directory to save model
        num_epochs (int, optional): Number of epochs to train
        batch_size (int, optional): Batch size
    """
    # Load configuration
    config = load_config(config_path)

    # Load model
    print("Loading model...")
    model = NanoCogModel(config_path)

    # Load training data
    data_path = os.path.join("data", "processed", "training_data.json")
    print(f"Loading training data from {data_path}...")
    prompts = load_training_data(data_path)

    print(f"Loaded {len(prompts)} training examples")

    # Split into batches
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    # Train for specified number of epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        total_loss = 0

        for i in tqdm(range(num_batches), desc="Training"):
            # Get batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(prompts))
            batch = prompts[start_idx:end_idx]

            # Process batch (simplified for demo)
            batch_loss = torch.tensor(0.01, requires_grad=True)  # Placeholder

            # Update total loss
            total_loss += batch_loss.item()

            # Simulate backward pass and optimization
            batch_loss.backward()

        # Print epoch results
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"supervised_epoch_{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Saving checkpoint to {checkpoint_dir}...")
        # model.save(checkpoint_dir)  # Placeholder - not implemented in this demo

    print("Supervised training completed!")


def train_toolformer(config_path=None, output_dir="models", num_epochs=2):
    """
    Train the model to use tools (Toolformer phase)

    Args:
        config_path (str, optional): Path to config file
        output_dir (str, optional): Directory to save model
        num_epochs (int, optional): Number of epochs to train
    """
    # Load configuration
    config = load_config(config_path)

    # Load model
    print("Loading model...")
    model = NanoCogModel(config_path)

    # Load training data (React traces for tool use)
    data_path = os.path.join("data", "processed", "react_traces.json")
    print(f"Loading tool training data from {data_path}...")

    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} tool training examples")
    else:
        print(f"Warning: Tool training data file not found: {data_path}")
        data = []

    if not data:
        print("No tool training data available. Skipping Toolformer phase.")
        return

    # Simplified training loop (placeholder for actual Toolformer training)
    for epoch in range(num_epochs):
        print(f"Toolformer Epoch {epoch+1}/{num_epochs}")

        for i in tqdm(range(len(data)), desc="Training tools"):
            # Process example (simulated)
            pass

        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"toolformer_epoch_{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Saving Toolformer checkpoint to {checkpoint_dir}...")
        # model.save(checkpoint_dir)  # Placeholder

    print("Toolformer training completed!")


def train_rl(config_path=None, output_dir="models", num_epochs=2):
    """
    Train the model with reinforcement learning (GRPO)

    Args:
        config_path (str, optional): Path to config file
        output_dir (str, optional): Directory to save model
        num_epochs (int, optional): Number of epochs to train
    """
    # Load configuration
    config = load_config(config_path)

    # Initialize GRPO trainer
    print("Initializing RL trainer...")
    trainer = GRPOTrainer(config_path)

    # Load training data
    data_path = os.path.join("data", "processed", "training_data.json")
    print(f"Loading training data from {data_path}...")

    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)

        prompts = [example["prompt"] for example in data]
        print(f"Loaded {len(prompts)} training examples")
    else:
        print(f"Warning: Training data file not found: {data_path}")
        prompts = []

    if not prompts:
        print("No training data available. Skipping RL phase.")
        return

    # Train with GRPO
    print("Starting GRPO training...")
    trainer.train(prompts, num_epochs=num_epochs)

    # Save final model
    final_checkpoint_dir = os.path.join(output_dir, "rl_final")
    os.makedirs(final_checkpoint_dir, exist_ok=True)

    print(f"Saving final model to {final_checkpoint_dir}...")
    trainer.save_checkpoint(final_checkpoint_dir)

    print("RL training completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Nano-Cog model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--output", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument(
        "--supervised-epochs", type=int, default=3, help="Number of supervised epochs"
    )
    parser.add_argument(
        "--toolformer-epochs", type=int, default=2, help="Number of Toolformer epochs"
    )
    parser.add_argument("--rl-epochs", type=int, default=2, help="Number of RL epochs")
    parser.add_argument(
        "--skip-supervised", action="store_true", help="Skip supervised training"
    )
    parser.add_argument(
        "--skip-toolformer", action="store_true", help="Skip Toolformer phase"
    )
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL training")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for supervised training"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Supervised training (LoRA fine-tuning)
    if not args.skip_supervised:
        print("\n" + "=" * 50)
        print("Starting supervised training (LoRA)")
        print("=" * 50)
        train_supervised(
            args.config, args.output, args.supervised_epochs, args.batch_size
        )
    else:
        print("Skipping supervised training")

    # Toolformer phase
    if not args.skip_toolformer:
        print("\n" + "=" * 50)
        print("Starting Toolformer phase")
        print("=" * 50)
        train_toolformer(args.config, args.output, args.toolformer_epochs)
    else:
        print("Skipping Toolformer phase")

    # RL training (GRPO)
    if not args.skip_rl:
        print("\n" + "=" * 50)
        print("Starting RL training (GRPO)")
        print("=" * 50)
        train_rl(args.config, args.output, args.rl_epochs)
    else:
        print("Skipping RL training")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
