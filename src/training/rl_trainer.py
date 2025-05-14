"""
GRPO (Guided Reward Policy Optimization) Trainer for Nano-Cog
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import load_config
from core.model import NanoCogModel


class GRPOTrainer:
    """
    Guided Reward Policy Optimization trainer for Nano-Cog
    """

    def __init__(self, config_path=None):
        """
        Initialize the GRPO trainer

        Args:
            config_path (str, optional): Path to config file
        """
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = NanoCogModel(config_path)

        # Load judge model (frozen Mamba for ranking)
        print("Loading judge model...")
        self.judge_tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["backbone"]["checkpoint"]
        )
        self.judge_model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["backbone"]["checkpoint"],
            load_in_4bit=True,  # Always use 4-bit for judge to save memory
            torch_dtype=torch.float16,
        )

        # Freeze judge model
        for param in self.judge_model.parameters():
            param.requires_grad = False

        # Training parameters
        self.batch_size = self.config["training"]["batch_size"]
        self.candidates_per_prompt = self.config["training"]["candidates_per_prompt"]
        self.learning_rate = self.config["training"]["lr"]
        self.beta1 = self.config["training"]["optimizer"]["beta1"]
        self.beta2 = self.config["training"]["optimizer"]["beta2"]

        # Setup optimizer
        trainable_params = []

        # Add LoRA parameters
        if hasattr(self.model.model, "peft_config"):
            trainable_params.extend(self.model.model.parameters())

        # Add MoE parameters
        trainable_params.extend(self.model.moe_router.parameters())

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.learning_rate, betas=(self.beta1, self.beta2)
        )

    def generate_candidates(self, prompts):
        """
        Generate candidate responses for each prompt

        Args:
            prompts (list): List of prompt strings

        Returns:
            list: List of lists of candidate responses
        """
        all_candidates = []

        for prompt in tqdm(prompts, desc="Generating candidates"):
            candidates = []

            # Generate multiple candidates with different temperatures
            temperatures = [0.5, 0.7, 0.9, 1.1]

            for temp in temperatures[: self.candidates_per_prompt]:
                response = self.model.generate(prompt, temperature=temp, do_sample=True)

                # Extract just the response part (after the prompt)
                response = response[len(prompt) :]
                candidates.append(response)

            all_candidates.append(candidates)

        return all_candidates

    def rank_candidates(self, prompts, candidate_lists):
        """
        Rank candidate responses using the judge model

        Args:
            prompts (list): List of prompt strings
            candidate_lists (list): List of lists of candidate responses

        Returns:
            list: List of lists of scores for each candidate
        """
        all_scores = []

        # Process each prompt and its candidates
        for prompt, candidates in tqdm(
            zip(prompts, candidate_lists), desc="Ranking candidates", total=len(prompts)
        ):
            candidate_scores = []

            for candidate in candidates:
                # Calculate quality score based on the judge model
                full_text = prompt + candidate

                # Tokenize input
                inputs = self.judge_tokenizer(full_text, return_tensors="pt").to(
                    self.device
                )

                # Calculate perplexity
                with torch.no_grad():
                    outputs = self.judge_model(**inputs)

                # Shift labels for causal language modeling
                labels = inputs["input_ids"].clone()
                labels[:, :-1] = labels[:, 1:].clone()
                labels[:, -1] = -100  # Ignore last token

                # Slice to only consider the tokens in the response
                prompt_length = len(self.judge_tokenizer.encode(prompt))
                response_logits = outputs.logits[:, prompt_length - 1 : -1]
                response_labels = labels[:, prompt_length - 1 :]

                # Calculate log probabilities
                log_probs = F.log_softmax(response_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs, dim=-1, index=response_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Mask out ignored tokens
                mask = response_labels != -100
                token_log_probs = token_log_probs * mask

                # Calculate average log probability
                avg_log_prob = token_log_probs.sum() / mask.sum()

                # Apply additional reward components
                reward = avg_log_prob.item()

                # Apply DSE compression reward
                compression = self.model.dse.calculate_compression(prompt, full_text)
                reward += 0.1 * compression  # Scale factor for compression reward

                # Apply length penalty (prefer shorter answers)
                token_length = len(self.judge_tokenizer.encode(candidate))
                length_penalty = -0.001 * token_length
                reward += length_penalty

                candidate_scores.append(reward)

            all_scores.append(candidate_scores)

        return all_scores

    def compute_grpo_loss(self, prompts, candidates, scores):
        """
        Compute GRPO loss for policy optimization

        Args:
            prompts (list): List of prompt strings
            candidates (list): List of lists of candidate responses
            scores (list): List of lists of scores for each candidate

        Returns:
            torch.Tensor: GRPO loss
        """
        total_loss = 0

        for prompt_idx, prompt in enumerate(prompts):
            prompt_candidates = candidates[prompt_idx]
            prompt_scores = scores[prompt_idx]

            # Find best and worst candidates
            best_idx = np.argmax(prompt_scores)
            worst_idx = np.argmin(prompt_scores)

            best_candidate = prompt_candidates[best_idx]
            worst_candidate = prompt_candidates[worst_idx]

            # Tokenize best and worst candidates
            best_inputs = self.model.tokenizer(
                prompt + best_candidate, return_tensors="pt"
            ).to(self.device)

            worst_inputs = self.model.tokenizer(
                prompt + worst_candidate, return_tensors="pt"
            ).to(self.device)

            # Get model outputs for best candidate
            best_outputs = self.model.model(**best_inputs)

            # Get model outputs for worst candidate
            worst_outputs = self.model.model(**worst_inputs)

            # Create shifted labels for causal language modeling
            best_labels = best_inputs["input_ids"].clone()
            best_labels[:, :-1] = best_labels[:, 1:].clone()
            best_labels[:, -1] = -100

            worst_labels = worst_inputs["input_ids"].clone()
            worst_labels[:, :-1] = worst_labels[:, 1:].clone()
            worst_labels[:, -1] = -100

            # Get prompt length in tokens
            prompt_length = len(self.model.tokenizer.encode(prompt))

            # Slice to only consider the tokens in the response
            best_logits = best_outputs.logits[:, prompt_length - 1 : -1]
            best_labels = best_labels[:, prompt_length - 1 :]

            worst_logits = worst_outputs.logits[:, prompt_length - 1 : -1]
            worst_labels = worst_labels[:, prompt_length - 1 :]

            # Calculate log probabilities
            best_log_probs = F.log_softmax(best_logits, dim=-1)
            best_token_log_probs = torch.gather(
                best_log_probs, dim=-1, index=best_labels.unsqueeze(-1)
            ).squeeze(-1)

            worst_log_probs = F.log_softmax(worst_logits, dim=-1)
            worst_token_log_probs = torch.gather(
                worst_log_probs, dim=-1, index=worst_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Mask out ignored tokens
            best_mask = best_labels != -100
            best_token_log_probs = best_token_log_probs * best_mask

            worst_mask = worst_labels != -100
            worst_token_log_probs = worst_token_log_probs * worst_mask

            # Compute GRPO loss: increase probability of best, decrease probability of worst
            best_loss = -best_token_log_probs.sum() / best_mask.sum()
            worst_loss = worst_token_log_probs.sum() / worst_mask.sum()

            # Combine losses with margin
            margin = 0.5  # Enforces separation between best and worst
            prompt_loss = best_loss + worst_loss + margin

            total_loss += prompt_loss

        # Average loss over prompts
        return total_loss / len(prompts)

    def train_epoch(self, prompts):
        """
        Train for one epoch on the given prompts

        Args:
            prompts (list): List of prompt strings

        Returns:
            float: Average loss for the epoch
        """
        total_loss = 0
        num_batches = (len(prompts) + self.batch_size - 1) // self.batch_size

        # Process in batches
        for batch_idx in tqdm(range(num_batches), desc="Training batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]

            # Generate candidates
            batch_candidates = self.generate_candidates(batch_prompts)

            # Rank candidates
            batch_scores = self.rank_candidates(batch_prompts, batch_candidates)

            # Compute loss
            loss = self.compute_grpo_loss(batch_prompts, batch_candidates, batch_scores)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def train(self, prompts, num_epochs=None):
        """
        Train the model using GRPO

        Args:
            prompts (list): List of prompt strings
            num_epochs (int, optional): Number of epochs to train for

        Returns:
            list: List of losses for each epoch
        """
        if num_epochs is None:
            num_epochs = self.config["training"]["epochs"]

        losses = []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # Shuffle prompts
            np.random.shuffle(prompts)

            # Train for one epoch
            epoch_loss = self.train_epoch(prompts)
            losses.append(epoch_loss)

            print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")

            # Save checkpoint
            self.save_checkpoint(f"models/checkpoint_epoch_{epoch+1}")

        return losses

    def save_checkpoint(self, path):
        """
        Save model checkpoint

        Args:
            path (str): Path to save checkpoint
        """
        os.makedirs(path, exist_ok=True)

        # Save model
        self.model.save(path)

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """
        Load model checkpoint

        Args:
            path (str): Path to load checkpoint from
        """
        # Load model
        self.model.load(path)

        # Load optimizer state if exists
        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))

        print(f"Checkpoint loaded from {path}")


if __name__ == "__main__":
    # Test with a few sample prompts
    sample_prompts = [
        "Calculate the area of a circle with radius 5 cm.",
        "Write a function to compute the Fibonacci sequence.",
        "Explain how state-space models work.",
        "What is the capital of France and what is its population?",
    ]

    trainer = GRPOTrainer()

    # Generate and rank candidates for a small test
    candidates = trainer.generate_candidates(sample_prompts[:1])
    scores = trainer.rank_candidates(sample_prompts[:1], candidates)

    print("\nCandidate responses and scores:")
    for i, (candidate_list, score_list) in enumerate(zip(candidates, scores)):
        print(f"\nPrompt: {sample_prompts[i]}")
        for j, (candidate, score) in enumerate(zip(candidate_list, score_list)):
            print(f"Candidate {j+1} (score: {score:.4f}):")
            print(candidate[:100] + "..." if len(candidate) > 100 else candidate)
            print()
