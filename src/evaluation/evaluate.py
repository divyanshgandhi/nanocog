"""
Evaluation script for Nano-Cog model
"""

import os
import json
import time
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import load_config
from core.model import NanoCogModel
from tools.dispatcher import ToolDispatcher


class NanoCogEvaluator:
    """
    Evaluator for Nano-Cog model on various benchmarks
    """

    def __init__(self, model_path=None, config_path=None):
        """
        Initialize evaluator

        Args:
            model_path (str, optional): Path to model checkpoint
            config_path (str, optional): Path to config file
        """
        self.config = load_config(config_path)

        # Load model
        print("Loading model...")
        self.model = NanoCogModel(config_path)

        if model_path:
            print(f"Loading model checkpoint from {model_path}")
            self.model.load(model_path)

        # Initialize tool dispatcher
        self.tool_dispatcher = ToolDispatcher(config_path)

        # Load evaluation datasets
        self.eval_data = self._load_eval_datasets()

    def _load_eval_datasets(self):
        """
        Load evaluation datasets

        Returns:
            dict: Dictionary of evaluation datasets
        """
        eval_dir = os.path.join("data", "evaluation")
        datasets = {}

        # GSM8K
        gsm8k_path = os.path.join(eval_dir, "gsm8k_10pct.json")
        if os.path.exists(gsm8k_path):
            with open(gsm8k_path, "r") as f:
                datasets["gsm8k"] = json.load(f)
            print(f"Loaded {len(datasets['gsm8k'])} GSM8K examples")

        # HumanEval-mini
        humaneval_path = os.path.join(eval_dir, "humaneval_mini.json")
        if os.path.exists(humaneval_path):
            with open(humaneval_path, "r") as f:
                datasets["humaneval"] = json.load(f)
            print(f"Loaded {len(datasets['humaneval'])} HumanEval examples")

        # Symbolic-Math
        symbolic_path = os.path.join(eval_dir, "symbolic_math_quiz.json")
        if os.path.exists(symbolic_path):
            with open(symbolic_path, "r") as f:
                datasets["symbolic_math"] = json.load(f)
            print(f"Loaded {len(datasets['symbolic_math'])} Symbolic-Math examples")

        return datasets

    def _extract_final_answer(self, response, dataset):
        """
        Extract final answer from model response based on dataset

        Args:
            response (str): Model response
            dataset (str): Dataset name

        Returns:
            str: Extracted final answer
        """
        if dataset == "gsm8k":
            # Look for the final answer after "####"
            if "####" in response:
                return response.split("####")[-1].strip()

            # Look for "FINAL:" marker
            if "FINAL:" in response:
                return response.split("FINAL:")[-1].strip()

            # Try to find a numerical answer at the end
            match = re.search(
                r"(?:answer|result|solution)[^\d]*(-?\d+(?:\.\d+)?)",
                response,
                re.IGNORECASE,
            )
            if match:
                return match.group(1)

            # Default to last line
            return response.strip().split("\n")[-1].strip()

        elif dataset == "humaneval":
            # Extract Python function code
            code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
            if code_blocks:
                return code_blocks[0].strip()

            # If no code blocks, try to extract function definition
            match = re.search(r"def\s+.*?:\s*(?:.*?\n)+", response, re.DOTALL)
            if match:
                return match.group(0).strip()

            # Default to full response
            return response.strip()

        elif dataset == "symbolic_math":
            # Look for "FINAL:" marker
            if "FINAL:" in response:
                return response.split("FINAL:")[-1].strip()

            # Look for "Therefore" statement
            match = re.search(r"Therefore,?\s*(.*?)\.", response)
            if match:
                return match.group(1).strip()

            # Default to last paragraph
            paragraphs = response.strip().split("\n\n")
            return paragraphs[-1].strip()

        # Default case
        return response.strip()

    def _count_reasoning_tokens(self, response):
        """
        Count reasoning tokens in a response

        Args:
            response (str): Model response

        Returns:
            int: Number of reasoning tokens
        """
        # Count tokens in response using model's tokenizer
        return len(self.model.tokenizer.encode(response))

    def _check_gsm8k_correctness(self, pred, gold):
        """
        Check correctness of GSM8K answer

        Args:
            pred (str): Predicted answer
            gold (str): Gold answer

        Returns:
            bool: Whether prediction is correct
        """
        # Extract numbers from both prediction and gold
        pred_numbers = re.findall(r"-?\d+(?:\.\d+)?", pred)
        gold_numbers = re.findall(r"-?\d+(?:\.\d+)?", gold)

        if not pred_numbers or not gold_numbers:
            return False

        # Compare the last number in each
        pred_val = float(pred_numbers[-1])
        gold_val = float(gold_numbers[-1])

        # Check if they're close enough (handle potential floating point issues)
        return abs(pred_val - gold_val) < 1e-6

    def _check_humaneval_correctness(self, pred, gold, task_id):
        """
        Placeholder for HumanEval correctness checking
        In a real implementation, this would use the official HumanEval evaluator

        Args:
            pred (str): Predicted code
            gold (str): Gold code
            task_id (str): HumanEval task ID

        Returns:
            bool: Whether prediction is correct
        """
        # This is a simplified version - actual evaluation would run test cases
        # For now, just check if essential function signature is present
        function_name = task_id.split("/")[-1].strip()

        # Check if function definition exists in prediction
        if not re.search(rf"def\s+{function_name}\s*\(", pred):
            return False

        # Placeholder for syntax check - in a real implementation, this would compile and run tests
        try:
            compile(pred, "<string>", "exec")
            return True
        except Exception:
            return False

    def _check_symbolic_math_correctness(self, pred, gold):
        """
        Check correctness of symbolic math answers
        This is a placeholder that would need domain-specific implementation

        Args:
            pred (str): Predicted answer
            gold (str): Gold answer

        Returns:
            bool: Whether prediction is correct
        """
        # Simple keyword-based check - in a real implementation, this would use symbolic comparison
        # Extract the final answer from gold (after "Therefore")
        gold_conclusion = ""
        if "Therefore" in gold:
            gold_conclusion = gold.split("Therefore")[-1].strip()

        # Check if key parts of the gold conclusion are in the prediction
        if gold_conclusion and all(
            term in pred for term in gold_conclusion.split()[2:6]
        ):
            return True

        # Fallback to checking if certain key expressions are present
        key_expressions = re.findall(
            r"([a-zA-Z0-9\^]+\s*=\s*[a-zA-Z0-9\^\/\+\-\.\(\)]+)", gold
        )
        if key_expressions:
            for expr in key_expressions:
                # Normalize spaces and check
                normalized_expr = re.sub(r"\s+", "", expr)
                normalized_pred = re.sub(r"\s+", "", pred)
                if normalized_expr in normalized_pred:
                    return True

        return False

    def evaluate_dataset(self, dataset_name, max_examples=None, verbose=True):
        """
        Evaluate model on a specific dataset

        Args:
            dataset_name (str): Name of dataset to evaluate
            max_examples (int, optional): Maximum number of examples to evaluate
            verbose (bool, optional): Whether to print individual example results

        Returns:
            dict: Evaluation results
        """
        if dataset_name not in self.eval_data:
            print(f"Dataset {dataset_name} not found")
            return {}

        dataset = self.eval_data[dataset_name]

        if max_examples is not None:
            dataset = dataset[:max_examples]

        results = []
        total_correct = 0
        total_reasoning_tokens = 0
        total_time = 0

        for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
            prompt = example["prompt"]
            gold = example.get("answer", "")

            # Record start time
            start_time = time.time()

            # Generate response
            response = self.model.generate(prompt)

            # Process any tool calls in the response
            processed_response = self.tool_dispatcher.process_text(response)

            # Record end time
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Count reasoning tokens
            reasoning_tokens = self._count_reasoning_tokens(processed_response)

            # Extract and check final answer
            final_answer = self._extract_final_answer(processed_response, dataset_name)

            # Check correctness based on dataset
            if dataset_name == "gsm8k":
                final_gold = example.get("final_answer", "")
                is_correct = self._check_gsm8k_correctness(final_answer, final_gold)
            elif dataset_name == "humaneval":
                task_id = example.get("task_id", "")
                is_correct = self._check_humaneval_correctness(
                    final_answer, gold, task_id
                )
            elif dataset_name == "symbolic_math":
                is_correct = self._check_symbolic_math_correctness(final_answer, gold)
            else:
                is_correct = final_answer.strip() == gold.strip()

            # Update totals
            if is_correct:
                total_correct += 1
            total_reasoning_tokens += reasoning_tokens
            total_time += elapsed_time

            # Store result
            result = {
                "example_id": i,
                "prompt": prompt,
                "response": processed_response,
                "final_answer": final_answer,
                "gold_answer": gold,
                "is_correct": is_correct,
                "reasoning_tokens": reasoning_tokens,
                "time": elapsed_time,
            }
            results.append(result)

            if verbose:
                print(f"\nExample {i+1}:")
                print(f"Prompt: {prompt[:100]}...")
                print(f"Final answer: {final_answer}")
                print(f"Correct: {is_correct}")
                print(f"Reasoning tokens: {reasoning_tokens}")
                print(f"Time: {elapsed_time:.2f}s")

        # Calculate overall metrics
        accuracy = total_correct / len(dataset) if dataset else 0
        avg_reasoning_tokens = total_reasoning_tokens / len(dataset) if dataset else 0
        avg_time = total_time / len(dataset) if dataset else 0

        metrics = {
            "dataset": dataset_name,
            "num_examples": len(dataset),
            "correct": total_correct,
            "accuracy": accuracy,
            "total_reasoning_tokens": total_reasoning_tokens,
            "avg_reasoning_tokens": avg_reasoning_tokens,
            "total_time": total_time,
            "avg_time": avg_time,
        }

        print(f"\n{dataset_name} Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f} ({total_correct}/{len(dataset)})")
        print(f"Average reasoning tokens: {avg_reasoning_tokens:.2f}")
        print(f"Average time per example: {avg_time:.2f}s")

        return {"metrics": metrics, "results": results}

    def evaluate_all(self, max_examples=None, output_dir="results"):
        """
        Evaluate model on all available datasets

        Args:
            max_examples (int, optional): Maximum number of examples per dataset
            output_dir (str, optional): Directory to save results

        Returns:
            dict: All evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)

        all_results = {}
        all_metrics = []

        for dataset_name in self.eval_data.keys():
            print(f"\nEvaluating on {dataset_name}...")
            results = self.evaluate_dataset(dataset_name, max_examples, verbose=False)

            if results:
                all_results[dataset_name] = results
                all_metrics.append(results["metrics"])

                # Save individual dataset results
                output_path = os.path.join(output_dir, f"{dataset_name}_results.json")
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

        # Save combined metrics as CSV
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_path = os.path.join(output_dir, "metrics_summary.csv")
            metrics_df.to_csv(metrics_path, index=False)

            # Also save as JSON
            metrics_json_path = os.path.join(output_dir, "metrics_summary.json")
            with open(metrics_json_path, "w") as f:
                json.dump(all_metrics, f, indent=2)

        # Save test report markdown
        self._generate_test_report(all_metrics, output_dir)

        return all_results

    def _generate_test_report(self, all_metrics, output_dir):
        """
        Generate markdown test report

        Args:
            all_metrics (list): List of metrics dictionaries
            output_dir (str): Output directory
        """
        report_path = os.path.join(output_dir, "test_report.md")

        with open(report_path, "w") as f:
            f.write("# Nano-Cog 0.1 Evaluation Report\n\n")
            f.write(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            f.write("## Summary\n\n")
            f.write("| Dataset | Accuracy | Avg. Reasoning Tokens | Avg. Time (s) |\n")
            f.write("|---------|----------|----------------------|---------------|\n")

            for metrics in all_metrics:
                f.write(
                    f"| {metrics['dataset']} | {metrics['accuracy']:.4f} | {metrics['avg_reasoning_tokens']:.1f} | {metrics['avg_time']:.2f} |\n"
                )

            f.write("\n## Detailed Results\n\n")

            for metrics in all_metrics:
                f.write(f"### {metrics['dataset']}\n\n")
                f.write(f"- **Examples evaluated:** {metrics['num_examples']}\n")
                f.write(
                    f"- **Correct answers:** {metrics['correct']}/{metrics['num_examples']}\n"
                )
                f.write(f"- **Accuracy:** {metrics['accuracy']:.4f}\n")
                f.write(
                    f"- **Total reasoning tokens:** {metrics['total_reasoning_tokens']}\n"
                )
                f.write(
                    f"- **Average reasoning tokens per answer:** {metrics['avg_reasoning_tokens']:.2f}\n"
                )
                f.write(f"- **Total evaluation time:** {metrics['total_time']:.2f}s\n")
                f.write(
                    f"- **Average time per example:** {metrics['avg_time']:.2f}s\n\n"
                )

        print(f"Test report saved to {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Nano-Cog model")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to evaluate (if not specified, evaluates all)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        help="Maximum number of examples to evaluate per dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )

    args = parser.parse_args()

    evaluator = NanoCogEvaluator(args.model, args.config)

    if args.dataset:
        evaluator.evaluate_dataset(args.dataset, args.max_examples)
    else:
        evaluator.evaluate_all(args.max_examples, args.output_dir)
