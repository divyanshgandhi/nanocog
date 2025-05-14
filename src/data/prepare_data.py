"""
Data Preparation Script for Nano-Cog

Downloads and processes training data from:
- mini-GSM8K
- HumanEval-mini
- ReAct traces
"""

import os
import json
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import load_config


def prepare_gsm8k(config, output_dir):
    """
    Prepare mini-GSM8K dataset

    Args:
        config (dict): Configuration dictionary
        output_dir (str): Output directory path

    Returns:
        list: List of processed examples
    """
    print("Preparing mini-GSM8K dataset...")

    # Load GSM8K dataset
    try:
        gsm8k = load_dataset("gsm8k", "main")

        # Take a subset for mini version (10% for training, 10% for evaluation)
        train_size = min(len(gsm8k["train"]), 1000)  # Limit to 1000 examples

        # Split into train and test
        train_indices = random.sample(range(len(gsm8k["train"])), train_size)
        train_data = [gsm8k["train"][i] for i in train_indices]

        # Process examples
        processed_examples = []

        for example in tqdm(train_data, desc="Processing GSM8K examples"):
            question = example["question"]
            answer_with_steps = example["answer"]

            # Extract just the final answer
            final_answer = answer_with_steps.split("####")[-1].strip()

            # Format as a prompt with CoT
            prompt = f"Solve the following math problem step-by-step:\n\nProblem: {question}\n\nSolution:"

            processed_examples.append(
                {
                    "prompt": prompt,
                    "answer": answer_with_steps,
                    "final_answer": final_answer,
                    "source": "gsm8k",
                }
            )

        # Save to disk
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "mini_gsm8k.json"), "w") as f:
            json.dump(processed_examples, f, indent=2)

        print(f"Saved {len(processed_examples)} GSM8K examples")
        return processed_examples

    except Exception as e:
        print(f"Error preparing GSM8K dataset: {str(e)}")
        return []


def prepare_humaneval(config, output_dir):
    """
    Prepare HumanEval-mini dataset

    Args:
        config (dict): Configuration dictionary
        output_dir (str): Output directory path

    Returns:
        list: List of processed examples
    """
    print("Preparing HumanEval-mini dataset...")

    try:
        # Load HumanEval dataset
        humaneval = load_dataset("openai_humaneval")

        # Take a subset for mini version (20 tasks as specified in architecture doc)
        eval_size = min(len(humaneval["test"]), 20)
        eval_indices = random.sample(range(len(humaneval["test"])), eval_size)
        eval_data = [humaneval["test"][i] for i in eval_indices]

        # Process examples
        processed_examples = []

        for example in tqdm(eval_data, desc="Processing HumanEval examples"):
            task_id = example["task_id"]
            prompt = example["prompt"]
            canonical_solution = example["canonical_solution"]

            # Format as a coding prompt
            formatted_prompt = f"Write a Python function according to the following specification:\n\n{prompt}"

            processed_examples.append(
                {
                    "prompt": formatted_prompt,
                    "answer": canonical_solution,
                    "task_id": task_id,
                    "source": "humaneval",
                }
            )

        # Save to disk
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "humaneval_mini.json"), "w") as f:
            json.dump(processed_examples, f, indent=2)

        print(f"Saved {len(processed_examples)} HumanEval examples")
        return processed_examples

    except Exception as e:
        print(f"Error preparing HumanEval dataset: {str(e)}")
        return []


def prepare_react_traces(config, output_dir):
    """
    Prepare ReAct traces dataset

    Args:
        config (dict): Configuration dictionary
        output_dir (str): Output directory path

    Returns:
        list: List of processed examples
    """
    print("Preparing ReAct traces dataset...")

    # Synthetic ReAct examples (since actual ReAct dataset might not be available)
    react_examples = [
        {
            "question": "What is the population of France divided by the population of Germany?",
            "thought": "I need to find the population of France and Germany, then divide them.\n1. Population of France is approximately 67 million.\n2. Population of Germany is approximately 83 million.\n3. Now I need to divide: 67 million / 83 million.",
            "action": "<<calc>>67/83<<calc>>",
            "observation": "0.8072289156626506",
            "answer": "The population of France divided by the population of Germany is approximately 0.807 or about 80.7%.",
        },
        {
            "question": "Write a Python function to check if a number is prime.",
            "thought": "I'll write a function to check if a number is prime.\nA prime number is only divisible by 1 and itself.\nWe can check this by trying to divide the number by all integers from 2 to the square root of the number.",
            "action": '<<py>>def is_prime(n):\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True\n\n# Test the function\nfor num in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:\n    print(f"{num} is prime: {is_prime(num)}")<<py>>',
            "observation": "2 is prime: True\n3 is prime: True\n4 is prime: False\n5 is prime: True\n6 is prime: False\n7 is prime: True\n8 is prime: False\n9 is prime: False\n10 is prime: False\n11 is prime: True",
            "answer": "Here's a Python function to check if a number is prime:\n```python\ndef is_prime(n):\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True\n```\nThis function efficiently checks if a number is prime by testing divisibility by 2 and 3, then checking all numbers of the form 6k±1 up to the square root of n.",
        },
        {
            "question": "What files are in the current directory?",
            "thought": "I need to list the files in the current directory. I can use the 'ls' command in bash.",
            "action": "<<bash>>ls<<bash>>",
            "observation": "README.md\nrequirements.txt\nsrc\ndata\nmodels\napp.py\nmain.py",
            "answer": "The current directory contains the following files and directories:\n- README.md\n- requirements.txt\n- src (directory)\n- data (directory)\n- models (directory)\n- app.py\n- main.py",
        },
    ]

    # Generate more synthetic examples by template modification
    additional_examples = []

    # Template for math questions
    math_templates = [
        "Calculate {operation} of {num1} and {num2}.",
        "What is {num1} {operator} {num2}?",
        "If I have {num1} items and use {num2}, how many remain?",
        "Find the {operation} of {num1} and {num2}.",
    ]

    operations = ["sum", "difference", "product", "quotient"]
    operators = ["+", "-", "*", "/"]

    for _ in range(5):
        template_idx = random.randint(0, len(math_templates) - 1)
        template = math_templates[template_idx]

        num1 = random.randint(10, 100)
        num2 = random.randint(1, 20)
        op_idx = random.randint(0, len(operations) - 1)

        question = template.format(
            operation=operations[op_idx],
            operator=operators[op_idx],
            num1=num1,
            num2=num2,
        )

        # Create synthetic ReAct trace
        thought = f"I need to calculate the {operations[op_idx]} of {num1} and {num2}."
        action = f"<<calc>>{num1}{operators[op_idx]}{num2}<<calc>>"

        # Calculate result
        if op_idx == 0:  # sum
            result = num1 + num2
        elif op_idx == 1:  # difference
            result = num1 - num2
        elif op_idx == 2:  # product
            result = num1 * num2
        else:  # quotient
            result = num1 / num2

        observation = str(result)
        answer = f"The {operations[op_idx]} of {num1} and {num2} is {result}."

        additional_examples.append(
            {
                "question": question,
                "thought": thought,
                "action": action,
                "observation": observation,
                "answer": answer,
            }
        )

    # Combine all examples
    react_examples.extend(additional_examples)

    # Process into prompt format
    processed_examples = []

    for example in tqdm(react_examples, desc="Processing ReAct examples"):
        # Format as ReAct prompt
        prompt = f"Answer the following question step-by-step using available tools:\n\nQuestion: {example['question']}\n\nThought:"

        # Format complete answer with ReAct steps
        complete_answer = f"Thought: {example['thought']}\n\nAction: {example['action']}\n\nObservation: {example['observation']}\n\nFINAL: {example['answer']}"

        processed_examples.append(
            {"prompt": prompt, "answer": complete_answer, "source": "react"}
        )

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "react_traces.json"), "w") as f:
        json.dump(processed_examples, f, indent=2)

    print(f"Saved {len(processed_examples)} ReAct examples")
    return processed_examples


def prepare_symbolic_math_quiz(config, output_dir):
    """
    Prepare custom Symbolic-Math quiz

    Args:
        config (dict): Configuration dictionary
        output_dir (str): Output directory path

    Returns:
        list: List of processed examples
    """
    print("Preparing Symbolic-Math quiz dataset...")

    # Create 10 symbolic math problems as specified in architecture doc
    symbolic_problems = [
        {
            "question": "Simplify the expression: (x^2 + 2x + 1) - (x^2 - 3x + 2)",
            "answer": "To simplify (x^2 + 2x + 1) - (x^2 - 3x + 2), I'll distribute the negative sign and combine like terms.\n\n(x^2 + 2x + 1) - (x^2 - 3x + 2)\n= x^2 + 2x + 1 - x^2 + 3x - 2\n= (x^2 - x^2) + (2x + 3x) + (1 - 2)\n= 0 + 5x - 1\n= 5x - 1\n\nTherefore, the simplified expression is 5x - 1.",
        },
        {
            "question": "Solve for x: 3(x - 4) = 2(x + 5)",
            "answer": "To solve for x in 3(x - 4) = 2(x + 5):\n\n3(x - 4) = 2(x + 5)\n3x - 12 = 2x + 10\n3x - 2x = 10 + 12\nx = 22\n\nTherefore, x = 22.",
        },
        {
            "question": "Factor the expression: x^2 - 7x + 12",
            "answer": "To factor x^2 - 7x + 12, I need to find two numbers that multiply to 12 and add to -7.\n\nThe factors of 12 are: 1, 2, 3, 4, 6, 12\nThe pairs of factors are: (1, 12), (2, 6), (3, 4)\n\nChecking the sums:\n1 + 12 = 13\n2 + 6 = 8\n3 + 4 = 7\n\nSince I need a sum of -7, I need negative factors: -3 and -4\n-3 + (-4) = -7 and (-3) × (-4) = 12\n\nTherefore: x^2 - 7x + 12 = (x - 3)(x - 4)",
        },
        {
            "question": "Find the derivative of f(x) = 3x^4 - 2x^2 + 5x - 7",
            "answer": "To find the derivative of f(x) = 3x^4 - 2x^2 + 5x - 7, I'll use the power rule: d/dx(x^n) = n·x^(n-1).\n\nf(x) = 3x^4 - 2x^2 + 5x - 7\n\nf'(x) = 3·4·x^(4-1) - 2·2·x^(2-1) + 5·1·x^(1-1) - 0\nf'(x) = 12x^3 - 4x^1 + 5 - 0\nf'(x) = 12x^3 - 4x + 5\n\nTherefore, the derivative is f'(x) = 12x^3 - 4x + 5.",
        },
        {
            "question": "Expand and simplify: (2x + 3)^2",
            "answer": "To expand (2x + 3)^2, I'll use the binomial formula: (a + b)^2 = a^2 + 2ab + b^2\n\n(2x + 3)^2 = (2x)^2 + 2(2x)(3) + 3^2\n= 4x^2 + 12x + 9\n\nTherefore, the expanded and simplified expression is 4x^2 + 12x + 9.",
        },
        {
            "question": "Find all values of x that satisfy the equation: x^2 - x - 6 = 0",
            "answer": "To solve x^2 - x - 6 = 0, I'll use the quadratic formula: x = (-b ± √(b^2 - 4ac))/2a, where a = 1, b = -1, and c = -6.\n\nx = (-(-1) ± √((-1)^2 - 4(1)(-6)))/2(1)\nx = (1 ± √(1 + 24))/2\nx = (1 ± √25)/2\nx = (1 ± 5)/2\n\nSo, x = (1 + 5)/2 = 6/2 = 3 or x = (1 - 5)/2 = -4/2 = -2\n\nTherefore, the solutions are x = 3 and x = -2.",
        },
        {
            "question": "Simplify the expression: (2^3 × 3^2) ÷ (2^2 × 3)",
            "answer": "To simplify (2^3 × 3^2) ÷ (2^2 × 3), I'll use the laws of exponents.\n\n(2^3 × 3^2) ÷ (2^2 × 3)\n= (2^3 ÷ 2^2) × (3^2 ÷ 3^1)\n= 2^(3-2) × 3^(2-1)\n= 2^1 × 3^1\n= 2 × 3\n= 6\n\nTherefore, the simplified expression is 6.",
        },
        {
            "question": "Find the indefinite integral of f(x) = 2x - 3",
            "answer": "To find the indefinite integral of f(x) = 2x - 3, I'll use the basic integration rules:\n∫(ax^n)dx = a(x^(n+1))/(n+1) + C, where C is the constant of integration.\n\n∫(2x - 3)dx = ∫2x dx - ∫3 dx\n= 2∫x dx - 3∫dx\n= 2(x^2/2) - 3x + C\n= x^2 - 3x + C\n\nTherefore, the indefinite integral is F(x) = x^2 - 3x + C, where C is the constant of integration.",
        },
        {
            "question": "Solve the system of equations: 2x + y = 8 and 3x - 2y = 1",
            "answer": "To solve the system of equations:\n2x + y = 8 ... (1)\n3x - 2y = 1 ... (2)\n\nI'll solve for y in equation (1):\ny = 8 - 2x ... (3)\n\nThen substitute this into equation (2):\n3x - 2(8 - 2x) = 1\n3x - 16 + 4x = 1\n7x - 16 = 1\n7x = 17\nx = 17/7\n\nNow I'll find y by substituting x = 17/7 into equation (3):\ny = 8 - 2(17/7)\ny = 8 - 34/7\ny = 56/7 - 34/7\ny = 22/7\n\nTherefore, the solution is x = 17/7 and y = 22/7.",
        },
        {
            "question": "If <define symbol=f(x): x^2 + 3x - 4:>, find f(2) and f(-1).",
            "answer": "I need to evaluate the function f(x) = x^2 + 3x - 4 at x = 2 and x = -1.\n\nFirst, I'll find f(2):\nf(2) = 2^2 + 3(2) - 4\nf(2) = 4 + 6 - 4\nf(2) = 6\n\nNext, I'll find f(-1):\nf(-1) = (-1)^2 + 3(-1) - 4\nf(-1) = 1 - 3 - 4\nf(-1) = -6\n\nTherefore, f(2) = 6 and f(-1) = -6.",
        },
    ]

    # Process examples
    processed_examples = []

    for example in tqdm(symbolic_problems, desc="Processing Symbolic-Math examples"):
        prompt = f"Solve the following symbolic math problem:\n\n{example['question']}"

        processed_examples.append(
            {"prompt": prompt, "answer": example["answer"], "source": "symbolic-math"}
        )

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "symbolic_math_quiz.json"), "w") as f:
        json.dump(processed_examples, f, indent=2)

    print(f"Saved {len(processed_examples)} Symbolic-Math examples")
    return processed_examples


def main():
    """Main function to prepare all datasets"""
    config = load_config()
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare all datasets
    gsm8k_examples = prepare_gsm8k(config, output_dir)
    humaneval_examples = prepare_humaneval(config, output_dir)
    react_examples = prepare_react_traces(config, output_dir)
    symbolic_examples = prepare_symbolic_math_quiz(config, output_dir)

    # Combine all for training
    all_train_examples = gsm8k_examples + react_examples

    # Deduplicate by prompt
    unique_prompts = set()
    deduplicated_examples = []

    for example in all_train_examples:
        if example["prompt"] not in unique_prompts:
            unique_prompts.add(example["prompt"])
            deduplicated_examples.append(example)

    # Save combined training data
    with open(os.path.join(output_dir, "training_data.json"), "w") as f:
        json.dump(deduplicated_examples, f, indent=2)

    print(f"Saved {len(deduplicated_examples)} combined training examples")

    # Create symlinks for evaluation
    eval_dir = os.path.join("data", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Create symbolic links for evaluation datasets
    for filename in [
        "gsm8k_10pct.json",
        "humaneval_mini.json",
        "symbolic_math_quiz.json",
    ]:
        src_path = os.path.join(output_dir, filename.replace("_10pct", ""))
        dst_path = os.path.join(eval_dir, filename)

        # Create symlink if source exists
        if os.path.exists(src_path):
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.symlink(os.path.relpath(src_path, os.path.dirname(dst_path)), dst_path)
            print(f"Created symlink for {filename}")

    print("Data preparation complete")


if __name__ == "__main__":
    main()
