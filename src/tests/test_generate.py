#!/usr/bin/env python3
"""
Smoke test for Nano-Cog generation
Tests that the model can generate a basic response
"""

import os
import sys
import pytest

# Add the parent directory to the path to import modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.core.model import NanoCogModel


def test_simple_generation():
    """Test that the model can generate a simple response"""
    # Initialize model with minimal config
    try:
        model = NanoCogModel()

        # Generate text from a simple prompt
        prompt = "Hello, how are you?"
        response = model.generate(prompt, max_new_tokens=5)

        # Check that response contains the prompt (echo) and some additional text
        assert prompt in response
        assert len(response) > len(prompt)
        print(f"Generated response: {response}")

    except Exception as e:
        # Convert assertion error to proper pytest failure
        pytest.fail(f"Generation test failed: {str(e)}")


if __name__ == "__main__":
    # Run the test directly
    print("Running simple generation test...")
    test_simple_generation()
    print("Test passed!")
