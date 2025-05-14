#!/usr/bin/env python3
"""
Device consistency test for Nano-Cog
Tests that the model properly handles devices and normalizes CUDA device representations
"""

import os
import sys
import pytest
import torch

# Add the parent directory to the path to import modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.core.model import NanoCogModel


def test_device_consistency():
    """Test that device representations are consistent and normalized"""
    # Skip test if no CUDA available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping device consistency test")

    try:
        # Initialize model with minimal config
        model = NanoCogModel()

        # Check that the device is properly normalized
        assert model.device.type == "cuda"
        assert model.device.index == 0

        # Check that model parameters are on the same device
        for name, param in model.model.named_parameters():
            if not param.is_meta:  # Skip meta tensors
                assert param.device.type == "cuda"
                assert param.device.index == 0

        print("Device consistency check passed!")
    except Exception as e:
        pytest.fail(f"Device consistency test failed: {str(e)}")


if __name__ == "__main__":
    # Run the test directly
    if torch.cuda.is_available():
        print("Running device consistency test...")
        test_device_consistency()
        print("Test passed!")
    else:
        print("CUDA not available, skipping device consistency test")
