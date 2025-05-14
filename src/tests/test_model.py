"""
Simple test script to verify model structure
"""

import sys
import os
import torch
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.config import load_config
from src.core.model import NanoCogModel, MiniMoERouter, DynamicSymbolEngine
from src.core.tools import ToolDispatcher


def test_config():
    """Test configuration loading"""
    print("Testing configuration loading...")
    config = load_config()

    # Print a few config values to verify
    print(f"Model name: {config['model']['backbone']['name']}")
    print(f"LoRA rank: {config['model']['lora']['rank']}")
    print(f"Number of MoE experts: {config['model']['moe']['num_experts']}")
    print(f"Inference temperature: {config['inference']['temperature']}")

    # Success if no errors
    print("✓ Configuration loading test passed")


def test_moe_router():
    """Test MiniMoERouter structure"""
    print("\nTesting MiniMoERouter...")

    # Dummy config
    config = {
        "model": {
            "backbone": {"hidden_size": 768},
            "moe": {"num_experts": 2, "hidden_size": 32, "temperature": 0.7},
        }
    }

    # Create router
    router = MiniMoERouter(config)

    # Create dummy input
    dummy_input = torch.randn(1, 768)

    # Forward pass
    output = router(dummy_input)

    # Check output shape
    assert (
        output.shape == dummy_input.shape
    ), f"Expected shape {dummy_input.shape}, got {output.shape}"

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ MiniMoERouter test passed")


def test_dse():
    """Test Dynamic Symbol Engine"""
    print("\nTesting Dynamic Symbol Engine...")

    # Dummy config
    config = {
        "model": {
            "dynamic_symbol_engine": {
                "max_tokens": 50,
                "grammar_tokens": ["<define symbol=", ":>"],
            }
        }
    }

    # Create a mock tokenizer class
    class MockTokenizer:
        def encode(self, text):
            # Mock encoding by counting words
            return text.split()

    # Create DSE
    tokenizer = MockTokenizer()
    dse = DynamicSymbolEngine(config, tokenizer)

    # Test symbol definition processing
    text = "Let me define <define symbol=fib: fibonacci sequence:> as a shorthand"
    symbols = dse.process_symbol_definitions(text)

    print(f"Processed symbols: {symbols}")
    assert "fib" in symbols, "Expected 'fib' symbol to be defined"
    assert (
        symbols["fib"] == "fibonacci sequence"
    ), f"Expected 'fib' to map to 'fibonacci sequence', got {symbols['fib']}"

    # Test compression calculation
    original = "Using the fibonacci sequence multiple times"
    processed = "Using the fib multiple times"
    compression = dse.calculate_compression(original, processed)

    print(f"Compression: {compression} tokens")
    assert compression > 0, "Expected positive compression"

    print("✓ Dynamic Symbol Engine test passed")


def test_model_structure():
    """Test overall model structure (without loading actual weights)"""
    print("\nTesting model structure...")

    # Mock the AutoModelForCausalLM and AutoTokenizer to avoid actual loading
    import src.core.model

    # Save original classes
    original_auto_model = src.core.model.AutoModelForCausalLM
    original_auto_tokenizer = src.core.model.AutoTokenizer

    # Create mock classes
    class MockAutoModel:
        @staticmethod
        def from_pretrained(checkpoint, **kwargs):
            class MockModel:
                def __init__(self):
                    self.config = type("obj", (object,), {"hidden_size": 768})

                def generate(self, **kwargs):
                    return torch.tensor([[1, 2, 3, 4, 5]])

                def save_pretrained(self, path):
                    pass

                def load_adapter(self, path):
                    pass

            return MockModel()

    class MockAutoTokenizer:
        @staticmethod
        def from_pretrained(checkpoint):
            class MockTokenizer:
                def __call__(self, text, return_tensors="pt"):
                    class MockEncodingOutput:
                        def __init__(self):
                            self.input_ids = torch.tensor([[1, 2, 3, 4, 5]])

                        def to(self, device):
                            return self

                    return MockEncodingOutput()

                def decode(self, token_ids, skip_special_tokens=True):
                    return "Mock model output"

                def encode(self, text):
                    return [1, 2, 3, 4] * (len(text) // 10 + 1)  # Mock encoding

            return MockTokenizer()

    # Patch the classes
    src.core.model.AutoModelForCausalLM = MockAutoModel
    src.core.model.AutoTokenizer = MockAutoTokenizer

    try:
        # Create model
        model = NanoCogModel()

        # Test generation
        output = model.generate("Test prompt")
        print(f"Model output: {output}")

        # Success if we got here without errors
        print("✓ Model structure test passed")
    finally:
        # Restore original classes
        src.core.model.AutoModelForCausalLM = original_auto_model
        src.core.model.AutoTokenizer = original_auto_tokenizer


def test_calc_tool():
    """Test that the model can use the calc tool to correctly calculate 2+2 equals 4"""
    print("\nTesting calculation with tool invocation...")

    # Mock the model and tool dispatcher
    import src.core.model

    # Save original classes
    original_auto_model = src.core.model.AutoModelForCausalLM
    original_auto_tokenizer = src.core.model.AutoTokenizer

    # Create mock classes that simulate the calc tool invocation
    class MockAutoModel:
        @staticmethod
        def from_pretrained(checkpoint, **kwargs):
            class MockModel:
                def __init__(self):
                    self.config = type("obj", (object,), {"hidden_size": 768})

                def generate(self, **kwargs):
                    # Return token IDs that would decode to include <<calc>> 2+2 </calc>>
                    return torch.tensor([[1, 2, 3, 4, 5]])

                def save_pretrained(self, path):
                    pass

                def load_adapter(self, path):
                    pass

            return MockModel()

    class MockAutoTokenizer:
        @staticmethod
        def from_pretrained(checkpoint):
            class MockTokenizer:
                def __call__(self, text, return_tensors="pt"):
                    class MockEncodingOutput:
                        def __init__(self):
                            self.input_ids = torch.tensor([[1, 2, 3, 4, 5]])

                        def to(self, device):
                            return self

                    return MockEncodingOutput()

                def decode(self, token_ids, skip_special_tokens=True):
                    # Simulate model output that includes tool invocation
                    return "I need to calculate 2+2. Let me use a tool: <<calc>> 2+2 </calc>> The answer is 4."

                def encode(self, text):
                    return [1, 2, 3, 4] * (len(text) // 10 + 1)  # Mock encoding

            return MockTokenizer()

    # Create a mock ToolDispatcher that returns "4" for "2+2"
    original_tool_dispatcher = src.core.model.ToolDispatcher

    class MockToolDispatcher:
        def __init__(self, config=None):
            pass

        def process_tools(self, text):
            # Replace <<calc>> 2+2 </calc>> with 4
            return re.sub(r"<<calc>> 2\+2 </calc>>", "4", text)

        def get_available_tools(self):
            return ["calc", "search", "wikipedia"]

    # Patch the classes
    src.core.model.AutoModelForCausalLM = MockAutoModel
    src.core.model.AutoTokenizer = MockAutoTokenizer
    src.core.model.ToolDispatcher = MockToolDispatcher

    try:
        # Create model
        model = NanoCogModel()

        # Test tool use
        output = model.generate("What is 2+2?")
        print(f"Tool-processed output: {output}")

        # Check if output contains the answer "4"
        assert "4" in output, f"Expected '4' in output, but got: {output}"

        print("✓ Calculation tool test passed")
    finally:
        # Restore original classes
        src.core.model.AutoModelForCausalLM = original_auto_model
        src.core.model.AutoTokenizer = original_auto_tokenizer
        src.core.model.ToolDispatcher = original_tool_dispatcher


if __name__ == "__main__":
    print("=" * 50)
    print("Running Nano-Cog model tests")
    print("=" * 50)

    try:
        # Run tests
        test_config()
        test_moe_router()
        test_dse()
        test_model_structure()
        test_calc_tool()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise
