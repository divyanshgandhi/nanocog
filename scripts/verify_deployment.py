#!/usr/bin/env python3
"""
Nano-Cog 0.1 - Deployment Verification Script
Checks the environment and GPU compatibility for deploying Nano-Cog
"""

import os
import sys
import torch
import subprocess
import platform
from tqdm import tqdm
import importlib.util


def check_environment():
    """Check system environment and dependencies"""
    print("=" * 70)
    print("DEPLOYMENT ENVIRONMENT VERIFICATION")
    print("=" * 70)

    # System info
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")

    # Check CUDA availability
    print("\nGPU CONFIGURATION:")
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: Yes")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"    - Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
            )

        # Test CUDA tensor creation
        try:
            x = torch.rand(100, 100, device="cuda")
            del x
            print("✓ CUDA Tensor Test: Passed")
        except Exception as e:
            print(f"✗ CUDA Tensor Test Failed: {e}")
    else:
        print("✗ CUDA Available: No")

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✓ Apple Silicon MPS Available: Yes")
        try:
            x = torch.rand(100, 100, device="mps")
            del x
            print("✓ MPS Tensor Test: Passed")
        except Exception as e:
            print(f"✗ MPS Tensor Test Failed: {e}")
    else:
        print("✗ Apple Silicon MPS: Not Available")

    # Check CPU
    try:
        x = torch.rand(1000, 1000)
        del x
        print("✓ CPU Tensor Test: Passed")
    except Exception as e:
        print(f"✗ CPU Tensor Test Failed: {e}")

    # Check PyTorch version
    print(f"\nPYTORCH CONFIGURATION:")
    print(f"PyTorch Version: {torch.__version__}")

    # Check for required libraries
    print("\nDEPENDENCY CHECKS:")
    required_libs = ["transformers", "peft", "tqdm", "numpy", "chromadb"]

    for lib in required_libs:
        spec = importlib.util.find_spec(lib)
        if spec is not None:
            try:
                module = importlib.import_module(lib)
                version = getattr(module, "__version__", "unknown")
                print(f"✓ {lib}: {version}")
            except ImportError:
                print(f"✗ {lib}: Not properly installed")
        else:
            print(f"✗ {lib}: Not found")

    # Check for Mamba-specific dependencies
    try:
        import causal_conv1d

        print(f"✓ causal_conv1d: {getattr(causal_conv1d, '__version__', 'unknown')}")
    except ImportError:
        print("✗ causal_conv1d: Not installed")

    try:
        import mamba_ssm

        print(f"✓ mamba_ssm: {getattr(mamba_ssm, '__version__', 'unknown')}")
    except ImportError:
        print("✗ mamba_ssm: Not installed")

    # Check disk space
    try:
        if platform.system() == "Windows":
            total, used, free = [
                int(val) / (1024**3)
                for val in os.popen(
                    "wmic logicaldisk where DeviceID='C:' get Size,FreeSpace"
                )
                .read()
                .strip()
                .split()[1:]
            ]
        else:
            stat = os.statvfs("/")
            total = stat.f_blocks * stat.f_frsize / (1024**3)
            free = stat.f_bavail * stat.f_frsize / (1024**3)
            used = total - free

        print(f"\nDISK SPACE:")
        print(f"Total: {total:.2f} GB")
        print(f"Used: {used:.2f} GB")
        print(f"Free: {free:.2f} GB")

        if free < 5:
            print("⚠️ Warning: Less than 5GB free disk space")
        else:
            print("✓ Sufficient disk space available")
    except Exception as e:
        print(f"Could not check disk space: {e}")

    # RAM check
    try:
        import psutil

        vm = psutil.virtual_memory()
        print(f"\nMEMORY:")
        print(f"Total: {vm.total / (1024**3):.2f} GB")
        print(f"Available: {vm.available / (1024**3):.2f} GB")
        if vm.available / (1024**3) < 8:
            print("⚠️ Warning: Less than 8GB available RAM")
        else:
            print("✓ Sufficient RAM available")
    except ImportError:
        print("Could not check RAM (psutil not installed)")


def test_model_loading():
    """Test loading the Nano-Cog model"""
    print("\n" + "=" * 70)
    print("MODEL LOADING TEST")
    print("=" * 70)

    try:
        # Add parent directory to path if running from scripts/
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        print("Attempting to import NanoCogModel...")
        try:
            from src.core.model import NanoCogModel

            print("✓ Successfully imported NanoCogModel")

            print("\nAttempting to load model...")
            model = NanoCogModel()
            print("✓ Successfully initialized model")

            print("\nTrying a test generation...")
            output = model.generate("What is the square root of 16?", max_length=50)
            print(f"\nTest output: {output}")
            print("✓ Successfully generated text")

        except ImportError as e:
            print(f"✗ Failed to import NanoCogModel: {e}")
        except Exception as e:
            print(f"✗ Failed to load or run model: {e}")
    except Exception as e:
        print(f"✗ Test failed: {e}")


def main():
    """Main function"""
    check_environment()
    test_model_loading()

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
