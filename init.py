#!/usr/bin/env python3
"""
Nano-Cog 0.1 - Initialization Script
Sets up the environment for Nano-Cog
"""

import os
import sys
import argparse
import subprocess
from tqdm import tqdm


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)

    print(
        f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected"
    )


def create_directories():
    """Create required directories"""
    directories = ["models", "data", "data/processed", "data/evaluation", "results"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("Installing dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("Error: Failed to install dependencies.")
        sys.exit(1)


def download_model():
    """
    Download Mamba model weights using the download_weights.py script
    """
    print("Downloading Mamba model weights...")

    try:
        # Check if the script exists
        script_path = os.path.join("scripts", "download_weights.py")
        if not os.path.exists(script_path):
            print(f"Error: Download script not found at {script_path}")
            # Create scripts directory if it doesn't exist
            os.makedirs("scripts", exist_ok=True)
            print("Creating minimal download script...")
            with open(script_path, "w") as f:
                f.write(
                    """#!/usr/bin/env python3
import os
import sys
from huggingface_hub import snapshot_download

def main():
    model_id = "state-spaces/mamba-130m"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    
    output_dir = os.path.join("models", model_id.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {model_id} to {output_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )
    print(f"Download complete.")

if __name__ == "__main__":
    main()
"""
                )
            print(f"Created minimal download script at {script_path}")

        # Make script executable
        os.chmod(script_path, 0o755)

        # Run the download script
        model_id = "state-spaces/mamba-130m"
        result = subprocess.run(
            [sys.executable, script_path, "--model-id", model_id],
            check=True,
        )

        if result.returncode == 0:
            model_path = os.path.join("models", model_id.split("/")[-1])
            # Verify that critical files exist
            config_path = os.path.join(model_path, "config.json")
            model_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(config_path) and os.path.exists(model_path):
                print("✓ Model downloaded successfully and verified")
            else:
                print("⚠ Model files may be incomplete. Please verify manually.")
        else:
            print("Error: Failed to download model. Check logs for details.")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to download model: {e}")
        print("You may need to download the model manually.")
        print("For Mamba-130M, visit: https://huggingface.co/state-spaces/mamba-130m")
        print("Then place the files in the models/mamba-130m directory")
        sys.exit(1)


def prepare_data():
    """Prepare training and evaluation data"""
    print("Preparing data...")

    try:
        subprocess.run([sys.executable, "src/data/prepare_data.py"], check=True)
        print("✓ Data preparation completed")
    except subprocess.CalledProcessError:
        print("Error: Failed to prepare data.")
        sys.exit(1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Nano-Cog 0.1 Initialization Script")
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )
    parser.add_argument("--skip-model", action="store_true", help="Skip model download")
    parser.add_argument(
        "--skip-data", action="store_true", help="Skip data preparation"
    )
    parser.add_argument(
        "--force-model",
        action="store_true",
        help="Force model re-download even if files exist",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Initializing Nano-Cog 0.1...")
    print("=" * 70)

    # Check Python version
    check_python_version()

    # Create directories
    create_directories()

    # Install dependencies
    if not args.skip_deps:
        install_dependencies()
    else:
        print("Skipping dependency installation")

    # Download model
    if not args.skip_model:
        # Pass force flag if specified
        if args.force_model:
            os.environ["FORCE_MODEL_DOWNLOAD"] = "1"
        download_model()
    else:
        print("Skipping model download")
        # Even if skipping download, verify model exists
        if not os.path.exists("models/mamba-130m/config.json"):
            print(
                "Warning: Model files not found. The application may not work correctly."
            )
            print(
                "Run without --skip-model to download the model, or download manually."
            )

    # Prepare data
    if not args.skip_data:
        prepare_data()
    else:
        print("Skipping data preparation")

    print("\n" + "=" * 70)
    print("Initialization complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. CLI interface: python main.py")
    print("2. Web interface: streamlit run app.py")
    print("3. Run evaluation: python -m src.evaluation.evaluate")
    print("=" * 70)


if __name__ == "__main__":
    main()
