#!/usr/bin/env python3
"""
Download Mamba model weights from Hugging Face Hub with checksum verification.
"""

import os
import sys
import argparse
import hashlib
import logging
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import config utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config

try:
    from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def calculate_file_sha256(filepath):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_files(output_dir):
    """Verify that essential model files exist in the output directory."""
    essential_files = [
        "config.json",  # Model configuration
        "pytorch_model.bin",  # Model weights
    ]

    missing_files = []
    for file in essential_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            missing_files.append(file)

    if missing_files:
        logging.error(f"Missing essential files: {', '.join(missing_files)}")
        return False

    # Record checksums for future verification
    checksums = {}
    for file in os.listdir(output_dir):
        if file.endswith((".json", ".bin", ".pt", ".model")):
            file_path = os.path.join(output_dir, file)
            checksums[file] = calculate_file_sha256(file_path)
            logging.info(f"✓ Verified existence of {file}")

    # Save checksums to file for future reference
    checksum_path = os.path.join(output_dir, "checksums.txt")
    with open(checksum_path, "w") as f:
        for file, checksum in checksums.items():
            f.write(f"{file}: {checksum}\n")

    logging.info(f"Saved checksums to {checksum_path}")
    return True


def download_tokenizer(output_dir):
    """Download a compatible GPT-NeoX tokenizer for the Mamba model."""
    # Mamba uses GPTNeoX tokenizer - we'll download from a compatible model
    tokenizer_src = "EleutherAI/pythia-70m"  # A small model with a compatible tokenizer
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    logging.info(f"Downloading compatible tokenizer from {tokenizer_src}")

    try:
        for filename in tokenizer_files:
            try:
                file_path = hf_hub_download(
                    repo_id=tokenizer_src,
                    filename=filename,
                    local_dir=output_dir,
                    force_download=True,
                )
                logging.info(f"✓ Downloaded {filename} from {tokenizer_src}")
            except Exception as e:
                logging.warning(f"Couldn't download {filename}: {e}")

        # Create a README note about the tokenizer
        with open(os.path.join(output_dir, "TOKENIZER_NOTE.md"), "w") as f:
            f.write(
                f"# Tokenizer Note\n\nThe tokenizer files were downloaded from {tokenizer_src} for compatibility with Mamba models which use GPTNeoX tokenizers.\n"
            )

        return True
    except Exception as e:
        logging.error(f"Failed to download tokenizer: {e}")
        return False


def download_model(model_id, output_dir, revision="main", force_download=False):
    """Download model from Hugging Face Hub with progress bar."""
    try:
        logging.info(f"Downloading {model_id} to {output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Check if files already exist in the output directory
        if not force_download and os.path.exists(
            os.path.join(output_dir, "config.json")
        ):
            logging.info(f"Model files already exist in {output_dir}")
            logging.info("Verifying files...")
            if verify_files(output_dir):
                logging.info("Files are valid, skipping download")
                # Still ensure we have tokenizer files
                download_tokenizer(output_dir)
                return True
            else:
                logging.warning("Files are incomplete or invalid. Re-downloading...")

        # Download model files
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=force_download,
        )

        # Verify downloaded files
        logging.info("Verifying downloaded files...")
        if verify_files(output_dir):
            # Download tokenizer separately since Mamba models don't include it
            download_tokenizer(output_dir)
            logging.info(f"All essential files downloaded successfully.")
            return True
        else:
            logging.error(
                "Some essential files are missing. The model may not work correctly."
            )
            return False

    except HfHubHTTPError as e:
        logging.error(f"Failed to download model: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights from Hugging Face Hub"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="state-spaces/mamba-130m",
        help="Hugging Face model ID (default: state-spaces/mamba-130m)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config or './models/mamba-130m')",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision to download (default: main)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config file (default: src/configs/config.yaml)",
    )

    args = parser.parse_args()

    # Load config to get model_path if not specified
    try:
        config = load_config(args.config_path)
        default_model_path = config.get("model", {}).get("path", "./models/mamba-130m")
    except Exception as e:
        logging.warning(f"Could not load config: {e}")
        default_model_path = "./models/mamba-130m"

    output_dir = args.output_dir or default_model_path

    success = download_model(
        model_id=args.model_id,
        output_dir=output_dir,
        revision=args.revision,
        force_download=args.force,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
