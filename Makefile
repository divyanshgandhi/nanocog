# Nano-Cog 0.1 Makefile

.PHONY: init setup test train evaluate run ui clean help lint format

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
CONFIG := src/configs/config.yaml

# Initialize environment
init:
	@echo "Initializing Nano-Cog environment..."
	@$(PYTHON) init.py

# Setup without downloading model weights
setup:
	@echo "Setting up Nano-Cog environment (without model download)..."
	@$(PYTHON) init.py --skip-model

# Run tests
test:
	@echo "Running tests..."
	@$(PYTHON) src/tests/test_model.py

# Train model
train:
	@echo "Training model..."
	@$(PYTHON) src/training/train.py --config $(CONFIG)

# Train supervised only
train-supervised:
	@echo "Training model (supervised only)..."
	@$(PYTHON) src/training/train.py --config $(CONFIG) --skip-toolformer --skip-rl

# Train toolformer only
train-toolformer:
	@echo "Training model (toolformer only)..."
	@$(PYTHON) src/training/train.py --config $(CONFIG) --skip-supervised --skip-rl

# Train RL only
train-rl:
	@echo "Training model (RL only)..."
	@$(PYTHON) src/training/train.py --config $(CONFIG) --skip-supervised --skip-toolformer

# Evaluate model
evaluate:
	@echo "Evaluating model..."
	@$(PYTHON) -m src.evaluation.evaluate --config $(CONFIG)

# Run CLI
run:
	@echo "Running Nano-Cog CLI..."
	@$(PYTHON) main.py --config $(CONFIG)

# Run UI
ui:
	@echo "Running Nano-Cog UI..."
	@streamlit run app.py

# Prepare data
data:
	@echo "Preparing data..."
	@$(PYTHON) src/data/prepare_data.py

# Lint code with ruff
lint:
	@echo "Linting code with ruff..."
	@pip install ruff > /dev/null
	@ruff check .

# Format code with black
format:
	@echo "Formatting code with black..."
	@pip install black > /dev/null
	@black .

# Clean up
clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".DS_Store" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name "*.egg" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

# Display help
help:
	@echo "Nano-Cog 0.1 - A laptop-scale language agent with high reasoning efficiency"
	@echo ""
	@echo "Usage:"
	@echo "  make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  init             Initialize environment and download model weights"
	@echo "  setup            Setup environment without downloading model weights"
	@echo "  test             Run tests"
	@echo "  train            Train full model (supervised → toolformer → RL)"
	@echo "  train-supervised Train supervised only (LoRA fine-tuning)"
	@echo "  train-toolformer Train toolformer only"
	@echo "  train-rl         Train RL only (GRPO)"
	@echo "  evaluate         Evaluate model"
	@echo "  run              Run CLI interface"
	@echo "  ui               Run Streamlit UI"
	@echo "  data             Prepare training data"
	@echo "  lint             Lint code with ruff"
	@echo "  format           Format code with black"
	@echo "  clean            Clean up temporary files"
	@echo "  help             Display this help message" 