# Nano-Cog 0.1

A laptop-scale language agent demonstrating high reasoning-per-FLOP efficiency by combining state-space sequence modelling (Mamba), retrieval, tool use, and reinforcement-tuned chain-of-thought (CoT).

## Architecture

Nano-Cog 0.1 is built on:
- **Mamba-130M** backbone model (state-space sequence model)
- **LoRA Reasoning Adapters** (+8M params)
- **Mini-MoE FFN Router** (+2M params)
- **Dynamic Symbol Engine** for scratch-pad tokens
- **Retrieval-Aware Prompt Composer** using ChromaDB
- **Tool Dispatcher** for calculator, Python, and Bash operations

## Setup

### Requirements
- Python 3.10+
- 16GB RAM (11GB minimum for M1)
- 5GB disk space

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourlabname/nano-cog.git
cd nano-cog
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the environment (downloads model weights):
```bash
python scripts/setup.py
```

## Known Issues

### Mamba Models on Apple Silicon

There's a known issue running Mamba models on Apple Silicon (M1/M2/M3) Macs:

- **Meta Tensor Error**: The model may load in "meta" tensor state which prevents inference.
- **Root Cause**: This appears to be related to the interaction between PyTorch's Metal backend (MPS) and Mamba's state-space architecture.

If you encounter an error like:
```
Error: Cannot copy out of meta tensor; no data!
```

Try these workarounds:
1. Use a smaller model by editing `config.yaml` (reducing number of layers)
2. Use CPU mode by setting `device = torch.device("cpu")` in the model.py file
3. Try a different model architecture that is better supported on Metal

Alternatively, the issue may be fixed in future versions of PyTorch or Mamba.

## Usage

Run the interactive CLI:
```bash
python main.py
```

For web UI:
```bash
python ui/app.py
```

## Training

Training scripts are provided to fine-tune the model on custom data:

```bash
# Supervised fine-tuning
make train-supervised

# Tool-use fine-tuning
make train-toolformer 

# RLHF fine-tuning
make train-rl
```

## License

[MIT License](LICENSE)

## Development Timeline

See [architecture.md](architecture.md) for the full development timeline and milestones.

## Citation

If you use Nano-Cog in your research, please cite:
```
@software{nano-cog2025,
  author = {Your Lab Name},
  title = {Nano-Cog: A Laptop-scale Language Agent},
  year = {2025},
  url = {https://github.com/yourlabname/nano-cog}
}
``` 