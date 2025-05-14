"""
Nano-Cog 0.1 Core Model Implementation
"""

import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import load_config


class MiniMoERouter(torch.nn.Module):
    """
    Mini Mixture of Experts router with 2 experts
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["model"]["moe"]["hidden_size"]
        self.input_size = config["model"]["backbone"]["hidden_size"]
        self.num_experts = config["model"]["moe"]["num_experts"]
        self.temperature = config["model"]["moe"]["temperature"]

        # Expert networks
        self.experts = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.input_size),
                )
                for _ in range(self.num_experts)
            ]
        )

        # Router network
        self.router = torch.nn.Linear(self.input_size, self.num_experts)

    def forward(self, x):
        # Compute routing probabilities
        routing_logits = self.router(x) / self.temperature
        routing_probs = torch.nn.functional.softmax(routing_logits, dim=-1)

        # Top-1 routing
        _, indices = torch.topk(routing_probs, k=1, dim=-1)

        # Process inputs through selected expert
        outputs = torch.zeros_like(x)
        for i in range(self.num_experts):
            # Create a mask for selecting this expert
            mask = (indices == i).float()
            # Only process inputs that route to this expert
            expert_inputs = x * mask
            expert_outputs = self.experts[i](expert_inputs)
            outputs = outputs + expert_outputs

        return outputs


class DynamicSymbolEngine(torch.nn.Module):
    """
    Manages dynamic symbol definitions for compression
    """

    def __init__(self, config, tokenizer):
        super().__init__()
        self.max_tokens = config["model"]["dynamic_symbol_engine"]["max_tokens"]
        self.grammar_tokens = config["model"]["dynamic_symbol_engine"]["grammar_tokens"]
        self.tokenizer = tokenizer
        self.symbols = {}

    def process_symbol_definitions(self, text):
        """Process and extract symbol definitions from text"""
        # Simple placeholder implementation
        start_token = self.grammar_tokens[0]
        end_token = self.grammar_tokens[1]

        if start_token in text and end_token in text:
            parts = text.split(start_token)
            for part in parts[1:]:  # Skip the first part before any symbol
                if end_token in part:
                    symbol_def = part.split(end_token)[0]
                    symbol_parts = symbol_def.split(":")
                    if len(symbol_parts) >= 2:
                        symbol_name = symbol_parts[0].strip()
                        symbol_value = symbol_parts[1].strip()
                        self.symbols[symbol_name] = symbol_value

        return self.symbols

    def calculate_compression(self, original_text, processed_text):
        """Calculate token compression savings"""
        original_tokens = len(self.tokenizer.encode(original_text))
        processed_tokens = len(self.tokenizer.encode(processed_text))
        return max(0, original_tokens - processed_tokens)


class NanoCogModel:
    """
    Main Nano-Cog model class that integrates all components
    """

    def __init__(self, config_path=None):
        self.config = load_config(config_path)

        # Enhanced device detection with better fallback
        self.device = self._get_optimal_device()
        print(f"Selected device: {self.device}")

        # Get model checkpoint path
        model_checkpoint = self.config["model"]["backbone"]["checkpoint"]
        # For local files, resolve the path
        if os.path.exists(model_checkpoint):
            model_path = model_checkpoint
        else:
            # Default to models directory
            model_path = os.path.join("models", model_checkpoint.split("/")[-1])
            if not os.path.exists(model_path):
                model_path = os.path.join("models", "mamba-130m")

        # Try to load tokenizer from local files first
        try:
            print(f"Attempting to load tokenizer from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Successfully loaded tokenizer from local path")
        except Exception as e:
            print(f"Could not load tokenizer from {model_path}: {e}")

            # Fallback to the HuggingFace hub directly
            print(f"Attempting to load tokenizer from HuggingFace hub...")
            try:
                # Try to load from hub with a compatible GPTNeoX tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "EleutherAI/pythia-70m", revision="main"
                )
                print(f"Successfully loaded tokenizer from EleutherAI/pythia-70m")
            except Exception as e2:
                print(f"Failed to load tokenizer from hub: {e2}")
                raise RuntimeError(f"Could not load tokenizer: {e}, {e2}")

        # Get tokenizer vocab size
        vocab_size = len(self.tokenizer)
        print(f"Tokenizer vocabulary size: {vocab_size}")

        # Load model configuration to adapt it to our tokenizer if needed
        try:
            model_config = AutoConfig.from_pretrained(model_path)
            orig_vocab_size = getattr(model_config, "vocab_size", None)

            if orig_vocab_size is not None and orig_vocab_size != vocab_size:
                print(
                    f"Model vocabulary size ({orig_vocab_size}) differs from tokenizer ({vocab_size})"
                )
                print("Will adjust model configuration to match tokenizer")
                model_config.vocab_size = vocab_size
            else:
                model_config = None
        except Exception as e:
            print(f"Could not load model config: {e}")
            model_config = None

        # Load model
        print(f"Loading Mamba backbone from {model_checkpoint}...")

        # Set quantization based on platform
        use_quantization = self.config["model"]["backbone"]["quantization"] == "4bit"
        torch_dtype = torch.float16

        # Detect if we're on macOS / Apple Silicon
        is_mac = sys.platform == "darwin"
        if is_mac and torch.backends.mps.is_available():
            print("Apple Silicon detected, using Metal backend")
            # Metal doesn't fully support 4-bit quantization yet
            use_quantization = False
            # Set precision for Metal
            torch.set_float32_matmul_precision("high")
            print("Using high precision for float32 matmul")

        try:
            # Try loading from local path first
            model_kwargs = {"torch_dtype": torch_dtype}

            # Add the adjusted config if needed
            if model_config is not None:
                model_kwargs["config"] = model_config
                print("Using adjusted model configuration")

            # Only add quantization on CUDA platforms
            if use_quantization and torch.cuda.is_available():
                print("Using 4-bit quantization")
                try:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype
                    )
                    model_kwargs["quantization_config"] = quantization_config
                except ImportError:
                    print(
                        "BitsAndBytes not available, falling back to standard loading"
                    )
                    model_kwargs["load_in_4bit"] = True
            else:
                print(f"Quantization disabled, using {torch_dtype}")

            # Add low CPU memory usage for model loading
            model_kwargs["low_cpu_mem_usage"] = True

            # Ignore missing or mismatched weights
            model_kwargs["ignore_mismatched_sizes"] = True

            # Configure device map for optimal loading
            if self.device.type == "cuda" and torch.cuda.is_available():
                print(f"Loading model directly on CUDA device: {self.device}")
                # Use device_map='auto' for CUDA to enable faster loading
                model_kwargs["device_map"] = "auto"
            else:
                # For CPU and other devices, disable device_map
                model_kwargs["device_map"] = None
                print(f"Loading model on {self.device}")

            # Load the model directly to the target device when possible
            if os.path.exists(os.path.join(model_path, "config.json")):
                if "config" not in model_kwargs:
                    # Load config from local path (needed for ignore_mismatched_sizes)
                    model_config = AutoConfig.from_pretrained(model_path)
                    model_kwargs["config"] = model_config

                # Add environment variable settings for bitsandbytes CUDA compatibility
                if self.device.type == "cuda" and use_quantization:
                    # Set LD_LIBRARY_PATH for CUDA discovery by bitsandbytes
                    cuda_lib_path = "/usr/local/cuda/lib64"
                    if os.path.exists(cuda_lib_path):
                        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                        if cuda_lib_path not in current_ld_path:
                            os.environ["LD_LIBRARY_PATH"] = (
                                f"{cuda_lib_path}:{current_ld_path}"
                            )
                            print(
                                f"Added {cuda_lib_path} to LD_LIBRARY_PATH for bitsandbytes"
                            )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, **model_kwargs
                )
                print(f"Successfully loaded model from local path: {model_path}")
            else:
                # Fallback to HuggingFace hub
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_checkpoint, **model_kwargs
                )
                print(f"Successfully loaded model from HuggingFace hub")

            # Check if model is successfully loaded and not on meta device
            if hasattr(self.model, "device") and str(self.model.device) == "meta":
                print(
                    "WARNING: Model loaded to meta device - attempting fallback without quantization"
                )
                # Remove quantization config and try again
                if "quantization_config" in model_kwargs:
                    del model_kwargs["quantization_config"]
                if "load_in_4bit" in model_kwargs:
                    del model_kwargs["load_in_4bit"]

                print("Retrying model loading without quantization...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        (
                            model_path
                            if os.path.exists(os.path.join(model_path, "config.json"))
                            else model_checkpoint
                        ),
                        **model_kwargs,
                    )
                    print("Successfully loaded model without quantization")
                except Exception as e:
                    print(f"Fallback loading also failed: {e}")
                    raise

            # Check if we need to resize token embeddings
            if self.model.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
                print(
                    f"Resizing token embeddings from {self.model.get_input_embeddings().weight.shape[0]} to {len(self.tokenizer)}"
                )
                self.model.resize_token_embeddings(len(self.tokenizer))

            # Move model to target device
            success = self._move_model_to_device()
            if not success:
                print("WARNING: Failed to move model to any device properly")

        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        # Apply LoRA adapters if available
        self._setup_lora_adapters()

        # Initialize MoE router
        self.moe_router = MiniMoERouter(self.config)

        # Initialize Dynamic Symbol Engine
        self.dse = DynamicSymbolEngine(self.config, self.tokenizer)

    def _get_optimal_device(self):
        """Determine the best available device with verification"""
        print("Detecting available compute devices...")

        # CUDA (NVIDIA GPU) - First priority
        if torch.cuda.is_available():
            try:
                # Verify CUDA works by creating a small tensor
                test_tensor = torch.zeros(1, 1, device="cuda")
                cuda_device_name = torch.cuda.get_device_name(0)
                print(f"CUDA is available: {cuda_device_name}")
                return torch.device("cuda")
            except Exception as e:
                print(f"CUDA detected but failed verification: {e}")
                print("Falling back to next available device")
        else:
            print("CUDA not available")

        # Metal (Apple Silicon) - Second priority
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Verify MPS works
                test_tensor = torch.zeros(1, 1, device="mps")
                print("Apple Silicon Metal acceleration is available")
                return torch.device("mps")
            except Exception as e:
                print(f"Metal detected but failed verification: {e}")
                print("Falling back to CPU")
        else:
            print("Metal acceleration not available")

        # CPU - Final fallback
        print("Using CPU device")
        return torch.device("cpu")

    def _move_model_to_device(self):
        """Safely move model to the selected device with proper error handling"""
        try:
            print(f"Moving model to {self.device}...")
            # Check current device
            current_device = next(self.model.parameters()).device

            # Only move if needed
            if current_device != self.device:
                self.model = self.model.to(self.device)
                # Verify model is on correct device
                new_device = next(self.model.parameters()).device
                print(f"Model moved from {current_device} to {new_device}")

                # Verify model isn't on meta device
                if str(new_device) == "meta":
                    print("WARNING: Model is still on meta device after move attempt!")
                    print("Attempting CPU fallback...")
                    self.device = torch.device("cpu")
                    self.model = self.model.to("cpu")
            else:
                print(f"Model already on {current_device}, no move needed")

            return True
        except Exception as e:
            print(f"Error moving model to {self.device}: {e}")
            print("Falling back to CPU")
            self.device = torch.device("cpu")
            try:
                self.model = self.model.to("cpu")
                return True
            except Exception as e2:
                print(f"Critical error: Failed to move model to CPU: {e2}")
                return False

    def _setup_lora_adapters(self):
        """Setup LoRA adapters for the model"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config["model"]["lora"]["rank"],
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=self.config["model"]["lora"]["target_modules"],
        )

        # Check if adapter weights exist
        adapter_path = os.path.join("models", "lora_adapters.bin")
        if os.path.exists(adapter_path):
            self.model = get_peft_model(self.model, lora_config)
            # Load adapter weights
            self.model.load_adapter(adapter_path)
            print(f"Loaded LoRA adapters from {adapter_path}")
        else:
            print("No pre-trained LoRA adapters found, using base model")

    def generate(self, prompt, max_length=None, **kwargs):
        """Generate text from prompt"""
        if max_length is None:
            max_length = self.config["inference"]["max_length"]

        # Process prompt through the DSE
        symbols = self.dse.process_symbol_definitions(prompt)

        # Verify model is on the correct device
        try:
            model_device = next(self.model.parameters()).device
            print(f"Current model device when generating: {model_device}")

            # If device mismatch, try to fix it
            if model_device != self.device:
                print(
                    f"Device mismatch detected: model on {model_device}, should be on {self.device}"
                )
                success = self._move_model_to_device()
                if not success:
                    return "[Error: Failed to prepare model for inference. Device configuration issue.]"

            # If model is on meta device, we can't use it for inference
            if str(model_device) == "meta":
                print("Model is on meta device - cannot perform inference")
                return (
                    "[Model Error: The model is not initialized properly for inference. "
                    + "This is a known issue with Mamba models. "
                    + "Please try again with a smaller model or on a different device.]"
                )

            # Prepare inputs
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Move inputs to the correct device
            try:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception as e:
                print(f"Error moving inputs to {self.device}: {e}")
                print("Falling back to CPU for inputs")
                self.device = torch.device("cpu")
                try:
                    # Move model to CPU as a last resort
                    self.model = self.model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                except Exception as e2:
                    print(f"Critical error moving to CPU: {e2}")
                    return f"[Critical Error: {str(e2)}]"

            # Set generation parameters
            gen_params = {
                "max_length": max_length,
                "temperature": self.config["inference"]["temperature"],
                "top_p": self.config["inference"]["top_p"],
                "top_k": self.config["inference"]["top_k"],
                "repetition_penalty": self.config["inference"]["repetition_penalty"],
                "do_sample": True,  # Enable sampling since we're using temperature, top_p and top_k
                **kwargs,
            }

            # Generate text
            try:
                with torch.no_grad():
                    # Debug info
                    print(
                        f"Generating with model on {self.device}, inputs on {next(iter(inputs.values())).device}"
                    )
                    outputs = self.model.generate(**inputs, **gen_params)

                # Decode output
                generated_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                return generated_text
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA out of memory error: {e}")
                    print("Attempting to fall back to CPU...")

                    # Move model to CPU and try again
                    self.device = torch.device("cpu")
                    self.model = self.model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, **gen_params)

                    generated_text = self.tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )
                    return generated_text
                else:
                    print(f"Error during text generation: {e}")
                    error_msg = (
                        f"[Model Error: {str(e)}. This might be due to a memory issue "
                        + "or incompatibility with this hardware.]"
                    )
                    return error_msg
            except Exception as e:
                print(f"Error during text generation: {e}")
                error_msg = (
                    f"[Model Error: {str(e)}. This is likely due to device mismatch "
                    + "or memory issues on this device.]"
                )
                return error_msg
        except Exception as e:
            print(f"Critical error during generation setup: {e}")
            return f"[Critical Error: {str(e)}]"

    def save(self, path):
        """Save model and adapters"""
        # Save LoRA adapters
        lora_path = os.path.join(path, "lora_adapters")
        os.makedirs(lora_path, exist_ok=True)
        self.model.save_pretrained(lora_path)

        # Save MoE router
        moe_path = os.path.join(path, "moe_router.pt")
        torch.save(self.moe_router.state_dict(), moe_path)

        print(f"Model saved to {path}")

    def load(self, path):
        """Load model and adapters"""
        # Load LoRA adapters
        lora_path = os.path.join(path, "lora_adapters")
        if os.path.exists(lora_path):
            self.model.load_adapter(lora_path)

        # Load MoE router
        moe_path = os.path.join(path, "moe_router.pt")
        if os.path.exists(moe_path):
            self.moe_router.load_state_dict(torch.load(moe_path))

        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Simple test
    model = NanoCogModel()
    output = model.generate("What is the square root of 16?")
    print(output)
