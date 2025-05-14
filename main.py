#!/usr/bin/env python3
"""
Nano-Cog 0.1 - A laptop-scale language agent with high reasoning-per-FLOP efficiency
CLI Interface
"""

import os
import sys
import argparse
import readline  # For better command line editing
import logging
import traceback
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nano-cog")

logger.info("Starting Nano-Cog CLI")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")

try:
    logger.info("Importing colorama...")
    from colorama import init, Fore, Style

    logger.info("Colorama imported successfully")
except ImportError as e:
    logger.error(f"Failed to import colorama: {e}")
    print(
        "Error: Missing dependency 'colorama'. Please install with: pip install colorama"
    )
    sys.exit(1)

try:
    logger.info("Importing config utilities...")
    from src.utils.config import load_config

    logger.info("Config utilities imported successfully")
except ImportError as e:
    logger.error(f"Failed to import config utilities: {e}")
    traceback.print_exc()
    print("Error: Failed to import configuration utilities.")
    sys.exit(1)

try:
    logger.info("Importing NanoCogModel...")
    from src.core.model import NanoCogModel

    logger.info("NanoCogModel imported successfully")
except ImportError as e:
    logger.error(f"Failed to import NanoCogModel: {e}")
    traceback.print_exc()
    print("Error: Failed to import model components.")
    sys.exit(1)

try:
    logger.info("Importing ToolDispatcher...")
    from src.tools.dispatcher import ToolDispatcher

    logger.info("ToolDispatcher imported successfully")
except ImportError as e:
    logger.error(f"Failed to import ToolDispatcher: {e}")
    traceback.print_exc()
    print(
        "Error: Failed to import tool dispatcher. Make sure 'restrictedpython' is installed."
    )
    print("Tip: Run 'pip install restrictedpython==8.0' and try again.")
    sys.exit(1)

try:
    logger.info("Importing RetrievalSystem...")
    from src.core.retrieval import RetrievalSystem

    logger.info("RetrievalSystem imported successfully")
except ImportError as e:
    logger.error(f"Failed to import RetrievalSystem: {e}")
    traceback.print_exc()
    print("Error: Failed to import retrieval system.")
    sys.exit(1)

# Initialize colorama
init()


def print_banner():
    """Print Nano-Cog ASCII banner"""
    banner = f"""
{Fore.CYAN}═══════════════════════════════════════════════════════{Style.RESET_ALL}
{Fore.CYAN}███╗   ██╗ █████╗ ███╗   ██╗ ██████╗      ██████╗ ██████╗  ██████╗{Style.RESET_ALL}
{Fore.CYAN}████╗  ██║██╔══██╗████╗  ██║██╔═══██╗    ██╔════╝██╔═══██╗██╔════╝{Style.RESET_ALL}
{Fore.CYAN}██╔██╗ ██║███████║██╔██╗ ██║██║   ██║    ██║     ██║   ██║██║  ███╗{Style.RESET_ALL}
{Fore.CYAN}██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║    ██║     ██║   ██║██║   ██║{Style.RESET_ALL}
{Fore.CYAN}██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝    ╚██████╗╚██████╔╝╚██████╔╝{Style.RESET_ALL}
{Fore.CYAN}╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝      ╚═════╝ ╚═════╝  ╚═════╝{Style.RESET_ALL}
{Fore.CYAN}═══════════════════════════════════════════════════════{Style.RESET_ALL}
{Fore.YELLOW}Laptop-scale language agent with high reasoning-per-FLOP efficiency{Style.RESET_ALL}
{Fore.YELLOW}Version 0.1{Style.RESET_ALL}
"""
    print(banner)


class NanoCogCLI:
    """
    Command Line Interface for Nano-Cog
    """

    def __init__(self, model_path=None, config_path=None):
        """
        Initialize the CLI

        Args:
            model_path (str, optional): Path to model checkpoint
            config_path (str, optional): Path to config file
        """
        print_banner()
        logger.info("Initializing NanoCogCLI")

        # Load configuration
        logger.info(f"Loading configuration from {config_path or 'default path'}")
        self.config = load_config(config_path)
        logger.info("Configuration loaded successfully")

        # Create output directories if they don't exist
        logger.info("Creating output directories")
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        logger.info("Output directories created")

        # Verify model files exist
        model_path = model_path or self.config.get("model", {}).get(
            "path", "./models/mamba-130m"
        )
        model_files_exist = os.path.exists(os.path.join(model_path, "config.json"))

        if not model_files_exist:
            logger.warning(f"Model files not found at {model_path}")
            print(
                f"{Fore.YELLOW}Model files not found. Attempting to download...{Style.RESET_ALL}"
            )

            # Try to download the model using our script
            try:
                download_script = os.path.join("scripts", "download_weights.py")
                if os.path.exists(download_script):
                    logger.info("Running model download script")
                    subprocess.run(
                        [sys.executable, download_script, "--output-dir", model_path],
                        check=True,
                    )
                    logger.info("Model download completed")
                    print(
                        f"{Fore.GREEN}Model downloaded successfully.{Style.RESET_ALL}"
                    )
                else:
                    logger.error(f"Download script not found at {download_script}")
                    print(
                        f"{Fore.RED}Download script not found. Please run 'python scripts/setup.py' first.{Style.RESET_ALL}"
                    )
                    sys.exit(1)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download model: {e}")
                print(
                    f"{Fore.RED}Failed to download model. Please run 'python scripts/download_weights.py' manually.{Style.RESET_ALL}"
                )
                sys.exit(1)

        print(f"{Fore.GREEN}Loading model...{Style.RESET_ALL}")
        logger.info("Initializing NanoCogModel")
        try:
            self.model = NanoCogModel(config_path)
            logger.info("NanoCogModel initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NanoCogModel: {e}")
            traceback.print_exc()
            print(f"Error: Failed to initialize model: {e}")
            sys.exit(1)

        if model_path and os.path.exists(model_path):
            print(
                f"{Fore.GREEN}Loading checkpoint from {model_path}...{Style.RESET_ALL}"
            )
            logger.info(f"Loading model checkpoint from {model_path}")
            try:
                self.model.load(model_path)
                logger.info("Model checkpoint loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model checkpoint: {e}")
                traceback.print_exc()
                print(f"Error: Failed to load model checkpoint: {e}")

        # Initialize tool dispatcher
        logger.info("Initializing ToolDispatcher")
        try:
            self.tool_dispatcher = ToolDispatcher(config_path)
            logger.info("ToolDispatcher initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ToolDispatcher: {e}")
            traceback.print_exc()
            print(f"Error: Failed to initialize tool dispatcher: {e}")
            sys.exit(1)

        # Initialize retrieval system
        logger.info("Initializing RetrievalSystem")
        try:
            self.retrieval = RetrievalSystem(config_path)
            logger.info("RetrievalSystem initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RetrievalSystem: {e}")
            traceback.print_exc()
            print(f"Error: Failed to initialize retrieval system: {e}")
            sys.exit(1)

        # Initialize conversation history
        self.history = []
        self.max_history = 10  # Number of turns to keep in context
        logger.info("Conversation history initialized")

        print(f"{Fore.GREEN}Nano-Cog is ready for conversation!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Type 'exit', 'quit', or Ctrl+C to exit.{Style.RESET_ALL}")
        print(
            f"{Fore.YELLOW}Type 'clear' to clear conversation history.{Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}Type 'save <filename>' to save conversation.{Style.RESET_ALL}"
        )
        print()
        logger.info("NanoCogCLI initialization complete")

    def process_special_commands(self, user_input):
        """
        Process special commands

        Args:
            user_input (str): User input

        Returns:
            bool: True if special command was processed, False otherwise
        """
        # Check for exit command
        if user_input.lower() in ["exit", "quit"]:
            logger.info("Exit command received")
            print(f"{Fore.YELLOW}Exiting Nano-Cog. Goodbye!{Style.RESET_ALL}")
            sys.exit(0)

        # Check for clear command
        if user_input.lower() == "clear":
            logger.info("Clear command received")
            self.history = []
            print(f"{Fore.GREEN}Conversation history cleared.{Style.RESET_ALL}")
            return True

        # Check for save command
        if user_input.lower().startswith("save "):
            filename = user_input[5:].strip()
            if not filename.endswith(".txt"):
                filename += ".txt"

            logger.info(f"Save command received. Saving to {filename}")
            with open(filename, "w") as f:
                for turn in self.history:
                    f.write(f"User: {turn['user']}\n\n")
                    f.write(f"Nano-Cog: {turn['response']}\n\n")
                    f.write("-" * 50 + "\n\n")

            print(f"{Fore.GREEN}Conversation saved to {filename}{Style.RESET_ALL}")
            logger.info(f"Conversation saved to {filename}")
            return True

        return False

    def get_context_from_history(self):
        """
        Get context from conversation history

        Returns:
            str: Context string
        """
        if not self.history:
            logger.debug("No conversation history available")
            return ""

        # Use last few turns as context
        recent_history = self.history[-self.max_history :]
        logger.debug(f"Using {len(recent_history)} turns from history as context")

        context = ""
        for turn in recent_history:
            context += f"User: {turn['user']}\n"
            context += f"Assistant: {turn['response']}\n"

        return context

    def run(self):
        """Run the CLI interface in a loop"""
        logger.info("Starting CLI interface loop")
        try:
            while True:
                # Get user input
                user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}")
                logger.info("Received user input")
                print()

                # Check for special commands
                if self.process_special_commands(user_input):
                    logger.info("Special command processed")
                    continue

                # Get context from history
                logger.info("Getting context from history")
                context = self.get_context_from_history()

                # Compose prompt with retrieval
                logger.info("Composing prompt")
                if context:
                    # If we have context, append the new user input
                    full_prompt = f"{context}\nUser: {user_input}\nAssistant:"
                    logger.debug("Using conversation history for context")
                else:
                    # If no context, use retrieval to enhance prompt
                    logger.debug("Using retrieval for context")
                    full_prompt = self.retrieval.compose_prompt(user_input)

                logger.info("Generating response")
                try:
                    # Generate response
                    response = self.model.generate(full_prompt)
                    logger.info("Response generated successfully")
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    traceback.print_exc()
                    print(f"Error generating response: {e}")
                    response = "I encountered an error while processing your request. Please try again."

                logger.info("Processing tool calls")
                try:
                    # Process any tool calls
                    processed_response = self.tool_dispatcher.process_text(response)
                    logger.info("Tool calls processed successfully")
                except Exception as e:
                    logger.error(f"Error processing tool calls: {e}")
                    traceback.print_exc()
                    print(f"Error processing tool calls: {e}")
                    processed_response = response

                # Print response
                print(f"{Fore.GREEN}Nano-Cog:{Style.RESET_ALL} {processed_response}")
                print()

                # Update history
                self.history.append(
                    {"user": user_input, "response": processed_response}
                )
                logger.info("Conversation history updated")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, exiting")
            print(f"\n{Fore.YELLOW}Exiting Nano-Cog. Goodbye!{Style.RESET_ALL}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Unhandled exception: {e}")
            traceback.print_exc()
            print(f"An unexpected error occurred: {e}")
            sys.exit(1)


def main():
    """Main function"""
    logger.info("Parsing command line arguments")
    parser = argparse.ArgumentParser(description="Nano-Cog 0.1 CLI")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info(f"Model path: {args.model or 'None'}")
    logger.info(f"Config path: {args.config or 'None'}")

    try:
        logger.info("Initializing NanoCogCLI")
        cli = NanoCogCLI(args.model, args.config)
        logger.info("Starting CLI interface")
        cli.run()
    except Exception as e:
        logger.error(f"Failed to start CLI: {e}")
        traceback.print_exc()
        print(f"Error: Failed to start Nano-Cog CLI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
