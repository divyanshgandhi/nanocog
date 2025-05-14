"""
Tool Dispatcher for Nano-Cog
Handles calculator, Python, and bash tool calls
"""

import os
import re
import json
import time
import sympy
import subprocess
import ast  # For safer Python code analysis
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import load_config


class ToolDispatcher:
    """
    Dispatches tool calls from the model to appropriate handlers
    """

    def __init__(self, config_path=None):
        """
        Initialize the tool dispatcher

        Args:
            config_path (str, optional): Path to config file
        """
        self.config = load_config(config_path)

        # Extract tool configurations
        self.calc_token = self.config["tools"]["calc"]["token"]
        self.python_token = self.config["tools"]["python"]["token"]
        self.bash_token = self.config["tools"]["bash"]["token"]

        self.allowed_python_modules = self.config["tools"]["python"]["allowed_modules"]
        self.allowed_bash_commands = self.config["tools"]["bash"]["allowed_commands"]

        self.timeout_ms = {
            "calc": self.config["tools"]["calc"]["timeout_ms"],
            "python": self.config["tools"]["python"]["timeout_ms"],
            "bash": self.config["tools"]["bash"]["timeout_ms"],
        }

    def extract_tool_calls(self, text):
        """
        Extract tool calls from generated text

        Args:
            text (str): Generated text from model

        Returns:
            list: List of dicts with tool, args, and full_match keys
        """
        tool_calls = []

        # Extract calculator calls
        calc_pattern = f"{self.calc_token}(.*?){self.calc_token}"
        calc_matches = re.finditer(calc_pattern, text, re.DOTALL)
        for match in calc_matches:
            tool_calls.append(
                {
                    "tool": "calc",
                    "args": match.group(1).strip(),
                    "full_match": match.group(0),
                }
            )

        # Extract Python calls
        python_pattern = f"{self.python_token}(.*?){self.python_token}"
        python_matches = re.finditer(python_pattern, text, re.DOTALL)
        for match in python_matches:
            tool_calls.append(
                {
                    "tool": "python",
                    "args": match.group(1).strip(),
                    "full_match": match.group(0),
                }
            )

        # Extract bash calls
        bash_pattern = f"{self.bash_token}(.*?){self.bash_token}"
        bash_matches = re.finditer(bash_pattern, text, re.DOTALL)
        for match in bash_matches:
            tool_calls.append(
                {
                    "tool": "bash",
                    "args": match.group(1).strip(),
                    "full_match": match.group(0),
                }
            )

        return tool_calls

    def execute_tool(self, tool_call):
        """
        Execute a tool call

        Args:
            tool_call (dict): Tool call dict with tool, args, and full_match keys

        Returns:
            dict: Result of tool execution with structure {ok: bool, result: str, error: str}
        """
        tool = tool_call["tool"]
        args = tool_call["args"]

        # Set timeout
        timeout_sec = self.timeout_ms[tool] / 1000

        # Execute appropriate tool
        try:
            start_time = time.time()

            if tool == "calc":
                result = self._execute_calculator(args)
            elif tool == "python":
                result = self._execute_python(args)
            elif tool == "bash":
                result = self._execute_bash(args)
            else:
                return {"ok": False, "result": "", "error": f"Unknown tool '{tool}'"}

            # Check if execution exceeded timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_sec:
                return {
                    "ok": False,
                    "result": "",
                    "error": f"TOOL_ERROR:{tool}:Execution timed out ({elapsed_time:.2f}s)",
                }

            # Check if result is already a structured response
            if isinstance(result, dict) and all(
                k in result for k in ["ok", "result", "error"]
            ):
                return result

            # Otherwise wrap the successful result
            return {"ok": True, "result": result, "error": ""}

        except Exception as e:
            # Sanitize traceback by removing file paths
            error_msg = str(e)
            # Remove any absolute file paths from error message
            error_msg = re.sub(r'File ".*?/([^/]+\.py)"', r'File "\1"', error_msg)
            # Truncate to 200 chars max
            if len(error_msg) > 200:
                error_msg = error_msg[:197] + "..."

            return {
                "ok": False,
                "result": "",
                "error": f"TOOL_ERROR:{tool}:{error_msg}",
            }

    def _execute_calculator(self, expression):
        """
        Execute mathematical expression using sympy

        Args:
            expression (str): Math expression to evaluate

        Returns:
            str: Result of calculation
        """
        try:
            # Sanitize input by removing unsafe characters
            expression = re.sub(
                r"[^\d\s\+\-\*\/\(\)\.\,\=\^\%\!\|\&\<\>\[\]\{\}]", "", expression
            )

            # Evaluate using sympy
            result = sympy.sympify(expression)

            return str(result)
        except Exception as e:
            return f"Calculator error: {str(e)}"

    def _is_safe_python_code(self, code):
        """
        Check if Python code is safe to execute by examining its AST

        Args:
            code (str): Python code to check

        Returns:
            bool: True if safe, False otherwise
        """
        # List of forbidden AST node types
        forbidden_nodes = [
            ast.Import,
            ast.ImportFrom,
            ast.ClassDef,
            ast.Lambda,
            ast.Exec,
            ast.SystemExit,
            ast.GeneratorExp,
            ast.Await,
            ast.AsyncFor,
            ast.AsyncWith,
        ]

        # List of forbidden attributes
        forbidden_attrs = [
            "eval",
            "exec",
            "compile",
            "__import__",
            "open",
            "file",
            "globals",
            "locals",
            "delattr",
            "getattr",
            "setattr",
            "os",
            "sys",
            "subprocess",
            "__builtins__",
            "__class__",
            "__base__",
            "__subclasses__",
        ]

        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Check for forbidden node types
            for node in ast.walk(tree):
                if any(isinstance(node, forbidden) for forbidden in forbidden_nodes):
                    return False

                # Check for attribute access
                if isinstance(node, ast.Attribute):
                    if node.attr in forbidden_attrs:
                        return False

                # Check for name access
                if isinstance(node, ast.Name):
                    if node.id in forbidden_attrs:
                        return False

            return True

        except SyntaxError:
            return False

    def _execute_python(self, code):
        """
        Execute Python code in a safer way

        Args:
            code (str): Python code to execute

        Returns:
            str: Result of execution
        """
        if not self._is_safe_python_code(code):
            return "Error: Code contains potentially unsafe operations"

        # Filter imports to only allow specific modules
        import_regex = r"(?:from|import)\s+(\w+)"
        imports = re.findall(import_regex, code)

        for module_name in imports:
            if module_name not in self.allowed_python_modules:
                return f"Error: Import of module '{module_name}' is not allowed"

        # Use string IO to capture printed output
        import io
        from contextlib import redirect_stdout

        output_buffer = io.StringIO()
        result = None

        # Dictionary to store local variables after execution
        local_vars = {}

        try:
            # Execute code with output redirection
            with redirect_stdout(output_buffer):
                exec(code, {"__builtins__": __builtins__}, local_vars)

            # Get any printed output
            output = output_buffer.getvalue()

            # If no printed output, try to get the last assigned variable
            if not output and local_vars:
                # Find variable that might represent the result
                for var_name in ["result", "answer", "output", "res"]:
                    if var_name in local_vars:
                        result = str(local_vars[var_name])
                        break

                # If no specific result variable, return the last defined variable
                if result is None:
                    # Get last alphabetically named variable as a fallback
                    last_var = sorted(local_vars.keys())[-1]
                    result = str(local_vars[last_var])

            return output or result or "Code executed successfully (no output)"

        except Exception as e:
            return f"Python execution error: {str(e)}"

    def _execute_bash(self, command):
        """
        Execute bash command in a restricted environment

        Args:
            command (str): Bash command to execute

        Returns:
            str: Result of execution
        """
        # Check if command is allowed
        command_parts = command.strip().split()
        base_command = command_parts[0] if command_parts else ""

        if base_command not in self.allowed_bash_commands:
            return f"Error: Command '{base_command}' is not in the allowed list"

        try:
            # Execute command and capture output
            process = subprocess.run(
                command_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_ms["bash"] / 1000,
            )

            # Return output or error
            if process.returncode == 0:
                return process.stdout
            else:
                return f"Error (code {process.returncode}): {process.stderr}"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {self.timeout_ms['bash']}ms"
        except Exception as e:
            return f"Bash execution error: {str(e)}"

    def process_text(self, text):
        """
        Process text and execute any tool calls

        Args:
            text (str): Text to process

        Returns:
            str: Text with tool calls replaced by results
        """
        # Extract tool calls
        tool_calls = self.extract_tool_calls(text)

        # Process each tool call
        for tool_call in tool_calls:
            result_dict = self.execute_tool(tool_call)

            if result_dict["ok"]:
                # Success case
                replacement = f"{tool_call['full_match']}\n{result_dict['result']}"
            else:
                # Error case - use the marker string for errors
                replacement = f"{tool_call['full_match']}\n{result_dict['error']}"

            # Replace the tool call in the text
            text = text.replace(tool_call["full_match"], replacement)

        return text


if __name__ == "__main__":
    # Test tool dispatcher
    dispatcher = ToolDispatcher()

    # Test calculator
    calc_test = f"To find the square root of 16, I'll use: {dispatcher.calc_token}sqrt(16){dispatcher.calc_token}"
    print("Calculator test:")
    print(f"Input: {calc_test}")
    print(f"Output: {dispatcher.process_text(calc_test)}\n")

    # Test Python
    python_test = f"Let's compute fibonacci: {dispatcher.python_token}def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n\nprint(fib(7)){dispatcher.python_token}"
    print("Python test:")
    print(f"Input: {python_test}")
    print(f"Output: {dispatcher.process_text(python_test)}\n")

    # Test bash
    bash_test = f"Let's see what files we have: {dispatcher.bash_token}ls -la{dispatcher.bash_token}"
    print("Bash test:")
    print(f"Input: {bash_test}")
    print(f"Output: {dispatcher.process_text(bash_test)}")
