"""
Ollama Manager - A comprehensive tool for managing Ollama LLM operations

This module provides a complete interface for:
- Starting/Restarting the Ollama server
- Pulling and running LLM models
- System health checks
- Dependency verification

Example usage:
    >>> manager = OllamaManager()
    >>> manager.execute_workflow("llama2")
"""

import os
import sys
import subprocess
import platform
import logging
import shutil
from typing import Optional

# Setup project root path
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(MAIN_DIR)
except Exception as e:
    logging.basicConfig(level=logging.ERROR)
    logging.error("Failed to set project root: %s", e)

from src.infra import setup_logging

logger = setup_logging(name="OLLAMA-MANEGER")

class OllamaManager:
    """
    Ollama Manager - Central class for managing Ollama operations
    
    Provides complete control over the Ollama server and model operations with:
    - Automatic server management
    - Model downloading and execution
    - Comprehensive health monitoring
    - Detailed logging with emoji indicators
    
    Attributes:
        server_url (str): The base URL for Ollama server (default: http://localhost:11434)
    """

    def __init__(self,
                 server_url: str = "http://localhost:11434"):
        """Initialize the Ollama manager with default server URL"""
        self.server_url = server_url

    def _run_command(self, cmd: str, check: bool = True,
                   capture_output: bool = False) -> Optional[subprocess.CompletedProcess]:
        """
        Execute shell command with comprehensive error handling
        
        Args:
            cmd: The command string to execute
            check: Whether to raise exception on non-zero exit code
            capture_output: Whether to capture and return command output
            
        Returns:
            CompletedProcess object if successful, None on failure
            
        Logs:
            DEBUG: Command being executed
            ERROR: Command failure details
        """
        logger.debug("Running command: %s", cmd)
        try:
            return subprocess.run(
                cmd,
                shell=True,
                check=check,
                text=True,
                capture_output=capture_output,
                encoding="utf-8"
            )
        except subprocess.CalledProcessError as e:
            logger.error("Command failed (exit %d): %s", e.returncode, e.stderr)
            return None
        except Exception as e:
            logger.error("Unexpected error running command: %s", str(e))
            return None

    def check_dependencies(self) -> bool:
        """
        Verify all system dependencies are available
        
        Checks for:
        - Ollama CLI
        - Systemd (for service management)
        - curl (for health checks)
        - pgrep (for process management)
        
        Returns:
            bool: True if all dependencies are available, False otherwise
            
        Logs:
            ERROR: Missing dependencies
            DEBUG: Dependency check passed
        """
        required_commands = [
            ("ollama", "Ollama CLI"),
            ("systemctl", "Systemd (for service management)"),
            ("curl", "HTTP client (for health checks)"),
            ("pgrep", "Process management")
        ]

        missing = []
        for cmd, description in required_commands:
            if not shutil.which(cmd):
                missing.append(cmd)
                logger.error("Missing dependency: %s (%s)", description, cmd)

        if missing:
            logger.critical("Missing critical dependencies: %s", ", ".join(missing))
            return False

        logger.debug("All dependencies are available")
        return True

    def is_server_running(self) -> bool:
        """
        Check if Ollama server process is running
        
        Returns:
            bool: True if server is running, False otherwise
            
        Logs:
            ERROR: Process check failure
        """
        try:
            result = subprocess.run(["pgrep", "-f", "ollama serve"],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 check=False)
            return result.returncode == 0
        except Exception as e:
            logger.error("Error checking Ollama process: %s", str(e))
            return False

    def start_server(self) -> bool:
        """
        Start Ollama server with fallback mechanisms
        
        Attempts:
        1. Systemd service start
        2. Direct process launch
        
        Returns:
            bool: True if server started successfully, False otherwise
            
        Logs:
             INFO: Server status messages
             WARNING: Fallback to direct start
            ERROR: Startup failure
        """
        if self.is_server_running():
            logger.info(" Ollama server is already running ")
            return True

        logger.info("Starting Ollama server...")

        # Try systemd first
        result = self._run_command("systemctl --user start ollama")
        if result and result.returncode == 0:
            logger.info(" Ollama service started via systemd ")
            return True

        # Fallback to direct start
        logger.warning(" Systemd start failed, trying direct start...")
        result = self._run_command("ollama serve > /dev/null 2>&1 &", check=False)

        if not self.is_server_running():
            logger.error("Failed to start Ollama server")
            return False

        logger.info("Ollama server started successfully")
        return True

    def health_check(self) -> bool:
        """
        Verify server is responding properly
        
        Returns:
            bool: True if health check passed, False otherwise
            
        Logs:
             INFO: Health check initiation
             WARNING: Unexpected response
            ERROR: Health check failure
            INFO: Health check passed
        """
        logger.info("Performing health check...")

        try:
            response = subprocess.run(
                f"curl -s {self.server_url}",
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )

            if "Ollama is running" in response.stdout:
                logger.info("Health check passed")
                return True
            logger.warning(" Unexpected health check response: %s", response.stdout)
            return False
        except Exception as e:
            logger.error("Health check failed: %s", str(e))
            return False

    def pull_model(self, model_name: str) -> bool:
        """
        Download the specified Ollama model
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            bool: True if download succeeded, False otherwise
            
        Logs:
             INFO: Download initiation
            ERROR: Download failure
            INFO: Download success
        """
        logger.info("Pulling model: %s", model_name)
        result = self._run_command(f"ollama pull {model_name}", check=False)

        if not result or result.returncode != 0:
            logger.error("Failed to pull model %s", model_name)
            return False

        logger.info("Successfully pulled model: %s", model_name)
        return True

    def run_model(self, model_name: str) -> bool:
        """
        Run the specified Ollama model interactively
        
        Args:
            model_name: Name of the model to run
            
        Returns:
            bool: True if session completed successfully, False otherwise
            
        Logs:
             INFO: Session start
             INFO: User termination
            ERROR: Runtime error
        """
        logger.info("🚀 Running model: %s", model_name)

        try:
            process = subprocess.Popen(
                ["ollama", "run", model_name],
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )

            logger.info("Interactive session started with %s. Press Ctrl+C to exit.", model_name)
            process.wait()
            return True
        except KeyboardInterrupt:
            logger.info(" Session terminated by user")
            return True
        except Exception as e:
            logger.error("Error running model: %s", str(e))
            return False

    def execute_workflow(self, model_name: str):
        """
        Executes the full workflow to:
        1. Check dependencies
        2. Start the Ollama server (if not running)
        3. Pull the specified model
        4. Run the model in background
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        logger.info("Starting Ollama Manager Workflow for model: %s", model_name)

        # Step 1: Check dependencies
        if not self.check_dependencies():
            logger.critical("Aborting workflow due to missing dependencies.")
            return

        # Step 2: Ensure server is running
        if not self.start_server():
            logger.critical("Aborting workflow. Could not start Ollama server.")
            return

        # Step 3: Pull the model
        if not self.pull_model(model_name):
            logger.critical("Aborting workflow. Could not pull model.")
            return

        # Step 4: Run the model
        logger.info("Launching model: %s", model_name)

        if platform.system() == "Windows":
            subprocess.Popen(
                ["start", "cmd", "/k", f"ollama run {model_name}"],
                shell=True
            )
        else:
            subprocess.Popen(
                ["ollama", "run", model_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True
            )

        logger.info("Model '%s' started successfully in background", model_name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Ollama Manager - Model Runner")
        print("Usage: python ollama_manager.py <model_name>")
        print("\nAvailable models (examples):")
        print("  - llama2")
        print("  - mistral")
        print("  - codellama")
        print("\nFirst run might take longer as it downloads the model")
        sys.exit(1)

    model_name = sys.argv[1]
    manager = OllamaManager()
    manager.execute_workflow(model_name)
