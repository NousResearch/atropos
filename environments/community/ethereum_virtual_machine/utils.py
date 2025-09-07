"""
Utility functions for the EVM Environment

This module contains cleanup handlers, signal management, and other utility functions.
"""

import atexit
import logging
import signal
import sys


class CleanupManager:
    """Manages cleanup operations for the EVM environment"""

    def __init__(self):
        self.cleanup_functions = []
        self.logger = logging.getLogger(__name__)
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup cleanup handlers for various exit scenarios"""
        # Register cleanup function to run on normal exit
        atexit.register(self._execute_cleanup)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # Termination signal

        # On Windows, also handle SIGBREAK
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._signal_handler)

    def register_cleanup(self, cleanup_func, *args, **kwargs):
        """Register a cleanup function to be called on exit"""
        self.cleanup_functions.append((cleanup_func, args, kwargs))

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self._execute_cleanup()
        sys.exit(0)

    def _execute_cleanup(self):
        """Execute all registered cleanup functions"""
        for cleanup_func, args, kwargs in self.cleanup_functions:
            try:
                cleanup_func(*args, **kwargs)
            except Exception as e:
                print(f"Error during cleanup: {e}")


def setup_evm_error_message(anvil_config, error: Exception) -> str:
    """Generate a comprehensive error message for EVM setup failures"""
    error_message = f"\n❌ Error setting up EVM environment: {error}"
    error_message += "\n\n🔧 Troubleshooting suggestions:"
    error_message += "\n   1. Check if Anvil is already running on the configured port"
    error_message += "\n   2. Ensure no previous Anvil processes are still running:"
    error_message += "\n      - Run: pkill -f anvil"
    error_message += "\n      - Or: ps aux | grep anvil"
    error_message += "\n   3. Verify Foundry/Anvil is properly installed:"
    error_message += "\n      - Run: anvil --version"
    error_message += "\n   4. Check if the port is available:"
    error_message += f"\n      - Run: netstat -tulpn | grep {anvil_config.anvil.port}"
    error_message += (
        "\n\n💡 Try restarting the environment after addressing these issues."
    )

    return error_message


def cleanup_blockchain(blockchain) -> None:
    """Clean up blockchain resources"""
    try:
        if blockchain:
            print("Stopping Anvil blockchain...")
            blockchain.stop()
            print("Anvil stopped successfully.")
    except Exception as e:
        print(f"Error during blockchain cleanup: {e}")


# Global cleanup manager instance
cleanup_manager = CleanupManager()
