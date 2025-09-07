"""Anvil blockchain simulation backend with integrated configuration.

This module provides a complete interface for managing Anvil (Foundry's local Ethereum node)
with integrated YAML configuration loading.
"""

from __future__ import annotations

import atexit
import logging
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

# Set up anvil logger to write to anvil.log
anvil_logger = logging.getLogger("anvil")
anvil_logger.setLevel(logging.INFO)
anvil_logger.propagate = False

# Create file handler for anvil.log
if not anvil_logger.handlers:
    file_handler = logging.FileHandler("anvil.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    anvil_logger.addHandler(file_handler)


class ConfigDict:
    """Helper class to provide dot-notation access to configuration dictionaries."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class AnvilConfig:
    """Configuration loader for Anvil EVM environment."""

    def __init__(self, config_file: str = "configs/token_transfers.yaml"):
        self.config_file = Path(__file__).parent / config_file
        self._raw_config = self._load_config()

        # Create dot-notation accessible config sections
        self.anvil = ConfigDict(
            self._raw_config.get("network", {})
        )  # Renamed from 'network' to 'anvil'
        self.timeouts = ConfigDict(self._raw_config.get("timeouts", {}))
        self.funding = ConfigDict(self._raw_config.get("funding", {}))
        self.whitelisted_tokens = ConfigDict(
            self._raw_config.get("whitelisted_tokens", {})
        )
        self.defi = ConfigDict(self._raw_config.get("defi", {}))
        self.swaps = ConfigDict(self._raw_config.get("swaps", {}))

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, "r") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    # Helper Methods
    def get_rpc_url(self) -> str:
        """Get the full RPC URL for the Anvil instance."""
        return f"http://127.0.0.1:{self.anvil.port}"

    def get_anvil_startup_command(
        self, port: int = None, fork_url: str = None
    ) -> list[str]:
        """Get the Anvil startup command with specified or default parameters."""
        cmd = ["anvil", "--port", str(port or self.anvil.port)]
        if fork_url or self.anvil.fork_url:
            cmd += ["--fork-url", fork_url or self.anvil.fork_url]
        return cmd


class AnvilBackend:
    """Anvil-specific blockchain simulation backend."""

    def __init__(
        self,
        config: AnvilConfig,
        port: Optional[int] = None,
        fork_url: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> None:
        self.config = config
        self.port = port or config.anvil.port
        self.fork_url = fork_url or config.anvil.fork_url
        self.log_file = log_file or config.anvil.log_file
        self._proc: Optional[subprocess.Popen[str]] = None
        self._is_wallet_setup = False
        self.rpc_url = f"http://127.0.0.1:{self.port}"

        # Register cleanup handlers
        self._setup_cleanup_handlers()

    def _setup_cleanup_handlers(self):
        """Setup cleanup handlers for various exit scenarios"""
        # Register cleanup function to run on normal exit
        atexit.register(self._cleanup_process)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # Termination signal

        # On Windows, also handle SIGBREAK
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        anvil_logger.info(
            f"Received signal {signum}, shutting down Anvil gracefully..."
        )
        self._cleanup_process()

    def _cleanup_process(self):
        """Clean up Anvil process"""
        if self._proc and self._proc.poll() is None:
            try:
                anvil_logger.info("Terminating Anvil process...")
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                    anvil_logger.info("Anvil process terminated gracefully")
                except subprocess.TimeoutExpired:
                    anvil_logger.warning(
                        "Anvil didn't terminate gracefully, killing process"
                    )
                    self._proc.kill()
                    self._proc.wait()
                    anvil_logger.info("Anvil process killed")
            except Exception as e:
                anvil_logger.error(f"Error during Anvil cleanup: {e}")
            finally:
                self._proc = None

    def start(self) -> None:
        """Start the Anvil process."""
        if self._proc is not None and self._proc.poll() is None:
            anvil_logger.info("Anvil is already running")
            return  # already running

        cmd = self.config.get_anvil_startup_command(self.port, self.fork_url)

        # Open log file for anvil output
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as log_f:
            log_f.write(f"=== Anvil started at port {self.port} ===\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write("=" * 50 + "\n")

        # spawn detached so we can ctrl-c main program without killing anvil
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # wait until RPC ready and log output
        started = False
        with open(log_path, "a") as log_f:
            for i in range(self.config.timeouts.anvil_startup_lines):
                line = self._proc.stdout.readline()  # type: ignore
                if line:
                    log_f.write(line)
                    log_f.flush()  # Ensure immediate write
                    if "Listening on" in line or "JSON-RPC server started" in line:
                        started = True
                        break
                else:
                    # No more output, break early
                    break
        if not started:
            anvil_logger.error("Failed to launch anvil; did you run the setup script?")
            raise RuntimeError("Failed to launch anvil; did you run the setup script?")

    def stop(self) -> None:
        """Stop the Anvil process."""
        self._cleanup_process()

    def get_rpc_url(self) -> str:
        """Get the RPC URL for this Anvil instance."""
        return self.rpc_url

    def execute_transaction(self, tx_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transaction using cast command.

        Args:
            tx_obj: Transaction object from agent (e.g., {"to": "0x...", "value": "0.5", "data": "0x"})

        Returns:
            Dict with success, gas_used, output, tx_hash, error
        """
        try:
            # Extract transaction fields
            to_address = tx_obj.get("to", "")
            value = tx_obj.get("value", "0")
            data = tx_obj.get("data", "0x")

            # Convert hex value to decimal for cast
            if isinstance(value, str) and value.startswith("0x"):
                try:
                    value_decimal = str(int(value, 16))
                except ValueError:
                    value_decimal = "0"
            else:
                value_decimal = str(value)

            # Build cast command - different approaches based on whether we have data
            if data and data != "0x" and len(data) > 2:
                # Transaction with data (contract interaction) - pass raw hex data as sig parameter
                cmd = [
                    "cast",
                    "send",
                    to_address,
                    data,  # Raw hex data as the sig parameter (selector + encoded calldata)
                    "--from",
                    self.config.funding.custom_wallet,
                    "--unlocked",
                    "--value",
                    value_decimal,
                    "--rpc-url",
                    self.get_rpc_url(),
                ]
            else:
                # Simple ETH transfer
                cmd = [
                    "cast",
                    "send",
                    to_address,
                    "--from",
                    self.config.funding.custom_wallet,
                    "--unlocked",
                    "--value",
                    value_decimal,
                    "--rpc-url",
                    self.get_rpc_url(),
                ]

            # Execute cast command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.cast_command,
            )

            # Parse result
            if result.returncode == 0:
                # Success - extract transaction hash and get receipt
                tx_hash = result.stdout.strip()
                gas_used = self._get_gas_used(tx_hash)

                return {
                    "success": True,
                    "status": "0x1",  # Success status for scoring
                    "gas_used": gas_used,
                    "tx_hash": tx_hash,
                    "output": result.stdout,
                }
            else:
                # Failure - parse error
                error_msg = result.stderr.strip() or result.stdout.strip()
                return {
                    "success": False,
                    "status": "0x0",  # Failure status for scoring
                    "gas_used": 0,
                    "error": error_msg,
                    "output": result.stderr + result.stdout,
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "status": "0x0",
                "gas_used": 0,
                "error": "Transaction timeout",
                "output": "cast command timed out",
            }
        except Exception as e:
            anvil_logger.error(f"Exception in execute_transaction: {str(e)}")
            return {
                "success": False,
                "status": "0x0",
                "gas_used": 0,
                "error": str(e),
                "output": f"Failed to execute cast: {str(e)}",
            }

    def setup_wallet(self, wallet_address: Optional[str] = None) -> None:
        """Setup custom wallet by impersonating it and funding with ETH."""
        if self._is_wallet_setup:
            return  # Already setup

        wallet = wallet_address or self.config.funding.custom_wallet

        try:
            # Impersonate the custom wallet using cast command
            result = subprocess.run(
                [
                    "cast",
                    "rpc",
                    "anvil_impersonateAccount",
                    wallet,
                    "--rpc-url",
                    self.get_rpc_url(),
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.cast_command,
            )

            if result.returncode != 0:
                anvil_logger.error(f"Failed to impersonate wallet: {result.stderr}")
                raise RuntimeError(f"Failed to impersonate wallet: {result.stderr}")

            # Add buffer time
            time.sleep(self.config.timeouts.wallet_setup_buffer)

            # Fund the custom wallet with ETH from Anvil account 0
            result = subprocess.run(
                [
                    "cast",
                    "send",
                    wallet,
                    "--private-key",
                    self.config.funding.anvil_private_key_0,
                    "--value",
                    self.config.funding.initial_funding_amount,
                    "--rpc-url",
                    self.get_rpc_url(),
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.cast_command,
            )

            if result.returncode != 0:
                anvil_logger.error(f"Failed to fund custom wallet: {result.stderr}")
                raise RuntimeError(f"Failed to fund custom wallet: {result.stderr}")

            # Add buffer time before starting swaps
            time.sleep(self.config.timeouts.wallet_setup_buffer)

            # Perform initial token swaps to diversify the wallet
            self._perform_initial_swaps()

            self._is_wallet_setup = True

        except Exception as e:
            anvil_logger.error(f"Error setting up custom wallet: {str(e)}")
            raise

    def snapshot(self) -> str:
        """Take a snapshot of the current blockchain state."""
        return self._rpc("evm_snapshot")

    def revert(self, snap_id: str) -> None:
        """Revert to a previous snapshot."""
        self._rpc("evm_revert", [snap_id])

    # Private helper methods
    def _rpc(self, method: str, params: Optional[List[Any]] = None) -> Any:
        """Make an RPC call to Anvil."""
        import json as _json
        from urllib import request

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or [],
        }
        data = _json.dumps(payload).encode()
        req = request.Request(
            self.get_rpc_url(), data=data, headers={"Content-Type": "application/json"}
        )
        resp = request.urlopen(req)
        result = _json.loads(resp.read())
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    def _get_gas_used(self, tx_hash: str) -> int:
        """Get gas used from transaction receipt using cast."""
        try:
            result = subprocess.run(
                [
                    "cast",
                    "receipt",
                    tx_hash,
                    "--field",
                    "gasUsed",
                    "--rpc-url",
                    self.get_rpc_url(),
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.cast_command,
            )
            if result.returncode == 0:
                return int(result.stdout.strip(), 16)  # Convert hex to int
        except Exception:
            pass
        return 0  # Default if we can't get gas info

    def _perform_initial_swaps(self):
        """Perform initial token swaps to give the wallet a diverse portfolio."""
        # Get token configuration from config
        tokens = self.config.whitelisted_tokens

        # Amount to swap for each token
        swap_amount = self.config.swaps.initial_swap_amount

        # Swap for all whitelisted tokens from config
        for token_name in tokens.__dict__.keys():
            try:
                token_info = getattr(tokens, token_name)

                # Try direct RPC approach
                success = self._execute_swap_direct(
                    token_name, token_info.address, swap_amount
                )

                if success:
                    # Check token balance after swap
                    self._check_token_balance(
                        token_name, token_info.address, token_info.decimals
                    )

                # Add buffer between swaps
                time.sleep(self.config.timeouts.operation_buffer)

            except Exception as e:
                anvil_logger.warning(f"Error swapping ETH for {token_name}: {str(e)}")
                continue

    def _check_token_balance(self, token_name: str, token_address: str, decimals: int):
        """Check and log the balance of a specific token."""
        try:
            balance_result = subprocess.run(
                [
                    "cast",
                    "call",
                    token_address,
                    "balanceOf(address)(uint256)",
                    self.config.funding.custom_wallet,
                    "--rpc-url",
                    self.get_rpc_url(),
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.cast_command,
            )

            if balance_result.returncode == 0:
                balance_output = balance_result.stdout.strip()
                if balance_output:
                    # Parse the balance - cast call returns decimal, not hex
                    # Handle format like "26432331438 [2.643e10]"
                    balance_str = balance_output.split()[
                        0
                    ]  # Take first part before any brackets
                    balance_raw = int(balance_str)
                    balance_formatted = balance_raw / (10**decimals)
                    anvil_logger.info(
                        f"✓ {token_name} balance: {balance_formatted:.6f} {token_name}"
                    )
                else:
                    anvil_logger.warning(
                        f"Empty response when checking {token_name} balance"
                    )
            else:
                anvil_logger.warning(f"Failed to check {token_name} balance")
        except Exception as e:
            anvil_logger.warning(f"Error checking {token_name} balance: {str(e)}")

    def _direct_rpc_call(
        self, method: str, params: Optional[List] = None
    ) -> Dict[str, Any]:
        """Make a direct RPC call to Anvil using HTTP requests."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params or [],
            }

            response = requests.post(
                self.get_rpc_url(),
                json=payload,
                timeout=self.config.timeouts.rpc,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()

                if "error" in result:
                    return {"success": False, "error": result["error"]}
                else:
                    return {"success": True, "result": result.get("result")}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                }

        except requests.exceptions.Timeout:
            return {"success": False, "error": "RPC timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_swap_direct(
        self, token_name: str, token_address: str, swap_amount: str
    ) -> bool:
        """Execute swap using direct RPC calls instead of subprocess."""
        try:
            # Get configuration values
            uniswap_router = self.config.defi.uniswap_v3_router
            weth_address = self.config.defi.weth_address

            # Create deadline
            deadline = hex(
                int(time.time()) + self.config.timeouts.transaction_deadline_offset
            )

            # Function selector for exactInputSingle
            function_selector = self.config.swaps.uniswap_exact_input_single_selector

            # Convert addresses to 32-byte hex (pad with zeros)
            token_in_padded = weth_address.lower().replace("0x", "").zfill(64)
            token_out_padded = token_address.lower().replace("0x", "").zfill(64)
            fee_padded = hex(self.config.defi.default_uniswap_fee)[2:].zfill(64)
            recipient_padded = (
                self.config.funding.custom_wallet.lower().replace("0x", "").zfill(64)
            )
            deadline_padded = deadline[2:].zfill(64)
            amount_in_padded = hex(int(swap_amount))[2:].zfill(64)
            amount_out_min_padded = "0".zfill(64)  # 0 minimum out
            sqrt_price_limit_padded = "0".zfill(64)  # 0 price limit

            # Construct the full calldata
            calldata = (
                function_selector
                + token_in_padded
                + token_out_padded
                + fee_padded
                + recipient_padded
                + deadline_padded
                + amount_in_padded
                + amount_out_min_padded
                + sqrt_price_limit_padded
            )

            # Prepare transaction parameters
            tx_params = {
                "from": self.config.funding.custom_wallet.lower(),
                "to": uniswap_router.lower(),
                "value": hex(int(swap_amount)),
                "data": calldata,
            }

            # Send the transaction via RPC
            result = self._direct_rpc_call("eth_sendTransaction", [tx_params])

            if result["success"]:
                tx_hash = result["result"]

                # Mine a block to include the transaction (Anvil in fork mode doesn't auto-mine)
                mine_result = self._direct_rpc_call("evm_mine")
                if not mine_result["success"]:
                    return False

                # Check the transaction receipt
                receipt_result = self._direct_rpc_call(
                    "eth_getTransactionReceipt", [tx_hash]
                )
                if receipt_result["success"] and receipt_result["result"]:
                    receipt = receipt_result["result"]
                    if receipt.get("status") == "0x1":
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False

        except Exception as e:
            anvil_logger.warning(f"Error in {token_name} swap: {str(e)}")
            return False

    def get_wallet_balances(
        self, wallet_address: Optional[str] = None, tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get wallet balances for specified tokens or default set.

        Args:
            wallet_address: Address to check balances for (defaults to custom wallet)
            tokens: List of token symbols to check (defaults to ETH + whitelisted tokens)

        Returns:
            Dict with token symbols as keys and balance info as values
        """
        wallet = wallet_address or self.config.funding.custom_wallet

        # Default to ETH + whitelisted tokens if none specified
        if tokens is None:
            tokens = ["ETH"] + list(self.config.whitelisted_tokens.__dict__.keys())

        balances = {}

        for token_symbol in tokens:
            try:
                if token_symbol.upper() == "ETH":
                    # Get ETH balance using RPC call
                    result = self._direct_rpc_call("eth_getBalance", [wallet, "latest"])
                    if result["success"]:
                        balance_wei = int(result["result"], 16)  # Convert hex to int
                        balance_eth = balance_wei / 10**18
                        balances["ETH"] = {
                            "symbol": "ETH",
                            "balance": balance_eth,
                            "balance_wei": str(balance_wei),
                            "decimals": 18,
                        }
                    else:
                        balances["ETH"] = {
                            "symbol": "ETH",
                            "balance": 0,
                            "error": result.get("error", "Unknown error"),
                        }

                else:
                    # Get ERC-20 token balance using existing token check pattern
                    token_info = getattr(
                        self.config.whitelisted_tokens, token_symbol, None
                    )
                    if token_info is None:
                        balances[token_symbol] = {
                            "symbol": token_symbol,
                            "balance": 0,
                            "error": "Token not found in config",
                        }
                        continue

                    # Use existing cast command execution pattern
                    balance_result = self._execute_cast_command(
                        [
                            "cast",
                            "call",
                            token_info.address,
                            "balanceOf(address)(uint256)",
                            wallet,
                            "--rpc-url",
                            self.get_rpc_url(),
                        ]
                    )

                    if balance_result["success"]:
                        balance_output = balance_result["output"].strip()
                        if balance_output:
                            # Parse the balance using existing pattern from _check_token_balance
                            balance_str = balance_output.split()[
                                0
                            ]  # Take first part before any brackets
                            balance_raw = int(balance_str)
                            balance_formatted = balance_raw / (10**token_info.decimals)
                            balances[token_symbol] = {
                                "symbol": token_symbol,
                                "balance": balance_formatted,
                                "balance_raw": balance_raw,
                                "decimals": token_info.decimals,
                                "address": token_info.address,
                            }
                        else:
                            balances[token_symbol] = {
                                "symbol": token_symbol,
                                "balance": 0,
                                "error": "Empty balance response",
                            }
                    else:
                        balances[token_symbol] = {
                            "symbol": token_symbol,
                            "balance": 0,
                            "error": balance_result.get("error", "Cast command failed"),
                        }

            except Exception as e:
                balances[token_symbol] = {
                    "symbol": token_symbol,
                    "balance": 0,
                    "error": str(e),
                }
                anvil_logger.error(
                    f"Exception getting {token_symbol} balance: {str(e)}"
                )

        return balances

    def _execute_cast_command(self, cmd: List[str]) -> Dict[str, Any]:
        """
        Execute a cast command and return standardized result.
        Reuses existing patterns for cast command execution.
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeouts.cast_command,
            )

            if result.returncode == 0:
                return {"success": True, "output": result.stdout, "error": None}
            else:
                return {
                    "success": False,
                    "output": result.stdout,
                    "error": result.stderr.strip() or result.stdout.strip(),
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": "Command timeout"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
