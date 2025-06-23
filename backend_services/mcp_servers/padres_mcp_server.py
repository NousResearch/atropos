# backend_services/mcp_servers/padres_mcp_server.py

import json  # For pretty printing in main example
import os
import random
import string
from typing import Any, Dict, List, Optional  # For type hinting

import httpx  # For making HTTP calls to the Padres service

from .base_mcp_server import BaseMCPServer


class PadresMCPServer(BaseMCPServer):
    """
    MCP Server implementation for interacting with a Padres Environment service API.
    The API is now expected to be running in the `padres_container` at `service_base_url`.
    """

    DEFAULT_TIMEOUT = 60.0  # Seconds

    def __init__(self, server_name: str, config: dict = None):
        """
        Initializes the PadresMCPServer.
        Expected config keys:
            - "service_base_url": The base URL of the Padres service
                                  (e.g., http://localhost:8088).
            - "timeout" (optional): HTTP request timeout in seconds.
            - "mock_mode" (optional): If True, uses internal mock logic instead of HTTP calls.
        """
        self.http_client: Optional[httpx.AsyncClient] = None

        # Initialize attributes needed by _initialize_client (called by super)
        self.config = config or {}
        self.mock_mode = self.config.get("mock_mode", False)  # Default to False
        self.service_base_url = self.config.get(
            "service_base_url"
        )  # No default, must be provided unless mock_mode

        super().__init__(
            server_name, config
        )  # Initialize server_name and basic config handling

        if self.mock_mode:
            self.mock_environments = {}  # For mock environment states
            self.mock_trials = {}  # For mock trial states
            self._next_mock_env_id = 1
            self._next_mock_trial_id = 1
            print("PadresMCPServer "{self.server_name}' initialized in MOCK mode.")
        else:
            if not self.service_base_url:
                raise ValueError(
                    ""service_base_url' must be provided for PadresMCPServer '{self.server_name}' when not in mock_mode."
                )
            if "://" not in self.service_base_url:  # Basic check
                raise ValueError(
                    ""service_base_url' ('{self.service_base_url}') appears invalid for PadresMCPServer '{self.server_name}'."
                )
            # _initialize_client() is called by super().__init__ if not in mock_mode
            # However, we ensure it's called if it wasn't (e.g. if logic changes in BaseMCPServer)
            # and to print the status.
            if (
                not self.http_client and not self.mock_mode
            ):  # if client not already set by super call
                self._initialize_client()

            if (
                self.http_client or self.mock_mode
            ):  # Successfully initialized or in mock mode
                print(
                    "PadresMCPServer "{self.server_name}' initialized. Mode: {'mock' if self.mock_mode else 'live'}. Target: {self.service_base_url if not self.mock_mode else 'N/A'}"
                )
            else:  # Failed to initialize client in live mode
                print(
                    "PadresMCPServer "{self.server_name}' FAILED to initialize for live mode. Check service_base_url and connectivity."
                )

    def _validate_config(self):
        """Validates config. Called by BaseMCPServer if implemented."""
        # Already handled in __init__ for mock_mode vs live_mode
        pass

    def _initialize_client(self):
        """Initializes an HTTP client for Padres service interaction."""
        # This method is called by BaseMCPServer's __init__
        if self.mock_mode:  # No client needed for mock mode
            self.http_client = None
            print(
                "PadresMCPServer "{self.server_name}': Client initialization skipped (mock mode)."
            )
            return

        if not self.service_base_url:
            print(
                "Warning: service_base_url not set for PadresMCPServer "{self.server_name}'. Client not initialized."
            )
            self.http_client = None
            return

        timeout = self.config.get("timeout", self.DEFAULT_TIMEOUT)
        try:
            self.http_client = httpx.AsyncClient(
                base_url=self.service_base_url,
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            print(
                "HTTP AsyncClient initialized for PadresMCPServer "{self.server_name}' targeting URL '{self.service_base_url}'."
            )
        except Exception as e:
            print(
                f"CRITICAL: Error initializing HTTP AsyncClient for PadresMCPServer {self.server_name}: {e}"
            )
            self.http_client = None

    async def _api_call(
        self,
        method: str,
        endpoint: str,
        json_payload: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Helper for making API calls."""
        if not self.http_client:
            return {
                "error": f"Padres client not initialized or in mock mode for {self.server_name}.",
                "status": "failure",
            }

        url = f"{self.service_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        print(
            f"  {self.server_name}: Calling {method} {url} with payload: {json_payload} and params: {params}"
        )

        try:
            response = await self.http_client.request(
                method, endpoint, json=json_payload, params=params
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = "No response body"
            try:
                error_detail = (
                    e.response.json() if e.response.content else e.response.text
                )
            except Exception:
                pass  # Keep error_detail as text if JSON parsing fails
            print(
                f"HTTP error for {self.server_name} calling {url}: {e.response.status_code} - {error_detail}"
            )
            return {
                "error": "Padres service HTTP error",
                "status_code": e.response.status_code,
                "details": error_detail,
                "status": "failure",
            }
        except httpx.RequestError as e:
            print(f"Request error for {self.server_name} calling {url}: {e}")
            return {
                "error": "Padres service request error",
                "details": str(e),
                "status": "failure",
            }
        except Exception as e:
            print(
                f"Generic error for {self.server_name} calling {url}: {type(e).__name__} - {e}"
            )
            return {
                "error": "Generic error during Padres API call",
                "details": str(e),
                "status": "failure",
            }

    # --- Methods mapped to new Padres API endpoints ---

    async def _padres_setup_environment(
        self, config: Dict[str, Any], environment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calls the /setup_environment endpoint of the Padres service."""
        payload = {"config": config}
        if environment_id:
            payload["environment_id"] = environment_id
        return await self._api_call("POST", "/setup_environment", json_payload=payload)

    async def _padres_run_trial(
        self,
        environment_id: str,
        parameters: Dict[str, Any],
        trial_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calls the /run_trial endpoint of the Padres service."""
        payload = {"environment_id": environment_id, "parameters": parameters}
        if trial_id:
            payload["trial_id"] = trial_id
        return await self._api_call("POST", "/run_trial", json_payload=payload)

    async def _padres_get_results(self, trial_id: str) -> Dict[str, Any]:
        """Calls the /get_results/{trial_id} endpoint of the Padres service."""
        return await self._api_call("GET", f"/get_results/{trial_id}")

    async def _padres_get_status(self) -> Dict[str, Any]:
        """Calls the /status endpoint of the Padres service."""
        return await self._api_call("GET", "/status")

    # --- Mock implementations for when mock_mode is True ---

    def _mock_setup_environment(
        self, config_payload: Dict[str, Any], environment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        env_id = environment_id or f"mock_env_{self._next_mock_env_id}"
        self._next_mock_env_id += 1
        self.mock_environments[env_id] = {
            "config": config_payload,
            "status": "READY",
            "created_at": "mock_timestamp_setup",
        }
        print(
            "MockPadresServer: Environment "{env_id}' set up with config: {config_payload}"
        )
        return {
            "environment_id": env_id,
            "status": "SUCCESS",
            "details": {
                "message": f"Mock environment {env_id} configured.",
                "current_config": config_payload,
            },
        }

    def _mock_run_trial(
        self,
        environment_id: str,
        parameters: Dict[str, Any],
        trial_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if environment_id not in self.mock_environments:
            return {
                "error": f"Mock environment {environment_id} not found.",
                "status": "failure",
                "trial_id": trial_id,
            }

        tr_id = trial_id or f"mock_trial_{self._next_mock_trial_id}"
        self._next_mock_trial_id += 1

        self.mock_trials[tr_id] = {
            "environment_id": environment_id,
            "parameters": parameters,
            "status": "COMPLETED",
            "results": {
                "mock_metric": 0.88,
                "steps": 50,
                "info": f"Mock trial {tr_id} completed.",
            },
        }
        print(
            "MockPadresServer: Trial "{tr_id}' run in env '{environment_id}' with params: {parameters}"
        )
        return {
            "trial_id": tr_id,
            "status": "COMPLETED",
            "environment_id": environment_id,
            "results_summary": {"outcome": "mock_success", "score": 0.88},
        }

    def _mock_get_results(self, trial_id: str) -> Dict[str, Any]:
        if trial_id not in self.mock_trials:
            return {
                "error": f"Mock trial {trial_id} not found.",
                "status": "failure",
                "trial_id": trial_id,
            }

        trial_data = self.mock_trials[trial_id]
        print("MockPadresServer: Results requested for trial "{trial_id}'")
        return {
            "trial_id": trial_id,
            "status": trial_data["status"],
            "results": trial_data.get("results"),
        }

    def _mock_get_status(self) -> Dict[str, Any]:
        return {
            "api_status": "OPERATIONAL_MOCK",
            "mock_mode_active": True,
            "active_environments": len(self.mock_environments),
            "completed_trials": len(self.mock_trials),
            "server_name": self.server_name,
        }

    # --- BaseMCPServer method implementations ---

    def get_status(self) -> dict:
        """
        Gets the status of the PadresMCPServer.
        If not in mock_mode, this method is synchronous and cannot call the async _padres_get_status.
        It will return a basic status. For live status, use async_call_tool with 'get_padres_api_status'.
        """
        if self.mock_mode:
            return self._mock_get_status()
        else:
            client_status = (
                "initialized" if self.http_client else "not_initialized_or_error"
            )
            return {
                "status": "available_for_async_calls",
                "mode": "live",
                "service_url": self.service_base_url,
                "client_status": client_status,
                "server_name": self.server_name,
                "note": "For live API status, use async_call_tool with 'get_padres_api_status'.",
            }

    async def async_call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        Asynchronously calls a tool on the PadresMCPServer.
        This is the primary method for interacting with this server due to its async nature.
        """
        print(
            "PadresMCPServer ("{self.server_name}', mock_mode={self.mock_mode}) async_call_tool: '{tool_name}' with params: {parameters}"
        )

        if self.mock_mode:
            if tool_name == "setup_environment":
                # Parameters: {"config": dict, "environment_id": Optional[str]}
                return self._mock_setup_environment(
                    config_payload=parameters.get("config", {}),
                    environment_id=parameters.get("environment_id"),
                )
            elif tool_name == "run_trial":
                # Parameters: {"environment_id": str, "parameters": dict, "trial_id": Optional[str]}
                return self._mock_run_trial(
                    environment_id=parameters.get("environment_id"),
                    parameters=parameters.get("parameters", {}),
                    trial_id=parameters.get("trial_id"),
                )
            elif tool_name == "get_results":
                # Parameters: {"trial_id": str}
                return self._mock_get_results(trial_id=parameters.get("trial_id"))
            elif tool_name == "get_padres_api_status":  # Renamed for clarity
                return self._mock_get_status()
            else:
                return {
                    "error": "Tool "{tool_name}' not supported in mock mode for PadresMCPServer.",
                    "status": "failure",
                }
        else:  # Live mode, use HTTP client
            if (
                not self.http_client and tool_name != "get_padres_api_status"
            ):  # Allow status check even if client failed init, API might be down
                # Re-initialize client if it's None and we are not in mock_mode
                print(
                    "PadresMCPServer "{self.server_name}': http_client is None. Attempting to re-initialize."
                )
                self._initialize_client()
                if not self.http_client:
                    return {
                        "error": "HTTP client for PadresMCPServer "{self.server_name}' is not initialized. Cannot call tool '{tool_name}'.",
                        "status": "failure",
                    }

            if tool_name == "setup_environment":
                # Parameters: {"config": dict, "environment_id": Optional[str]}
                return await self._padres_setup_environment(
                    config=parameters.get("config", {}),
                    environment_id=parameters.get("environment_id"),
                )
            elif tool_name == "run_trial":
                # Parameters: {"environment_id": str, "parameters": dict, "trial_id": Optional[str]}
                return await self._padres_run_trial(
                    environment_id=parameters.get("environment_id"),
                    parameters=parameters.get("parameters", {}),
                    trial_id=parameters.get("trial_id"),
                )
            elif tool_name == "get_results":
                # Parameters: {"trial_id": str}
                return await self._padres_get_results(
                    trial_id=parameters.get("trial_id")
                )
            elif tool_name == "get_padres_api_status":
                if not self.http_client:  # If client never initialized or failed
                    # Try to initialize it for a status check, or return error if still fails.
                    print(
                        "PadresMCPServer "{self.server_name}': http_client is None for status check. Attempting to re-initialize."
                    )
                    self._initialize_client()
                    if not self.http_client:
                        return {
                            "error": "Cannot get live API status for "{self.server_name}'. HTTP client failed to initialize.",
                            "status": "failure",
                            "service_base_url": self.service_base_url,
                        }
                return await self._padres_get_status()
            else:
                return {
                    "error": "Tool "{tool_name}' not supported in live mode for PadresMCPServer.",
                    "status": "failure",
                }

    def call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        Synchronous tool call. For PadresMCPServer, most operations are async.
        This method will raise an error or return a message indicating to use async_call_tool.
        The get_status tool is an exception and can be called synchronously.
        """
        if tool_name == "get_status":  # This is the synchronous BaseMCPServer method
            return self.get_status()

        # All other tools specific to Padres should be async
        # For mock mode, some could technically be sync, but we'll enforce async for consistency
        # with live mode.
        if self.mock_mode:
            # Even in mock, we'll suggest async_call_tool for the specific Padres operations
            # to maintain a consistent interface.
            if tool_name in [
                "setup_environment",
                "run_trial",
                "get_results",
                "get_padres_api_status",
            ]:
                return {
                    "error": "Tool "{tool_name}' on PadresMCPServer '{self.server_name}' (mock mode) should be called via async_call_tool for consistency.",
                    "status": "hint",
                    "suggestion": "Use async_call_tool.",
                }
            else:
                return {
                    "error": "Tool "{tool_name}' not recognized or not synchronously callable on PadresMCPServer '{self.server_name}'.",
                    "status": "failure",
                }
        else:  # Live mode
            return {
                "error": "Tool "{tool_name}' on PadresMCPServer '{self.server_name}' (live mode) must be called via async_call_tool.",
                "status": "failure",
                "suggestion": "Use async_call_tool.",
            }

    async def close_client(self):
        """Closes the HTTP client if it was initialized."""
        if self.http_client:
            try:
                await self.http_client.aclose()
                print(
                    "HTTP AsyncClient closed for PadresMCPServer "{self.server_name}'."
                )
            except Exception as e:
                print(
                    "Error closing HTTP client for PadresMCPServer "{self.server_name}': {e}"
                )
            finally:
                self.http_client = None
        else:
            print(
                "No active HTTP client to close for PadresMCPServer "{self.server_name}' (mock_mode: {self.mock_mode})."
            )


# Example usage (main_test)
async def main_test():
    print("--- Testing PadresMCPServer ---")

    # Test Mock Mode
    print("\n--- Mock Mode Test ---")
    mock_config = {
        "mock_mode": True,
        # No service_base_url needed for mock mode
    }
    mock_padres_server = PadresMCPServer(
        server_name="padres_mock_test", config=mock_config
    )

    print(f"Mock Server Status: {mock_padres_server.get_status()}")
    print(
        f"Mock Server Status (async): {await mock_padres_server.async_call_tool('get_padres_api_status', {})}"
    )

    env_setup_params_mock = {"config": {"scene": "kitchen_v1", "max_steps": 100}}
    setup_result_mock = await mock_padres_server.async_call_tool(
        "setup_environment", env_setup_params_mock
    )
    print(f"Mock Setup Environment Result: {json.dumps(setup_result_mock, indent=2)}")

    mock_env_id = setup_result_mock.get("environment_id")
    if mock_env_id:
        trial_params_mock = {
            "environment_id": mock_env_id,
            "parameters": {"agent_type": "random", "num_episodes": 1},
        }
        trial_result_mock = await mock_padres_server.async_call_tool(
            "run_trial", trial_params_mock
        )
        print(f"Mock Run Trial Result: {json.dumps(trial_result_mock, indent=2)}")

        mock_trial_id = trial_result_mock.get("trial_id")
        if mock_trial_id:
            results_mock = await mock_padres_server.async_call_tool(
                "get_results", {"trial_id": mock_trial_id}
            )
            print(f"Mock Get Results: {json.dumps(results_mock, indent=2)}")

    await mock_padres_server.close_client()  # Should indicate no client to close

    # Test Live Mode (requires the mock Padres API service from padres_container/app/main.py to be running)
    # To run it:
    # 1. cd padres_container
    # 2. (Optional, if you use a venv for the container's app) source venv/bin/activate
    # 3. python app/main.py  (or uvicorn app.main:app --host 0.0.0.0 --port 8088)
    # Ensure PYTHONPATH is set for this script to run if you execute it directly.
    print("\n--- Live Mode Test (requires mock API at http://localhost:8088) ---")
    live_config = {
        "mock_mode": False,
        "service_base_url": "http://localhost:8088",  # Target the new mock API
    }
    live_padres_server = None
    try:
        live_padres_server = PadresMCPServer(
            server_name="padres_live_test", config=live_config
        )
        print(f"Live Server Status (sync): {live_padres_server.get_status()}")

        # Test get_padres_api_status
        api_status = await live_padres_server.async_call_tool(
            "get_padres_api_status", {}
        )
        print(f"Live API Status from service: {json.dumps(api_status, indent=2)}")

        if (
            api_status.get("api_status") == "OPERATIONAL"
        ):  # Check if the mock API is running
            env_setup_params_live = {
                "config": {"scene": "office_v2", "complexity": "high"}
            }
            setup_result_live = await live_padres_server.async_call_tool(
                "setup_environment", env_setup_params_live
            )
            print(
                f"Live Setup Environment Result: {json.dumps(setup_result_live, indent=2)}"
            )
            live_env_id = setup_result_live.get("environment_id")

            if live_env_id and not setup_result_live.get("error"):
                trial_params_live = {
                    "environment_id": live_env_id,
                    "parameters": {"algorithm": "PPO", "seed": 123},
                }
                trial_result_live = await live_padres_server.async_call_tool(
                    "run_trial", trial_params_live
                )
                print(
                    f"Live Run Trial Result: {json.dumps(trial_result_live, indent=2)}"
                )
                live_trial_id = trial_result_live.get("trial_id")

                if live_trial_id and not trial_result_live.get("error"):
                    results_live = await live_padres_server.async_call_tool(
                        "get_results", {"trial_id": live_trial_id}
                    )
                    print(f"Live Get Results: {json.dumps(results_live, indent=2)}")
            else:
                print(
                    "Skipping further live tests as environment setup failed or returned an error."
                )
        else:
            print(
                "Skipping live API interaction tests as mock API status is not OPERATIONAL or fetch failed."
            )
            print(
                "Ensure padres_container/app/main.py is running on http://localhost:8088."
            )

    except ValueError as ve:
        print(f"ValueError during live_padres_server setup: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during live mode test: {e}")
    finally:
        if live_padres_server:
            await live_padres_server.close_client()


if __name__ == "__main__":
    import asyncio

    # To run this main_test, ensure you are in the project root and use:
    # python -m backend_services.mcp_servers.padres_mcp_server
    # Make sure padres_container/app/main.py is also running for the live test part.
    asyncio.run(main_test())
