# backend_services/mcp_manager.py
import importlib
import os
import inspect
import asyncio
import json

class MCPManager:
    """
    A class to provide a standardized interface (call_tool) to various "MCP servers"
    (LLMs, BigQuery, Perplexity, GitHub, Google Docs, Slack, etc.).
    This promotes modularity and allows swapping out tools or services easily.
    """

    def __init__(self, config: dict = None):
        """
        Initializes the MCPManager.
        Args:
            config (dict, optional): Configuration for MCPManager, including server setups.
                                     Expected key: "mcp_server_configurations" as shown in docs.
        """
        self.config = config if config else {}
        self.mcp_servers = {} # Stores instantiated server objects
        self._initialize_servers()

    def _initialize_servers(self):
        """
        Initializes MCP servers based on the "mcp_server_configurations" in self.config.
        Dynamically loads server classes from the .mcp_servers package.
        """
        if "mcp_server_configurations" not in self.config:
            print("Warning: 'mcp_server_configurations' not found in MCPManager config. No servers will be initialized.")
            return

        server_configs = self.config["mcp_server_configurations"]
        # Assuming mcp_servers is a package relative to this file's location
        # For dynamic loading, the package path needs to be correct.
        # If mcp_manager.py is in backend_services, and mcp_servers is a sub-directory,
        # the import path would be .mcp_servers
        servers_package_path = "backend_services.mcp_servers"

        for server_name, conf in server_configs.items():
            module_name = conf.get("module_name")
            class_name = conf.get("class_name")
            specific_config = conf.get("config", {})

            if not module_name or not class_name:
                print(f"Warning: Skipping server '{server_name}' due to missing 'module_name' or 'class_name' in config.")
                continue
            
            try:
                # Construct the full module path for importlib
                full_module_path = f"{servers_package_path}.{module_name}"
                # Dynamically import the module
                module = importlib.import_module(full_module_path)
                # Get the class from the imported module
                ServerClass = getattr(module, class_name)
                # Instantiate the server with its name and specific config
                server_instance = ServerClass(server_name=server_name, config=specific_config)
                self.mcp_servers[server_name] = server_instance
                print(f"Successfully initialized and registered MCP Server: {server_name} of type {class_name}")
            except ImportError as e:
                print(f"Error importing module for server '{server_name}' (module: {full_module_path}): {e}")
            except AttributeError as e:
                print(f"Error: Class '{class_name}' not found in module '{full_module_path}' for server '{server_name}': {e}")
            except Exception as e:
                print(f"Error initializing server '{server_name}' ({class_name}): {e}")

    async def call_tool(self, server_name: str, tool_name: str, parameters: dict) -> any:
        """
        Calls a specific tool on a designated MCP server.

        Args:
            server_name (str): The name of the MCP server (e.g., "claude_llm", "bigquery_data").
            tool_name (str): The name of the tool to call on that server.
            parameters (dict): A dictionary of parameters for the tool.

        Returns:
            any: The result from the tool call.

        Raises:
            ValueError: If the server_name is not found.
            NotImplementedError: If the tool is not supported by the server.
        """
        if server_name not in self.mcp_servers:
            raise ValueError(f"MCP Server '{server_name}' not found or not initialized.")

        server = self.mcp_servers[server_name]
        
        # First, check if the server has an async_call_tool method
        if hasattr(server, "async_call_tool"):
            try:
                return await server.async_call_tool(tool_name, parameters)
            except NotImplementedError:
                # Fall back to call_tool if async_call_tool doesn't support this tool
                pass
        
        # Delegate to the server's call_tool method
        # The specific server is responsible for knowing its tools.
        try:
            # Check if the server's call_tool is awaitable (async)
            server_call_tool_method = getattr(server, "call_tool")
            
            # Use inspect.iscoroutinefunction to check if it's an async method
            if inspect.iscoroutinefunction(server_call_tool_method):
                return await server_call_tool_method(tool_name, parameters)
            else:
                return server_call_tool_method(tool_name, parameters) # Synchronous call
        except NotImplementedError: # Re-raise if tool isn't found on the server
            raise NotImplementedError(f"Tool '{tool_name}' is not supported by server '{server_name}'.")
        except Exception as e:
            # Catch other exceptions from the server's tool call and log/re-raise appropriately
            print(f"Error during call_tool on server '{server_name}' for tool '{tool_name}': {e}")
            raise # Or return a structured error object

    def get_server(self, server_name: str) -> any:
        """
        Retrieves a specific initialized MCP server instance.
        Returns None if not found.
        """
        return self.mcp_servers.get(server_name)

    def list_servers(self) -> list:
        """
        Lists the names of all initialized MCP servers.
        """
        return list(self.mcp_servers.keys())

# Example Usage (conceptual):
if __name__ == "__main__":
    # This demonstrates how MCPManager might be configured and used.
    # For this to run, you need:
    # 1. backend_services/mcp_servers/base_mcp_server.py (created earlier)
    # 2. backend_services/mcp_servers/claude_mcp_server.py (created earlier)
    # 3. An __init__.py in backend_services/mcp_servers/ to make it a package.

    # Create a dummy __init__.py for the mcp_servers package if it doesn't exist
    # This is needed for the dynamic import to work correctly with relative paths.
    # In a real setup, this __init__.py should exist.
    if not os.path.exists("backend_services/mcp_servers/__init__.py"):
        with open("backend_services/mcp_servers/__init__.py", "w") as f:
            f.write("# This file makes mcp_servers a Python package\n")
            print("Created dummy backend_services/mcp_servers/__init__.py for test.")

    sample_manager_config = {
        "mcp_server_configurations": {
            "claude_main": {
                "module_name": "claude_mcp_server",
                "class_name": "ClaudeMCPServer",
                "config": {
                    "api_key": "YOUR_ANTHROPIC_API_KEY_PLACEHOLDER",
                    "model_name": "claude-3-opus-20240229"
                }
            },
            "bq_main_data": {
                "module_name": "bigquery_mcp_server",
                "class_name": "BigQueryMCPServer",
                "config": {
                    "project_id": "your-gcp-project-id-placeholder",
                    "dataset_id": "ai_research_data_main"
                }
            },
            "firestore_config_db": {
                "module_name": "firestore_mcp_server",
                "class_name": "FirestoreMCPServer",
                "config": {
                    "project_id": "your-gcp-project-id-placeholder"
                }
            },
            "padres_env_service": {
                "module_name": "padres_mcp_server",
                "class_name": "PadresMCPServer",
                "config": {
                    "service_base_url": "http://mock-padres-service.example.com/api"
                }
            },
            "pubsub_events": {
                "module_name": "pubsub_mcp_server",
                "class_name": "PubSubMCPServer",
                "config": {
                    "project_id": "your-gcp-project-id-placeholder"
                }
            }
        }
    }

    print("\nInitializing MCPManager with extended server configurations...")
    manager = MCPManager(config=sample_manager_config)

    # Need to run async tests in an event loop
    async def run_mcp_manager_tests(manager_instance):
        print(f"\nAvailable MCP Servers: {manager_instance.list_servers()}")

        if "claude_main" in manager_instance.list_servers():
            print("\nTesting ClaudeMCPServer via MCPManager...")
            try:
                gen_params = {
                    "prompt": "Hello Claude, from MCPManager! Write a haiku about code.",
                    "max_tokens": 30
                }
                response = await manager_instance.call_tool(server_name="claude_main", tool_name="generate_text", parameters=gen_params)
                print(f"MCPManager Response from claude_main.generate_text:\n{response}")

                count_params = {"text": "This is a test."}
                token_count = await manager_instance.call_tool(server_name="claude_main", tool_name="count_tokens", parameters=count_params)
                print(f"MCPManager Response from claude_main.count_tokens: {token_count}")
                
                claude_server_instance = manager_instance.get_server("claude_main")
                if claude_server_instance:
                    print(f"Status of claude_main server: {claude_server_instance.get_status()}")

            except ValueError as ve:
                print(f"ValueError during manager test for claude_main: {ve}")
            except NotImplementedError as nie:
                print(f"NotImplementedError during manager test for claude_main: {nie}")
            except Exception as e:
                print(f"An unexpected error occurred during manager test for claude_main: {e}")
        else:
            print("\nClaude server ('claude_main') was not initialized, skipping tests through manager.")

        if "padres_env_service" in manager_instance.list_servers():
            print("\nTesting PadresMCPServer via MCPManager...")
            try:
                setup_params = {"task_name": "TestPadresTask", "task_params": {"mode": "easy"}}
                setup_response = await manager_instance.call_tool(server_name="padres_env_service", tool_name="setup_environment", parameters=setup_params)
                print(f"MCPManager Response from padres_env_service.setup_environment:\n{json.dumps(setup_response, indent=2)}")
                
                env_id = setup_response.get("environment_id") if isinstance(setup_response, dict) else None
                if env_id and setup_response.get("error") is None:
                    action_params = {"environment_id": env_id, "action_name": "look_around"}
                    action_response = await manager_instance.call_tool(server_name="padres_env_service", tool_name="execute_action", parameters=action_params)
                    print(f"MCPManager Response from padres_env_service.execute_action:\n{json.dumps(action_response, indent=2)}")
            except Exception as e:
                print(f"Error testing padres_env_service: {e}")

        # Add other server tests similarly, using await for call_tool
        # ... (bq_main_data, firestore_config_db tests would also need await)
        # For brevity, only Claude and Padres tests are fully converted to async here.

    asyncio.run(run_mcp_manager_tests(manager))
    print("\nMCPManager conceptual test complete.") 