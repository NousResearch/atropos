from abc import ABC, abstractmethod

class BaseMCPServer(ABC):
    """
    Abstract base class for all MCP (Multi-Controller Piper) server implementations.
    Each MCP server will be responsible for interacting with a specific tool or service
    (e.g., an LLM, a database, a web search API).
    """

    def __init__(self, server_name: str, config: dict):
        """
        Initializes the base MCP server.

        Args:
            server_name (str): A unique name for this server instance (e.g., "claude_opus_llm", "main_bigquery").
            config (dict): Configuration specific to this server instance,
                             such as API keys, model IDs, connection strings, etc.
                             Keys like 'api_key', 'project_id', 'model_name' are common.
        """
        self.server_name = server_name
        self.config = config
        self._validate_config()
        self._initialize_client()
        print(f"MCP Server '{self.server_name}' initialized of type {self.__class__.__name__}.")

    @abstractmethod
    def _validate_config(self):
        """
        Validates the provided configuration to ensure all necessary parameters are present.
        Should raise ValueError if critical configuration is missing.
        """
        pass

    @abstractmethod
    def _initialize_client(self):
        """
        Initializes the actual client or SDK for the target service (e.g., LLM client, DB connector).
        This method will use the validated self.config.
        """
        pass

    @abstractmethod
    def call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        The primary method for interacting with this MCP server.
        It executes a specific "tool" (i.e., a function or capability) offered by this server.

        Args:
            tool_name (str): The name of the tool/function to execute 
                               (e.g., "generate_text", "run_query", "search_web").
            parameters (dict): A dictionary of parameters required by the tool.

        Returns:
            any: The result of the tool execution. The structure of this result will
                 depend on the tool and the underlying service.

        Raises:
            NotImplementedError: If the tool_name is not supported by this server.
            Exception: Any exceptions raised by the underlying service client during execution.
        """
        pass

    def get_server_name(self) -> str:
        """
        Returns the name of this MCP server instance.
        """
        return self.server_name

    def get_status(self) -> dict:
        """
        Optional: Returns the current status of the MCP server (e.g., connectivity, rate limits).
        Default implementation indicates it's a basic, active server.
        """
        return {"server_name": self.server_name, "status": "active", "type": self.__class__.__name__}

# Example of a concrete (though still abstract in functionality) server for an LLM
# This would typically be in its own file like `llm_mcp.py`

# class BaseLLMMCPServer(BaseMCPServer):
#     @abstractmethod
#     def generate_text(self, prompt: str, model_parameters: dict = None) -> str:
#         pass

#     @abstractmethod
#     def count_tokens(self, text: str) -> int:
#         pass

#     def call_tool(self, tool_name: str, parameters: dict) -> any:
#         if tool_name == "generate_text":
#             prompt = parameters.get("prompt")
#             if not prompt:
#                 raise ValueError("'prompt' parameter is required for generate_text tool.")
#             return self.generate_text(prompt, parameters.get("model_parameters"))
#         elif tool_name == "count_tokens":
#             text = parameters.get("text")
#             if not text:
#                 raise ValueError("'text' parameter is required for count_tokens tool.")
#             return self.count_tokens(text)
#         else:
#             raise NotImplementedError(f"Tool '{tool_name}' is not supported by {self.server_name}.") 