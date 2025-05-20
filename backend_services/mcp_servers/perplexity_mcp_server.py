# backend_services/mcp_servers/perplexity_mcp_server.py

import os
import json # For example usage block, if needed for printing
from .base_mcp_server import BaseMCPServer
import httpx # For making HTTP API calls
from typing import List, Dict, Any, Optional # For type hinting

class PerplexityMCPServer(BaseMCPServer):
    """
    MCP Server implementation for interacting with Perplexity AI API.
    API Documentation: https://docs.perplexity.ai/reference/post_pplx_chat_completions
    """

    DEFAULT_API_BASE_URL = "https://api.perplexity.ai"
    # Common models: "sonar-small-chat", "sonar-medium-chat", "codellama-70b-instruct", "mistral-7b-instruct"
    # Online models (web-connected): "sonar-small-online", "sonar-medium-online"
    DEFAULT_MODEL_NAME = "sonar-small-chat"

    def __init__(self, server_name: str, config: dict):
        """
        Initializes the PerplexityMCPServer.
        Expected config keys:
            - "api_key" (optional if PERPLEXITY_API_KEY env var set): Your Perplexity AI API key.
            - "model_name" (optional): Specific Perplexity model to use (defaults to sonar-small-chat).
            - "api_base_url" (optional): If the API is not on the standard Perplexity domain.
            - "timeout" (optional): HTTP request timeout in seconds.
        """
        self.http_client: Optional[httpx.Client] = None
        self.configured_api_key_source: Optional[str] = None # To track API key origin
        super().__init__(server_name, config)

    def _validate_config(self):
        """Validates that 'api_key' is available from config, ENV_ prefixed var, or PERPLEXITY_API_KEY env var."""
        config_api_key = self.config.get("api_key")
        env_var_name_in_config = None

        if isinstance(config_api_key, str) and config_api_key.startswith("ENV_"):
            env_var_name_in_config = config_api_key[4:]
            print(f"  PerplexityMCPServer '{self.server_name}': Config requests API key from env var '{env_var_name_in_config}'.")
            loaded_env_api_key = os.environ.get(env_var_name_in_config)
            if loaded_env_api_key:
                self.config["api_key"] = loaded_env_api_key 
                self.configured_api_key_source = f"environment ({env_var_name_in_config}) via config"
            else:
                # Fallback to PERPLEXITY_API_KEY if specific ENV_ var not found
                generic_env_key = os.environ.get("PERPLEXITY_API_KEY")
                if generic_env_key:
                    self.config["api_key"] = generic_env_key
                    self.configured_api_key_source = "environment (PERPLEXITY_API_KEY fallback)"
                    print(f"  PerplexityMCPServer '{self.server_name}': '{env_var_name_in_config}' not set, using PERPLEXITY_API_KEY fallback.")
                else:
                    raise ValueError(f"PerplexityMCPServer '{self.server_name}': API key '{env_var_name_in_config}' (from config) not set, and PERPLEXITY_API_KEY env var also not set.")
        elif config_api_key: # Direct API key in config
            self.configured_api_key_source = "config_direct"
        elif os.environ.get("PERPLEXITY_API_KEY"): # No api_key in config, check standard env var
            self.config["api_key"] = os.environ.get("PERPLEXITY_API_KEY") 
            self.configured_api_key_source = "environment (PERPLEXITY_API_KEY)"
        else:
            raise ValueError(f"Missing 'api_key' in config for PerplexityMCPServer '{self.server_name}' and PERPLEXITY_API_KEY environment variable not set.")
        
        if "model_name" not in self.config or not self.config["model_name"]:
            self.config["model_name"] = self.config.get("model_name", self.DEFAULT_MODEL_NAME)
            print(f"  PerplexityMCPServer '{self.server_name}': Using model_name '{self.config['model_name']}'.")

    def _initialize_client(self):
        """Initializes the HTTP client for Perplexity AI."""
        api_base_url = self.config.get("api_base_url", self.DEFAULT_API_BASE_URL)
        api_key = self.config.get("api_key")

        if not api_key:
            print(f"CRITICAL: No API key available for PerplexityMCPServer '{self.server_name}' during client initialization. Client not created.")
            self.http_client = None
            return

        try:
            self.http_client = httpx.Client(
                base_url=api_base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                },
                timeout=self.config.get("timeout", 60.0) # Increased default timeout for potentially longer PPLX calls
            )
            print(f"HTTP client initialized for PerplexityMCPServer '{self.server_name}' targeting '{api_base_url}'. Key from: {self.configured_api_key_source or 'unknown'}. Model: '{self.config['model_name']}'")
            # A simple way to check if API key is accepted is to make a lightweight call if one exists.
            # Perplexity doesn't have a simple /models or /health. We'll rely on the first actual call to validate.
        except Exception as e:
            print(f"CRITICAL: Error initializing HTTP client for PerplexityMCPServer {self.server_name}: {e}")
            self.http_client = None

    def _perform_chat_completion(self, messages: List[Dict[str, str]], model_override: Optional[str] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Performs a chat completion using the Perplexity API /chat/completions endpoint.
        """
        if not self.http_client:
            return {"error": f"Perplexity client not initialized for {self.server_name}.", "status": "failure"}

        endpoint = "/chat/completions"
        model_to_use = model_override or self.config.get("model_name", self.DEFAULT_MODEL_NAME)

        payload = {
            "model": model_to_use,
            "messages": messages,
            "stream": stream
        }
        # Add optional parameters if they are not None
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        if temperature is not None: payload["temperature"] = temperature
        if top_p is not None: payload["top_p"] = top_p
        # Add any other valid PPLX API parameters from kwargs (e.g., frequency_penalty, presence_penalty)
        # Be careful to only add parameters supported by the PPLX API
        supported_extra_params = ["top_k", "presence_penalty", "frequency_penalty"]
        for k, v in kwargs.items():
            if k in supported_extra_params and v is not None:
                payload[k] = v

        print(f"  {self.server_name}._perform_chat_completion (model: {model_to_use}) called. Stream: {stream}. Endpoint: {self.http_client.base_url}{endpoint}")
        # print(f"    Payload: {json.dumps(payload, indent=2)}") # Can be verbose

        try:
            response = self.http_client.post(endpoint, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            if stream:
                # For an MCP tool, true streaming is difficult to represent in a single return.
                # This mock will return a summary or the first chunk if an iterator were implemented.
                # For now, we return the full (non-streamed like) response data if stream=True was requested but API was called non-streamed.
                # If PPLX SDK/API returns an iterator for stream=True, that needs different handling.
                # The PPLX API for stream=true sends Server-Sent Events (SSE).
                # This method is not designed to consume SSE directly for a single MCP return.
                # It would be better to have stream=False for this MCP tool or a dedicated stream tool.
                print(f"    Stream requested for {self.server_name}, but this tool returns a single aggregated response. Full response data (if stream behaved like non-stream): {str(response_data)[:200]}...")
                # Assuming non-streaming behavior even if stream=True was in payload, 
                # because this function structure isn't set up for yielding SSE events.
                # We fall through to normal processing.
                pass # Fall through to normal processing of the (assumed) single JSON response.

            if response_data.get("choices") and len(response_data["choices"]) > 0:
                first_choice = response_data["choices"][0]
                text_content = first_choice.get("message", {}).get("content", "")
                finish_reason = first_choice.get("finish_reason", "unknown")
                usage = response_data.get("usage", {})
                return {
                    "text": text_content,
                    "status": "success",
                    "model_used": response_data.get("model"),
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "raw_response_id": response_data.get("id")
                }
            else:
                # Handle cases where 'choices' might be missing or empty, e.g. API error structure
                error_detail = response_data.get("detail")
                if error_detail:
                    return {"error": f"Perplexity API error: {error_detail}", "status": "failure", "raw_response": response_data}
                return {"error": "No choices found in Perplexity response", "status": "failure", "raw_response": response_data}

        except httpx.HTTPStatusError as e:
            error_body_text = e.response.text
            try: 
                error_json = e.response.json()
                error_detail = error_json.get("detail") if isinstance(error_json, dict) else error_body_text
                if isinstance(error_detail, dict) and "message" in error_detail:
                    error_detail_msg = error_detail["message"]
                elif isinstance(error_detail, str):
                    error_detail_msg = error_detail
                else:
                    error_detail_msg = str(error_detail) # Fallback
            except json.JSONDecodeError:
                error_detail_msg = error_body_text
            
            print(f"HTTP error calling Perplexity API for {self.server_name}: {e.response.status_code} - {error_detail_msg}")
            return {"error": "Perplexity API HTTP error", "status_code": e.response.status_code, "details": error_detail_msg, "status": "failure"}
        except httpx.RequestError as e:
            print(f"Request error (e.g., network issue) calling Perplexity API for {self.server_name}: {type(e).__name__} - {e}")
            return {"error": "Perplexity API request error", "details": str(e), "status": "failure"}
        except Exception as e:
            print(f"Generic error during Perplexity API call for {self.server_name}: {type(e).__name__} - {e}")
            return {"error": f"Perplexity API call failed for {self.server_name}", "details": str(e), "status": "failure"}

    def call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        Executes a tool supported by the PerplexityMCPServer.
        Tool: "chat_completion" (or alias "search")
          Params:
            - "messages": List[Dict[str,str]] (e.g. [{"role": "user", "content": ...}]) OR
            - "query": str (will be converted to a user message)
            - "model_name" or "model_override" (optional): str
            - "max_tokens" (optional): int
            - "temperature" (optional): float
            - "top_p" (optional): float
            - "stream" (optional): bool (Note: this tool returns aggregated response even if stream=True)
            - "focus" (optional): str (will be prepended as a system message)
            - Other PPLX API params like "top_k", "presence_penalty", "frequency_penalty" can be passed in `parameters`.
        """
        if tool_name == "chat_completion" or tool_name == "search":
            messages = parameters.get("messages")
            query = parameters.get("query")

            if not messages and query:
                messages = [{"role": "user", "content": query}]
            elif not messages:
                raise ValueError(f"Missing 'messages' (or 'query') for '{tool_name}' tool in {self.server_name}.")
            
            # Make a copy to avoid modifying original parameters dict if it's reused
            api_call_params = parameters.copy()
            api_call_params.pop("messages", None)
            api_call_params.pop("query", None)
            api_call_params.pop("focus", None) # Handled separately
            # Ensure known parameter names for _perform_chat_completion are used
            model_override = api_call_params.pop("model_name", api_call_params.pop("model_override", None))
            max_tokens = api_call_params.pop("max_tokens", None)
            temperature = api_call_params.pop("temperature", None)
            top_p = api_call_params.pop("top_p", None)
            stream = api_call_params.pop("stream", False)
            # Remaining items in api_call_params are passed as **kwargs

            focus = parameters.get("focus") 
            if focus and messages:
                system_prompt_content = f"Please focus your search and response on {focus} topics. Be comprehensive yet concise."
                if messages[0]["role"] == "system":
                    messages[0]["content"] = f"{messages[0]['content']}\n{system_prompt_content}"
                else:
                    messages.insert(0, {"role": "system", "content": system_prompt_content})
            
            return self._perform_chat_completion(messages, model_override=model_override, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=stream, **api_call_params)
        else:
            raise NotImplementedError(f"Tool '{tool_name}' is not supported by {self.server_name} ({self.__class__.__name__}).")

# Example Usage (conceptual)
if __name__ == "__main__":
    print("Testing PerplexityMCPServer with httpx...")
    # Test cases should now rely on PERPLEXITY_API_KEY env var primarily, or a direct key in test_config for isolated tests.
    test_api_key_source = "environment (PERPLEXITY_API_KEY)" if os.environ.get("PERPLEXITY_API_KEY") else "config_direct (dummy for test)" 
    test_key_value = os.environ.get("PERPLEXITY_API_KEY") or "DUMMY_PPLX_KEY_FOR_INIT_TEST"

    if not os.environ.get("PERPLEXITY_API_KEY"):
        print(f"  WARNING: PERPLEXITY_API_KEY environment variable not set. Real API calls will fail if dummy key is used.")
    else:
        print(f"  Using PERPLEXITY_API_KEY from environment for testing.")

    sample_pplx_config = {
        "api_key": test_key_value, # This will be used by the server
        "model_name": "sonar-small-chat",
        "timeout": 45.0
    }

    pplx_server = None
    try:
        pplx_server = PerplexityMCPServer(server_name="perplexity_real_test", config=sample_pplx_config)

        if pplx_server and pplx_server.http_client:
            print(f"\n--- Test 1: Simple query using 'search' tool name ---")
            search_params_simple = {
                "query": "What is the capital of France?",
                "model_name": "sonar-small-chat"
            }
            results_simple = pplx_server.call_tool("search", search_params_simple)
            print(f"Simple Query Result Text:\n{results_simple.get('text', results_simple)}")

            print(f"\n--- Test 2: Chat completion with messages ---")
            chat_params = {
                "messages": [
                    {"role": "user", "content": "Explain quantum entanglement in three sentences or less."}
                ],
                "model_name": "sonar-medium-chat",
                "max_tokens": 100,
                "temperature": 0.3
            }
            results_chat = pplx_server.call_tool("chat_completion", chat_params)
            print(f"Chat Completion Result Text:\n{results_chat.get('text', results_chat)}")
            if results_chat.get("status") == "success":
                print(f"Usage: Prompt Tokens: {results_chat.get('usage',{}).get('prompt_tokens')}, Completion Tokens: {results_chat.get('usage',{}).get('completion_tokens')}")

            print(f"\n--- Test 3: Search with 'focus' (prepending system prompt) using online model ---")
            search_params_focus = {
                "query": "Impact of renewable energy adoption on global carbon emissions in 2023.",
                "focus": "scientific journals and official government reports",
                "model_name": "sonar-medium-online"
            }
            results_focus = pplx_server.call_tool("search", search_params_focus)
            print(f"Focused Search Result Text:\n{results_focus.get('text', results_focus)}")
            if results_focus.get("status") == "success":
                print(f"Model used: {results_focus.get('model_used')}")
        else:
            print("\nPerplexityMCPServer HTTP client did not initialize. Skipping tool tests.")

    except ValueError as ve: # Config validation errors
        print(f"ValueError during PPLX test setup: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during PerplexityMCPServer test: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pplx_server and pplx_server.http_client:
            pplx_server.http_client.close()
            print("\n  Closed Perplexity HTTP client.")
        print("\nPerplexityMCPServer test attempt finished.") 