# backend_services/mcp_servers/gemini_mcp_server.py

import json  # For example usage block
import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

from .base_mcp_server import BaseMCPServer


class GeminiMCPServer(BaseMCPServer):
    """
    MCP Server implementation for interacting with Google's Gemini models via the google-generativeai SDK.
    """

    def __init__(self, server_name: str, config: dict):
        """
        Initializes the GeminiMCPServer.
        Expected config keys:
            - "api_key": Your Google AI Studio API key for Gemini.
                         (Can also be sourced from GOOGLE_API_KEY environment variable by the SDK).
            - "model_name": The specific Gemini model to use (e.g., "gemini-pro", "gemini-1.5-pro-latest").
            - "generation_config" (optional): Dict for default GenerationConfig (e.g., {"temperature": 0.7, "max_output_tokens": 1024, "top_p": 0.9, "top_k": 40}).
            - "safety_settings" (optional): List of dicts for default SafetySettings (e.g., [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}]).
        """
        self.model: Optional[genai.GenerativeModel] = None
        self.configured_api_key_source: Optional[str] = None
        super().__init__(server_name, config)

    def _validate_config(self):
        """Validates that 'model_name' is in the config. API key is checked during SDK configuration."""
        if "model_name" not in self.config:
            raise ValueError(
                "Missing "model_name' in config for GeminiMCPServer: {self.server_name}"
            )

        # Prioritize API key from config, then ENV_ prefixed config, then GOOGLE_API_KEY env var
        config_api_key = self.config.get("api_key")
        env_var_name_in_config = None

        if isinstance(config_api_key, str) and config_api_key.startswith("ENV_"):
            env_var_name_in_config = config_api_key[4:]  # Get the actual env var name
            print(
                "  GeminiMCPServer "{self.server_name}': Config requests API key from env var '{env_var_name_in_config}'."
            )
            loaded_env_api_key = os.environ.get(env_var_name_in_config)
            if loaded_env_api_key:
                self.config["api_key"] = (
                    loaded_env_api_key  # Replace placeholder with actual key
                )
                self.configured_api_key_source = (
                    f"environment ({env_var_name_in_config}) via config"
                )
            elif os.environ.get(
                "GOOGLE_API_KEY"
            ):  # Fallback to standard GOOGLE_API_KEY if specific ENV_ one not found
                self.config["api_key"] = os.environ.get("GOOGLE_API_KEY")
                self.configured_api_key_source = "environment (GOOGLE_API_KEY fallback)"
                print(
                    "  GeminiMCPServer "{self.server_name}': '{env_var_name_in_config}' not set, using GOOGLE_API_KEY fallback."
                )
            else:
                print(
                    "Warning: GeminiMCPServer "{self.server_name}': API key specified as ENV_ variable '{env_var_name_in_config}' in config, but it's not set, and GOOGLE_API_KEY is also not set."
                )
                # SDK will likely raise an error later if no key is configured
        elif config_api_key:  # Direct API key in config
            self.configured_api_key_source = "config_direct"
        elif os.environ.get(
            "GOOGLE_API_KEY"
        ):  # No api_key in config, check standard env var
            self.config["api_key"] = os.environ.get(
                "GOOGLE_API_KEY"
            )  # Store for SDK configure()
            self.configured_api_key_source = "environment (GOOGLE_API_KEY)"
        else:
            print(
                "Warning: GeminiMCPServer "{self.server_name}': No API key found in config or GOOGLE_API_KEY env var. SDK initialization may fail."
            )

    def _initialize_client(self):
        """Initializes the Gemini client (configures the SDK and gets a model instance)."""
        try:
            # SDK's genai.configure() uses GOOGLE_API_KEY env var by default if no api_key is passed.
            # If self.config["api_key"] has been populated (either directly or from an ENV_ var),
            # passing it to genai.configure() overrides the env var for this specific configuration instance.
            api_key_to_use = self.config.get("api_key")
            if api_key_to_use:
                genai.configure(api_key=api_key_to_use)
                print(
                    "Gemini SDK configured for "{self.server_name}' using API key from: {self.configured_api_key_source or 'unknown'}. "
                )
            elif (
                self.configured_api_key_source
                and "GOOGLE_API_KEY" in self.configured_api_key_source
            ):  # Relies on env GOOGLE_API_KEY set previously
                print(
                    "Gemini SDK for "{self.server_name}' will use GOOGLE_API_KEY from environment (source: {self.configured_api_key_source})."
                )
                # No explicit genai.configure(api_key=None) call needed as SDK handles env var.
            else:
                # This case implies no key was found anywhere, SDK will raise an error.
                print(
                    "Attempting to initialize Gemini model for "{self.server_name}' without explicit API key; SDK will search GOOGLE_API_KEY or fail."
                )

            model_name = self.config["model_name"]

            raw_gen_config = self.config.get("generation_config")
            generation_config_sdk: Optional[GenerationConfig] = None
            if isinstance(raw_gen_config, dict):
                generation_config_sdk = genai.types.GenerationConfig(**raw_gen_config)

            raw_safety_settings = self.config.get("safety_settings")
            safety_settings_sdk: Optional[List[SafetySetting]] = None
            if isinstance(raw_safety_settings, list):
                safety_settings_sdk = []
                for setting in raw_safety_settings:
                    if (
                        isinstance(setting, dict)
                        and "category" in setting
                        and "threshold" in setting
                    ):
                        try:
                            category_enum = HarmCategory[setting["category"]]
                            threshold_enum = HarmBlockThreshold[setting["threshold"]]
                            safety_settings_sdk.append(
                                SafetySetting(
                                    category=category_enum, threshold=threshold_enum
                                )
                            )
                        except KeyError as ke:
                            print(
                                f"Warning: Invalid HarmCategory or HarmBlockThreshold string in safety_settings for {self.server_name}: {ke}. Setting will be skipped."
                            )
                    else:
                        print(
                            f"Warning: Invalid safety_setting format for {self.server_name}, skipping: {setting}"
                        )

            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config_sdk,
                safety_settings=safety_settings_sdk,
            )
            print(
                "Gemini GenerativeModel "{model_name}' initialized for server '{self.server_name}'."
            )

            # Test with a count_tokens call
            test_token_count = self.model.count_tokens("test initialization")
            print(
                "  Successful test: count_tokens("test initialization') = {test_token_count.total_tokens} for {self.server_name}"
            )

        except Exception as e:
            print(
                f"CRITICAL: Error initializing Gemini client/model for {self.server_name}: {e}"
            )
            self.model = None
            # Depending on requirements, you might want to raise an exception here to halt startup
            # if this MCP server is critical.
            # raise

    def _generate_content(
        self,
        prompt_or_messages: Any,
        generation_config_override: Optional[Dict] = None,
        safety_settings_override: Optional[List[Dict]] = None,
        stream: bool = False,
    ) -> Dict:
        """
        Generates content using the Gemini model.
        `prompt_or_messages` can be a string or a list of content parts/messages as per Gemini SDK.
        Returns a dictionary with the response text, finish reason, and other metadata.
        """
        if not self.model:
            return {
                "error": f"Gemini model not initialized for {self.server_name}. Cannot generate content.",
                "status": "failure",
            }

        print(
            f"  {self.server_name}._generate_content (model: {self.model.model_name}) called. Stream: {stream}"
        )
        if isinstance(prompt_or_messages, str):
            print("    Prompt (first 50 chars): "{prompt_or_messages[:50]}...'")
        else:  # Assuming list of parts or messages
            print(
                f"    Content/Messages (type: {type(prompt_or_messages)}, count: {len(prompt_or_messages) if isinstance(prompt_or_messages, list) else 'N/A'})"
            )

        gen_config_sdk_override: Optional[GenerationConfig] = None
        if isinstance(generation_config_override, dict):
            gen_config_sdk_override = genai.types.GenerationConfig(
                **generation_config_override
            )

        safety_sdk_override: Optional[List[SafetySetting]] = None
        if isinstance(safety_settings_override, list):
            safety_sdk_override = []
            for setting in safety_settings_override:
                if (
                    isinstance(setting, dict)
                    and "category" in setting
                    and "threshold" in setting
                ):
                    try:
                        safety_sdk_override.append(
                            SafetySetting(
                                category=HarmCategory[setting["category"]],
                                threshold=HarmBlockThreshold[setting["threshold"]],
                            )
                        )
                    except KeyError:
                        pass  # Skip invalid

        try:
            if stream:
                # For now, MCP tool simulates stream by concatenating. True streaming would need a different return type.
                print(
                    "    Streaming requested. SDK will stream, this method will concatenate for a single response."
                )
                response_chunks = self.model.generate_content(
                    prompt_or_messages,
                    generation_config=gen_config_sdk_override,
                    safety_settings=safety_sdk_override,
                    stream=True,
                )
                full_text = []
                final_response_candidate = None
                for chunk in response_chunks:
                    if chunk.parts:
                        full_text.append(
                            chunk.text
                        )  # chunk.text directly for convenience
                    final_response_candidate = (
                        chunk.candidates[0] if chunk.candidates else None
                    )  # Keep the last candidate info

                response_text = "".join(full_text)
                finish_reason = (
                    final_response_candidate.finish_reason.name
                    if final_response_candidate
                    and final_response_candidate.finish_reason
                    else "STREAM_END_UNKNOWN"
                )
                safety_ratings = (
                    str(final_response_candidate.safety_ratings)
                    if final_response_candidate
                    and final_response_candidate.safety_ratings
                    else "N/A"
                )
            else:
                response = self.model.generate_content(
                    prompt_or_messages,
                    generation_config=gen_config_sdk_override,
                    safety_settings=safety_sdk_override,
                    stream=False,
                )
                response_text = ""
                try:
                    response_text = response.text
                except ValueError as ve:
                    print(f"    ValueError accessing response.text: {ve}")
                    if (
                        response.prompt_feedback
                        and response.prompt_feedback.block_reason
                    ):
                        return {
                            "text": "",
                            "status": "blocked",
                            "error": "Content generation blocked by safety filter",
                            "finish_reason": "SAFETY",
                            "prompt_feedback": str(response.prompt_feedback),
                            "candidates_info": str(response.candidates),
                        }
                    return {
                        "text": "",
                        "status": "failure",
                        "error": "No content in response or value error",
                        "details": str(ve),
                        "prompt_feedback": str(response.prompt_feedback),
                        "candidates_info": str(response.candidates),
                    }

                finish_reason = (
                    response.candidates[0].finish_reason.name
                    if response.candidates and response.candidates[0].finish_reason
                    else "UNKNOWN"
                )
                safety_ratings = (
                    str(response.candidates[0].safety_ratings)
                    if response.candidates and response.candidates[0].safety_ratings
                    else "N/A"
                )

            return {
                "text": response_text,
                "status": "success",
                "finish_reason": finish_reason,
                "safety_ratings": safety_ratings,
            }

        except Exception as e:
            print(
                f"Error during Gemini content generation for {self.server_name}: {type(e).__name__} - {e}"
            )
            return {
                "text": "",
                "status": "failure",
                "error": f"Gemini API call failed for {self.server_name}",
                "details": str(e),
            }

    def _count_tokens(self, content: Any) -> Dict:
        """
        Counts tokens for the given content using the initialized model.
        Content can be a string, or a list of parts/messages for chat.
        """
        if not self.model:
            return {
                "error": f"Gemini model not initialized for {self.server_name}. Cannot count tokens.",
                "total_tokens": -1,
            }
        try:
            token_count_response = self.model.count_tokens(content)
            return {
                "total_tokens": token_count_response.total_tokens,
                "status": "success",
            }
        except Exception as e:
            print(f"Error counting tokens with Gemini for {self.server_name}: {e}")
            return {
                "error": "Token counting failed",
                "details": str(e),
                "total_tokens": -1,
            }

    def call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        Executes a tool supported by the GeminiMCPServer.
        """
        if tool_name == "generate_content":
            prompt_or_messages = parameters.get("prompt_or_messages")
            if prompt_or_messages is None:
                raise ValueError(
                    "Missing "prompt_or_messages' for 'generate_content' tool in {self.server_name}."
                )
            return self._generate_content(
                prompt_or_messages,
                generation_config_override=parameters.get("generation_config"),
                safety_settings_override=parameters.get("safety_settings"),
                stream=parameters.get("stream", False),
            )
        elif tool_name == "count_tokens":
            content = parameters.get("content")
            if content is None:
                raise ValueError(
                    "Missing "content' for 'count_tokens' tool in {self.server_name}."
                )
            return self._count_tokens(content)
        else:
            raise NotImplementedError(
                "Tool "{tool_name}' is not supported by {self.server_name} ({self.__class__.__name__})."
            )


# Example Usage (conceptual)
if __name__ == "__main__":
    print("Testing GeminiMCPServer with google-generativeai SDK...")
    gemini_api_key_from_env = os.environ.get("GOOGLE_API_KEY")
    config_api_key = None
    if not gemini_api_key_from_env:
        print(
            "  WARNING: GOOGLE_API_KEY environment variable not set. Test may fail if API key is strictly required by SDK."
        )
        config_api_key = "DUMMY_FOR_LOCAL_TEST_NO_ENV_KEY"  # Provide a dummy if not in env, for structure testing
    else:
        print("  Using GOOGLE_API_KEY from environment for testing.")

    sample_gemini_config = {
        "model_name": "gemini-pro",
        "generation_config": {
            "temperature": 0.2,
            "max_output_tokens": 50,
            "top_p": 0.8,
            "top_k": 30,
        },
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ],
    }
    if (
        config_api_key
    ):  # Only add to config if it was set as a dummy (because env var was missing)
        sample_gemini_config["api_key"] = config_api_key

    gemini_server = None
    try:
        gemini_server = GeminiMCPServer(
            server_name="gemini_pro_sdk_test", config=sample_gemini_config
        )

        if gemini_server and gemini_server.model:
            print("\n--- Test: Simple Text Prompt ---")
            gen_params_text = {
                "prompt_or_messages": "What are the first 5 prime numbers? Call should succeed even with dummy key if SDK handles it gracefully for some ops like count, or if it truly allows init without key if env var also missing."
            }
            response_text = gemini_server.call_tool("generate_content", gen_params_text)
            print(
                f"Generate Content (text) Response:\n{json.dumps(response_text, indent=2)}"
            )

            print("\n--- Test: Chat-like Messages ---")
            gen_params_chat = {
                "prompt_or_messages": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": "Hi Gemini, can you write a short poem about code? Call should succeed even with dummy key if SDK handles it gracefully for some ops like count, or if it truly allows init without key if env var also missing."
                            }
                        ],
                    },
                ],
                "generation_config": {"temperature": 0.8, "max_output_tokens": 100},
            }
            response_chat = gemini_server.call_tool("generate_content", gen_params_chat)
            print(
                f"Generate Content (chat) Response :\n{json.dumps(response_chat, indent=2)}"
            )

            print("\n--- Test: Token Counting ---")
            count_params_str = {
                "content": "How many tokens are in this sentence? This should work with a configured model even if API key is dummy/missing for the count itself."
            }
            token_count_str = gemini_server.call_tool("count_tokens", count_params_str)
            print(f"Token count (string): {json.dumps(token_count_str, indent=2)}")

            count_params_chat_content = [
                {"role": "user", "parts": [{"text": "Hello!"}]}
            ]
            token_count_chat = gemini_server.call_tool(
                "count_tokens", {"content": count_params_chat_content}
            )
            print(
                f"Token count (chat content): {json.dumps(token_count_chat, indent=2)}"
            )

            status = gemini_server.get_status()
            print(f"\nServer Status: {status}")
        else:
            print(
                "\nGemini server model did not initialize. Skipping tool tests. Check API key and SDK setup."
            )

    except Exception as e:
        print(f"An unexpected error occurred during GeminiMCPServer test: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nGeminiMCPServer test attempt finished.")
