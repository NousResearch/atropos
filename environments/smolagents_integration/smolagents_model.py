"""
Process-safe implementation of the AtroposServerModel for SmolaGents.
"""

import logging
import traceback

from smolagents.models import ChatMessage, MessageRole, Model

from .server_proxy import ServerProxy

# Configure logger for the model class
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProcessSafeAtroposServerModel(Model):
    """
    A SmolaGents Model implementation that works with a server proxy.

    This class is designed to be used in separate processes and
    communicates with the Atropos server through a proxy mechanism.
    """

    def __init__(
        self,
        server_proxy: ServerProxy,
        use_chat_completion: bool = False,
        model_id: str = None,
        **kwargs,
    ):
        self.server_proxy = server_proxy
        self.use_chat_completion = use_chat_completion

        # Automatically set chat completion for GPT models which require it
        if model_id and any(
            name in model_id.lower()
            for name in ["gpt-4", "gpt-3.5-turbo", "claude", "gemini", "o", "llama"]
        ):
            logger.info(
                f"Model {model_id} detected as a chat model. Forcing chat completion API."
            )
            self.use_chat_completion = True

        # Log the configuration
        logger.info(
            f"Initializing ProcessSafeAtroposServerModel with model_id={model_id}, "
            f"use_chat_completion={self.use_chat_completion}"
        )

        super().__init__(model_id=model_id, **kwargs)

    def _prepare_completion_args(self, messages, stop_sequences=None, **kwargs):
        """
        Convert SmolaGents message format to Atropos server parameters.
        """

        # Always use chat completion if configured that way
        if self.use_chat_completion:
            # For chat completion, we format messages and don't use prompt
            server_args = {
                "messages": self._format_chat_messages(messages),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.0),
                "stop": stop_sequences,
            }
            logger.debug(
                f"Prepared chat completion args: messages count={len(server_args['messages'])}"
            )
            return server_args
        else:
            # Extract the user message for completion API
            prompt = self._extract_user_message(messages)
            server_args = {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.0),
                "stop": stop_sequences,
            }
            logger.debug(f"Prepared completion args: prompt length={len(prompt)}")
            return server_args

    def _extract_user_message(self, messages):
        """Extract content from the last user message."""
        for msg in reversed(messages):
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    # Handle list format [{"type": "text", "text": "content"}]
                    return "\n".join(
                        item["text"] for item in content if item["type"] == "text"
                    )
                return content
        raise ValueError("No user message found")

    def _format_chat_messages(self, messages):
        """Format messages for the chat completion API."""
        formatted_messages = []

        # For OpenAI API, we need to map roles to the ones they support
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            # Map any role to either system, user, or assistant
            if isinstance(role, str):
                role_str = role.lower()
            elif hasattr(role, "value"):
                role_str = str(role.value).lower()
            else:
                role_str = str(role).lower()

            # Mapping to OpenAI roles
            if role_str == "system":
                openai_role = "system"
            elif role_str == "user":
                openai_role = "user"
            elif role_str == "assistant":
                openai_role = "assistant"
            elif role_str in (
                "tool_call",
                "tool_response",
                "function_call",
                "function_response",
            ):
                # Silently map tool and function calls/responses to user roles
                openai_role = "user"
            else:
                # Default everything else to user without logging
                openai_role = "user"

            # Extract text content if it's in the list format
            if isinstance(content, list):
                text_content = "\n".join(
                    item["text"] for item in content if item["type"] == "text"
                )
                formatted_messages.append(
                    {"role": openai_role, "content": text_content}
                )
            else:
                formatted_messages.append({"role": openai_role, "content": content})

        return formatted_messages

    def generate(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> ChatMessage:
        """
        Process the input messages and return the model's response.
        Uses the server proxy to communicate with the Atropos server.

        Parameters:
            messages: A list of message dictionaries to be processed.
            stop_sequences: A list of strings that will stop the generation if encountered.
            grammar: The grammar or formatting structure to use (not used with Atropos).
            tools_to_call_from: List of tools (not used with Atropos).
            **kwargs: Additional keyword arguments for the server.

        Returns:
            ChatMessage: A chat message object containing the model's response.
        """
        # Special handling for CodeAgent stop sequences
        if stop_sequences is None:
            stop_sequences = ["Observation:", "<end_code>", "Calling tools:"]

        logger.info(
            f"Generate called with {len(messages)} messages, use_chat_completion={self.use_chat_completion}"
        )

        # Prepare completion arguments
        completion_kwargs = self._prepare_completion_args(
            messages=messages, stop_sequences=stop_sequences, **kwargs
        )

        # Extract timeout from kwargs or use default (but not used in this method)
        kwargs.pop("timeout", 120)  # Default 2 minutes

        try:
            # Use chat_completion if configured
            if self.use_chat_completion:
                logger.info("Using chat completion API through proxy")

                # Convert prompt to messages format if needed
                if (
                    "prompt" in completion_kwargs
                    and "messages" not in completion_kwargs
                ):
                    logger.info("Converting prompt to messages format")
                    completion_kwargs["messages"] = [
                        {"role": "user", "content": completion_kwargs.pop("prompt")}
                    ]

                # Call chat_completion via proxy
                resp = self.server_proxy.chat_completion(**completion_kwargs)

                # Process response
                if resp and hasattr(resp, "choices") and len(resp.choices) > 0:
                    content = resp.choices[0].message.content
                    logger.info(f"Got response with {len(content)} chars")
                else:
                    content = "No response content"
                    logger.warning("No content found in response")
            else:
                logger.info("Using completion API through proxy")

                # Call completion via proxy
                resp = self.server_proxy.completion(**completion_kwargs)

                # Process response
                if resp and hasattr(resp, "choices") and len(resp.choices) > 0:
                    content = resp.choices[0].text
                    logger.info(f"Got response with {len(content)} chars")
                else:
                    content = "No response content"
                    logger.warning("No content found in response")

            # Track token usage
            if hasattr(resp, "usage"):
                self.last_input_token_count = resp.usage.prompt_tokens
                self.last_output_token_count = resp.usage.completion_tokens
                logger.info(
                    f"Token usage: input={self.last_input_token_count}, output={self.last_output_token_count}"
                )

            # Return result in SmolaGents format
            logger.info("Successfully returning ChatMessage")
            return ChatMessage(role=MessageRole.ASSISTANT, content=content, raw=resp)

        except Exception as e:
            # Provide more detailed error information
            error_msg = f"Error during server proxy call: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)

            # Print full stack trace for debugging
            logger.error(f"Full traceback: {traceback.format_exc()}")

            raise ValueError(error_msg)
