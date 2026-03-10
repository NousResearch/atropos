"""
Managed server wrapper that tracks text sequences with aligned tokens and logprobs.

This wrapper maintains a tree structure of sequences, where:
- Each node represents a complete text sequence (prompt + completion)
- Tokens and logprobs are tracked with proper masking for training
- Branching occurs organically from different contexts and n > 1 completions
"""

import logging
import os
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional

from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import Completion, CompletionChoice
from pydantic import BaseModel

from atroposlib.envs.server_handling.server_baseline import APIServer

logger = logging.getLogger(__name__)


class SequenceNode(BaseModel):
    """
    A node in the sequence tree representing a complete text sequence.

    Attributes:
        full_text: Complete text (prompt + completion)
        tokens: Full token sequence (actual token IDs)
        masked_tokens: Tokens with -100 for prompt positions, actual IDs for completion
        logprobs: Logprobs with 1.0 for prompt positions, actual values for completion
        metadata: Optional metadata (e.g., role information, finish_reason, etc.)
    """

    full_text: str
    tokens: List[int]
    masked_tokens: List[int]
    logprobs: List[float]
    metadata: Optional[Dict[str, Any]] = None


class ManagedServer:
    """
    Wrapper around APIServer that tracks sequences with aligned tokens and logprobs.

    Maintains a tree structure keyed by input text, where each completion creates
    new branches. Provides proper masking for training (prompt tokens masked with -100,
    logprobs set to 1.0).

    Uses the clean tokens_and_logprobs_completion interface internally.
    """

    def __init__(
        self,
        server: APIServer,
        tokenizer: Optional[Any] = None,
        track_tree: bool = False,
        tool_parser: Optional[str] = None,
        preserve_think_blocks: bool = False,
    ):
        """
        Initialize the managed server.

        Args:
            server: The underlying APIServer instance to wrap
            tokenizer: Optional tokenizer for encoding/decoding. If not provided,
                      will attempt to extract from server or create from model name.
            track_tree: If True, maintains a tree structure with parent-child links
                       (for multi-turn RL with per-step advantages). If False (default),
                       maintains a simple list of current nodes that updates in-place.
            tool_parser: Optional vLLM tool parser name (e.g. "hermes", "llama3_json",
                        "mistral", etc.). If provided, enables tool call support in
                        chat_completion(). The parser handles extraction of structured
                        tool calls from raw model output. See
                        ToolParserManager.list_registered() for available parsers.
            preserve_think_blocks: If True, preserves <think> blocks in assistant messages,
                        which are sometimes stripped by chat templates. Defaults to False.
                        Usually not needed, since the chat template should be configured
                        to preserve thinking blocks until a user message arrives.
        """
        self.server = server
        self.tokenizer = tokenizer
        self.track_tree = track_tree
        self._tool_parser_name = tool_parser
        self._translator = None  # Lazy init
        self._preserve_think_blocks = preserve_think_blocks

        # Initialize storage based on mode
        if track_tree:
            self.sequences: Dict[str, SequenceNode] = {}  # Tree mode: dict lookup
        else:
            self.current_nodes: List[SequenceNode] = []  # Default mode: simple list

        # Try to get tokenizer from server if not provided
        if self.tokenizer is None:
            self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize tokenizer from server or model name."""
        # Check if the wrapped server has a tokenizer
        if hasattr(self.server, "tokenizer"):
            self.tokenizer = self.server.tokenizer
        else:
            # Try to create from model name
            try:
                from transformers import AutoTokenizer

                model_name = self.server.config.model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                warnings.warn(
                    f"Could not initialize tokenizer: {e}. "
                    "Sequence tracking will be limited without tokenizer."
                )
                self.tokenizer = None

    def _get_translator(self):
        """Lazily create the ToolCallTranslator when first needed.

        Returns None if tool_parser was not specified or if vLLM is not
        installed (the translator will warn on creation in that case).
        """
        if self._translator is None and self._tool_parser_name and self.tokenizer:
            try:
                from atroposlib.envs.server_handling.tool_call_translator import (
                    ToolCallTranslator,
                )

                self._translator = ToolCallTranslator(
                    tokenizer=self.tokenizer,
                    parser_name=self._tool_parser_name,
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to create ToolCallTranslator: {e}. "
                    "Tool call parsing will be disabled. "
                    "Install vllm to enable structured tool call extraction from model output (pip install vllm "
                    "or pip install 'atroposlib[openai_endpoint]').",
                    stacklevel=2,
                )
                self._tool_parser_name = None  # Don't retry
                return None
        return self._translator

    # Placeholder used to protect <think> blocks from chat templates that strip them
    _THINK_OPEN = "__MNGD_THINK__"
    _THINK_CLOSE = "__MNGD_ENDTHINK__"

    def _convert_messages_to_prompt(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[dict]] = None,
    ) -> str:
        """
        Convert chat messages to prompt text using tokenizer's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions (OpenAI format). Passed to
                  apply_chat_template() so the template can inject tool defs
                  into the system prompt.

        Returns:
            Formatted prompt string
        """
        # If tools are active and we have a translator, convert any assistant
        # messages with tool_calls back to raw text first
        if tools and self._get_translator():
            messages = self._get_translator().convert_messages_for_template(messages)

        if self.tokenizer is None:
            # Fallback: simple concatenation
            return "\n".join([f"{m['role']}: {m.get('content', '')}" for m in messages])

        if hasattr(self.tokenizer, "apply_chat_template"):
            # Only add generation prompt if last message is not from assistant
            add_generation_prompt = (
                len(messages) == 0 or messages[-1].get("role") != "assistant"
            )

            if not self._preserve_think_blocks:
                # Protect <think> blocks in assistant messages — some chat templates
                # (e.g. Qwen3) strip them during re-rendering, which breaks prefix
                # matching for multi-turn sequence extension.
                messages = self._protect_think_blocks(messages)

            # Build kwargs
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            if tools:
                template_kwargs["tools"] = tools

            # Use the tokenizer's chat template
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)

            # Restore <think> blocks
            prompt = prompt.replace(self._THINK_OPEN, "<think>")
            prompt = prompt.replace(self._THINK_CLOSE, "</think>")

            return prompt
        else:
            # Fallback for tokenizers without chat template
            return "\n".join([f"{m['role']}: {m.get('content', '')}" for m in messages])

    def _protect_think_blocks(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Replace <think>...</think> with placeholders in assistant messages.

        Only touches assistant messages that already have content (i.e., messages
        being replayed from prior turns, not the generation prompt). This prevents
        chat templates from stripping or relocating think blocks.
        """
        out = []
        for msg in messages:
            if (
                msg.get("role") == "assistant"
                and msg.get("content")
                and "<think>" in msg["content"]
            ):
                content = msg["content"]
                content = content.replace("<think>", self._THINK_OPEN)
                content = content.replace("</think>", self._THINK_CLOSE)
                out.append({**msg, "content": content})
            else:
                out.append(msg)
        return out

    def _debug_requests_enabled(self) -> bool:
        """Enable verbose request construction logs with ATROPOS_DEBUG_REQUESTS=1."""
        return os.getenv("ATROPOS_DEBUG_REQUESTS", "0") == "1"

    def _find_extending_node(self, input_text: str) -> Optional[SequenceNode]:
        """
        Find a node that this input extends (default mode).

        Args:
            input_text: The input text to check

        Returns:
            The node that input_text extends, or None if no match
        """
        if not self.current_nodes:
            return None

        # Check if any current node's full_text is a prefix of the input
        # This means the input is extending that node
        for node in self.current_nodes:
            if input_text.startswith(node.full_text):
                return node
        return None

    def _compute_input_ids(
        self, input_text: str, extending_node: Optional[SequenceNode]
    ) -> List[int]:
        """
        Compute input_ids for the prompt, using existing tokens if extending.

        Args:
            input_text: The full input prompt text
            extending_node: Node being extended, if any

        Returns:
            List of token IDs to use as input_ids
        """
        if extending_node is not None:
            # Extending an existing sequence - use its tokens + tokenize the new part
            existing_text = extending_node.full_text
            new_text_suffix = input_text[len(existing_text) :]

            # Tokenize only the new suffix (without BOS since we're continuing)
            if new_text_suffix:
                new_tokens = self.tokenizer.encode(
                    new_text_suffix, add_special_tokens=False
                )
                return extending_node.tokens + new_tokens
            else:
                # No new text, just use existing tokens
                return extending_node.tokens.copy()
        else:
            # New sequence - tokenize the whole thing
            return self.tokenizer.encode(input_text, add_special_tokens=True)

    def _find_parent_node(self, input_text: str) -> Optional[SequenceNode]:
        """
        Find a parent node whose full_text matches the input_text (tree mode).

        Args:
            input_text: The input text to search for

        Returns:
            Parent SequenceNode if found, None otherwise
        """
        return self.sequences.get(input_text, None)

    def _create_sequence_node(
        self,
        input_text: str,
        parent_node: Optional[SequenceNode],
        prompt_tokens: List[int],
        output_tokens: List[int],
        output_logprobs: List[float],
        completion_text: str,
        finish_reason: str = "stop",
        extending_node: Optional[SequenceNode] = None,
    ) -> SequenceNode:
        """
        Create a sequence node with proper masking.

        Args:
            input_text: The input prompt text
            parent_node: Parent node to extend from (tree mode)
            prompt_tokens: Token IDs for the prompt
            output_tokens: Token IDs for the output/completion
            output_logprobs: Logprobs for output tokens
            completion_text: The completion text
            finish_reason: Finish reason from server
            extending_node: Node being extended (default mode). When provided,
                carries forward its masked_tokens and logprobs so previous
                completions stay unmasked across multi-turn extensions.

        Returns:
            SequenceNode with properly masked tokens and logprobs
        """
        # Combine text
        full_text = input_text + completion_text

        # Pad logprobs to match token length if needed
        if len(output_logprobs) < len(output_tokens):
            output_logprobs = output_logprobs + [1.0] * (
                len(output_tokens) - len(output_logprobs)
            )
        elif len(output_logprobs) > len(output_tokens):
            output_logprobs = output_logprobs[: len(output_tokens)]

        # If we have a parent node (tree mode), use its tokens as the prompt base
        if parent_node is not None:
            prompt_tokens = parent_node.tokens.copy()

        # Combine tokens
        full_tokens = prompt_tokens + output_tokens

        if extending_node is not None:
            # Carry forward the extending node's mask and logprobs.
            # The prompt_tokens = extending_node.tokens + new_suffix_tokens.
            # We preserve the extending node's mask (which has previous
            # completions unmasked) and mask only the new suffix as prompt.
            suffix_len = len(prompt_tokens) - len(extending_node.tokens)
            masked_tokens = (
                extending_node.masked_tokens + [-100] * suffix_len + output_tokens
            )
            full_logprobs = (
                extending_node.logprobs + [1.0] * suffix_len + output_logprobs
            )
        else:
            # Fresh node — mask entire prompt
            prompt_len = len(prompt_tokens)
            masked_tokens = [-100] * prompt_len + output_tokens
            full_logprobs = [1.0] * prompt_len + output_logprobs

        return SequenceNode(
            full_text=full_text,
            tokens=full_tokens,
            masked_tokens=masked_tokens,
            logprobs=full_logprobs,
            metadata={"finish_reason": finish_reason},
        )

    async def chat_completion(self, **kwargs) -> ChatCompletion:
        """
        Intercept chat completion call and track sequences.

        Internally converts to prompt, calls tokens_and_logprobs_completion,
        tracks the sequence, and reconstructs a ChatCompletion response.

        Supports tool calling when a tool_parser was provided at init:
        - Accepts `tools` and `tool_choice` kwargs
        - Converts inbound assistant tool_call messages to raw text
        - Parses outbound model output for tool calls
        - Returns ChatCompletion with proper tool_calls in choices
        - Preserves raw text in tracked nodes (tool parsing is response-only)

        Args:
            **kwargs: Standard chat completion kwargs (messages, n, max_tokens,
                temperature, tools, tool_choice, etc.)

        Returns:
            ChatCompletion response (with tool_calls if detected)
        """
        # Extract tool-related kwargs
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        has_tools = bool(tools) and self._get_translator() is not None

        # Default tool_choice to "auto" if tools provided
        if has_tools and tool_choice is None:
            tool_choice = "auto"

        # Get input text — passes tools for template rendering and
        # handles reconstruction of inbound tool_call messages
        messages = kwargs.get("messages", [])
        prompt = self._convert_messages_to_prompt(
            messages, tools=tools if has_tools else None
        )

        # Handle parent node and extending logic based on mode
        if self.track_tree:
            # Tree mode: look up parent in dict
            parent_node = self._find_parent_node(prompt)
            extending_node = None
        else:
            # Default mode: check if extending existing sequence
            extending_node = self._find_extending_node(prompt)
            parent_node = None  # Don't use parent merging in default mode

        # Convert to completion format
        completion_kwargs = kwargs.copy()
        completion_kwargs["prompt"] = prompt
        completion_kwargs.pop("messages", None)
        if self._debug_requests_enabled():
            msg_count = len(messages)
            prompt_preview = prompt.replace("\n", "\\n")[:600]
            logger.debug(
                "[ATROPOS_REQ_DEBUG] chat_completion messages=%s n=%s max_tokens=%s temperature=%s tools=%s",
                msg_count,
                completion_kwargs.get("n"),
                completion_kwargs.get("max_tokens"),
                completion_kwargs.get("temperature"),
                bool(tools),
            )
            logger.debug("[ATROPOS_REQ_DEBUG] prompt_preview=%r", prompt_preview)

        # Set model name if not provided
        if "model" not in completion_kwargs:
            completion_kwargs["model"] = self.server.config.model_name

        # Compute input_ids (using existing tokens if extending)
        if not self.track_tree and self.tokenizer is not None:
            input_ids = self._compute_input_ids(prompt, extending_node)
            completion_kwargs["input_ids"] = input_ids

        # Call the tokens and logprobs wrapper directly
        (
            prompt_tokens,
            output_tokens_list,
            output_logprobs_list,
            finish_reasons,
        ) = await self.server.tokens_and_logprobs_completion(**completion_kwargs)

        # Track each completion and build choices
        n = len(output_tokens_list)
        choices = []

        for i in range(n):
            output_tokens = output_tokens_list[i]
            output_logprobs = output_logprobs_list[i]
            finish_reason_raw = finish_reasons[i] if i < len(finish_reasons) else "stop"

            # Extract finish_reason string from dict if needed
            if isinstance(finish_reason_raw, dict):
                finish_reason = finish_reason_raw.get("type", "stop")
            else:
                finish_reason = finish_reason_raw

            # Decode completion text — use skip_special_tokens=False when
            # tools are active so <tool_call> tags aren't stripped
            if self.tokenizer is not None:
                completion_text = self.tokenizer.decode(
                    output_tokens,
                    skip_special_tokens=not has_tools,
                )
            else:
                completion_text = "".join([chr(t) for t in output_tokens if t > 31])

            # Create and store sequence node — always uses the raw text,
            # tool parsing only affects the ChatCompletion response
            node = self._create_sequence_node(
                input_text=prompt,
                parent_node=parent_node,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                output_logprobs=output_logprobs,
                completion_text=completion_text,
                finish_reason=finish_reason,
                extending_node=extending_node,
            )

            # Store node based on mode
            if self.track_tree:
                # Tree mode: key by full text in dict
                self.sequences[node.full_text] = node
            else:
                # Default mode: replace if extending, append if new context
                if extending_node is not None:
                    # Replace the extending node with the new extended version
                    try:
                        idx = self.current_nodes.index(extending_node)
                        self.current_nodes[idx] = node
                    except ValueError:
                        # Extending node not in list anymore, just append
                        self.current_nodes.append(node)
                else:
                    # New context - append to list
                    self.current_nodes.append(node)

            # Parse tool calls from raw output if tools are active
            tool_calls_parsed = None
            content_for_response = completion_text
            if has_tools and tool_choice != "none":
                translator = self._get_translator()
                content_for_response, tool_calls_parsed, finish_reason = (
                    translator.parse_model_output(
                        raw_text=completion_text,
                        tool_choice=(
                            tool_choice if isinstance(tool_choice, str) else "auto"
                        ),
                        tools=tools,
                    )
                )

            # Build choice
            message_kwargs = {
                "content": content_for_response,
                "role": "assistant",
            }
            # Note: openai's ChatCompletionMessage model handles tool_calls
            # but we can't pass them through the constructor easily. We'll
            # attach them after construction if needed.
            choice = Choice(
                finish_reason=finish_reason,
                index=i,
                message=ChatCompletionMessage(**message_kwargs),
            )

            # Attach tool_calls to the message if present
            if tool_calls_parsed:
                choice.message.tool_calls = [
                    # Convert vLLM ToolCall to openai ToolCall format
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls_parsed
                ]

            choices.append(choice)

        # Construct ChatCompletion response
        return ChatCompletion(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=self.server.config.model_name,
            object="chat.completion",
            choices=choices,
        )

    async def completion(self, **kwargs) -> Completion:
        """
        Intercept completion call and track sequences.

        Uses tokens_and_logprobs_completion internally, tracks the sequence,
        and reconstructs a Completion response.

        Args:
            **kwargs: Standard completion kwargs (prompt, n, etc.)

        Returns:
            Completion response
        """
        # Get input text
        prompt = kwargs.get("prompt", "")

        # Handle parent node and extending logic based on mode
        if self.track_tree:
            # Tree mode: look up parent in dict
            parent_node = self._find_parent_node(prompt)
            extending_node = None
        else:
            # Default mode: check if extending existing sequence
            extending_node = self._find_extending_node(prompt)
            parent_node = None  # Don't use parent merging in default mode

        # Set model name if not provided
        if "model" not in kwargs:
            kwargs["model"] = self.server.config.model_name

        # Compute input_ids (using existing tokens if extending)
        if not self.track_tree and self.tokenizer is not None:
            input_ids = self._compute_input_ids(prompt, extending_node)
            kwargs["input_ids"] = input_ids

        # Call the tokens and logprobs wrapper directly
        (
            prompt_tokens,
            output_tokens_list,
            output_logprobs_list,
            finish_reasons,
        ) = await self.server.tokens_and_logprobs_completion(**kwargs)

        # Track each completion and build choices
        n = len(output_tokens_list)
        choices = []

        for i in range(n):
            output_tokens = output_tokens_list[i]
            output_logprobs = output_logprobs_list[i]
            finish_reason_raw = finish_reasons[i] if i < len(finish_reasons) else "stop"

            # Extract finish_reason string from dict if needed
            if isinstance(finish_reason_raw, dict):
                finish_reason = finish_reason_raw.get("type", "stop")
            else:
                finish_reason = finish_reason_raw

            # Decode completion text
            if self.tokenizer is not None:
                completion_text = self.tokenizer.decode(
                    output_tokens, skip_special_tokens=True
                )
            else:
                completion_text = "".join([chr(t) for t in output_tokens if t > 31])

            # Create and store sequence node
            node = self._create_sequence_node(
                input_text=prompt,
                parent_node=parent_node,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                output_logprobs=output_logprobs,
                completion_text=completion_text,
                finish_reason=finish_reason,
            )

            # Store node based on mode
            if self.track_tree:
                # Tree mode: key by full text in dict
                self.sequences[node.full_text] = node
            else:
                # Default mode: replace if extending, append if new context
                if extending_node is not None:
                    # Replace the extending node with the new extended version
                    try:
                        idx = self.current_nodes.index(extending_node)
                        self.current_nodes[idx] = node
                    except ValueError:
                        # Extending node not in list anymore, just append
                        self.current_nodes.append(node)
                else:
                    # New context - append to list
                    self.current_nodes.append(node)

            # Build choice
            choice = CompletionChoice(
                finish_reason=finish_reason, index=i, text=completion_text
            )
            choices.append(choice)

        # Construct Completion response
        return Completion(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=self.server.config.model_name,
            object="text_completion",
            choices=choices,
        )

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of tracked sequences.

        Returns:
            For default mode (track_tree=False):
                Dictionary with 'nodes': List[SequenceNode] - ready for training
            For tree mode (track_tree=True):
                Dictionary with 'sequences': Dict[str, SequenceNode] and 'tree' alias
        """
        if self.track_tree:
            return {
                "sequences": self.sequences.copy(),
                "tree": self.sequences.copy(),  # Alias for compatibility
            }
        else:
            return {
                "nodes": self.current_nodes.copy(),  # Return a copy so reset() doesn't affect it
            }

    def reset(self):
        """Clear all tracked sequences."""
        if self.track_tree:
            self.sequences.clear()
        else:
            self.current_nodes.clear()

    async def get_logprobs(self, **kwargs) -> Dict[str, Any]:
        """
        Fetch prompt logprobs via wrapped server with a normalized schema.

        Supported inputs:
          - prompt
          - messages (converted to prompt)
          - input_ids

        Returns:
            Dict with:
              - prompt_tokens
              - prompt_topk_token_ids
              - prompt_topk_logprobs
        """
        request_kwargs = kwargs.copy()
        messages = request_kwargs.pop("messages", None)

        if messages is not None:
            prompt = self._convert_messages_to_prompt(messages)
            request_kwargs["prompt"] = prompt
        else:
            prompt = request_kwargs.get("prompt")

        if not hasattr(self.server, "get_logprobs"):
            raise NotImplementedError(
                f"{self.server.__class__.__name__} does not implement get_logprobs. "
                "Strict mode requires backend prompt logprobs."
            )

        payload = await self.server.get_logprobs(**request_kwargs)
        return payload


class DummyManagedServer:
    """
    A simple managed server wrapper for OpenAI endpoints that don't support token IDs/logprobs.

    Uses fixed placeholder values for tokens and logprobs. NOT suitable for training.
    """

    # Fixed dummy values
    DUMMY_TOKENS = [i for i in range(128)]
    DUMMY_MASKED_TOKENS = [-100] + DUMMY_TOKENS[1:]
    DUMMY_LOGPROBS = [-0.5 for _ in range(128)]

    def __init__(
        self,
        server: APIServer,
        tokenizer: Optional[Any] = None,
        track_tree: bool = False,
    ):
        self.server = server
        self.track_tree = track_tree
        # tokenizer is accepted but ignored - we don't tokenize anything

        if track_tree:
            self.sequences: Dict[str, SequenceNode] = {}
        else:
            self.current_nodes: List[SequenceNode] = []

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to simple text format."""
        return "\n\n".join([f"{m['role']}:{m['content']}" for m in messages])

    def _create_dummy_node(
        self,
        full_text: str,
        finish_reason: str = "stop",
    ) -> SequenceNode:
        """Create a sequence node with fixed dummy values."""
        return SequenceNode(
            full_text=full_text,
            tokens=self.DUMMY_TOKENS,
            masked_tokens=self.DUMMY_MASKED_TOKENS,
            logprobs=self.DUMMY_LOGPROBS,
            metadata={"finish_reason": finish_reason, "dummy_tokens": True},
        )

    async def chat_completion(self, **kwargs) -> ChatCompletion:
        """Make a chat completion call and track with dummy tokens."""
        messages = kwargs.get("messages", [])

        response = await self.server.chat_completion(**kwargs)

        for choice in response.choices:
            completion_content = choice.message.content or ""
            # Append assistant response to messages for full_text
            all_messages = messages + [
                {"role": "assistant", "content": completion_content}
            ]
            full_text = self._messages_to_text(all_messages)

            node = self._create_dummy_node(
                full_text=full_text,
                finish_reason=choice.finish_reason or "stop",
            )

            if self.track_tree:
                self.sequences[node.full_text] = node
            else:
                self.current_nodes.append(node)

        return response

    async def completion(self, **kwargs) -> Completion:
        """Make a completion call and track with dummy tokens."""
        prompt = kwargs.get("prompt", "")

        response = await self.server.completion(**kwargs)

        for choice in response.choices:
            completion_text = choice.text or ""
            full_text = f"{prompt}{completion_text}"

            node = self._create_dummy_node(
                full_text=full_text,
                finish_reason=choice.finish_reason or "stop",
            )

            if self.track_tree:
                self.sequences[node.full_text] = node
            else:
                self.current_nodes.append(node)

        return response

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of tracked sequences."""
        if self.track_tree:
            return {
                "sequences": self.sequences.copy(),
                "tree": self.sequences.copy(),
            }
        else:
            return {"nodes": self.current_nodes.copy()}

    def reset(self):
        """Clear all tracked sequences."""
        if self.track_tree:
            self.sequences.clear()
        else:
            self.current_nodes.clear()

    async def get_logprobs(self, **kwargs) -> Dict[str, Any]:
        """
        Dummy managed server does not provide real prompt logprobs.
        """
        raise NotImplementedError(
            "DummyManagedServer does not support get_logprobs in strict mode. "
            "Use a backend with real prompt logprob support."
        )


class ManagedServerAdapter:
    """
    Adapter that makes ManagedServer look like AsyncOpenAI for external libraries.

    Implements the subset of AsyncOpenAI interface commonly used:
    - client.chat.completions.create()
    - client.completions.create()
    - client.base_url

    This allows libraries like verifiers to use ManagedServer transparently
    while still getting automatic token and logprob tracking.
    """

    def __init__(self, managed_server: ManagedServer, base_url: str):
        """
        Initialize the adapter.

        Args:
            managed_server: The ManagedServer instance to wrap
            base_url: The base URL to expose (for compatibility checks)
        """
        self._managed = managed_server
        self.base_url = base_url
        self.chat = self._ChatNamespace(self._managed)
        self.completions = self._CompletionsNamespace(self._managed)

    class _ChatNamespace:
        def __init__(self, managed: ManagedServer):
            self._managed = managed
            self.completions = ManagedServerAdapter._ChatCompletionsNamespace(managed)

    class _ChatCompletionsNamespace:
        def __init__(self, managed: ManagedServer):
            self._managed = managed

        async def create(self, **kwargs):
            return await self._managed.chat_completion(**kwargs)

    class _CompletionsNamespace:
        def __init__(self, managed: ManagedServer):
            self._managed = managed

        async def create(self, **kwargs):
            return await self._managed.completion(**kwargs)

    async def post(self, path: str, body: dict, cast_to: type):
        """Not supported - raises NotImplementedError."""
        raise NotImplementedError(
            f"ManagedServerAdapter does not support post() for path '{path}'. "
            "This is used for vLLM interleaved rollouts. Use standard chat completions."
        )

    def copy(self, **kwargs):
        """Not supported - raises NotImplementedError."""
        raise NotImplementedError(
            "ManagedServerAdapter does not support copy(). "
            "This is used for vLLM tokenization endpoints."
        )
