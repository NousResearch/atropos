"""
Bidirectional translation between OpenAI tool_calls format and raw model text.

Uses vLLM's tool parsers directly — same parsing logic as vLLM's chat
completions endpoint. Supports 30+ model-specific parsers (hermes, llama,
mistral, deepseek, qwen3, etc.) via ToolParserManager.

Outbound (model → client): raw text with <tool_call> tags → structured OpenAI tool_calls
Inbound (client → model): OpenAI messages with tool roles → raw text for chat template
"""

import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# vLLM is optional — tool call parsing degrades gracefully without it
try:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.engine.protocol import (
        ExtractedToolCallInformation,
        FunctionCall,
        ToolCall,
    )
    from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    ChatCompletionRequest = None
    ExtractedToolCallInformation = None
    FunctionCall = None
    ToolCall = None
    ToolParser = None
    ToolParserManager = None


class ToolCallTranslator:
    """Bidirectional translation between OpenAI tool_calls and raw model text.

    Uses vLLM's tool parsers directly for outbound parsing (model output →
    OpenAI format). Maintains a lookup table mapping tool_call IDs back to
    the raw text that produced them, for reconstructing messages when the
    caller sends tool results back.

    The ManagedServer always stores raw text — this translator only
    transforms what goes over the HTTP wire.
    """

    def __init__(self, tokenizer: Any, parser_name: str = "hermes"):
        """
        Args:
            tokenizer: HuggingFace tokenizer instance.
            parser_name: Name of the vLLM tool parser to use.
                Available: hermes, llama3_json, llama4_json, mistral,
                deepseek_v3, qwen3_coder, granite, internlm, etc.
                See ToolParserManager.list_registered() for full list.

        Raises:
            Warning if vLLM is not installed — tool call parsing will be
            disabled but the translator can still handle message conversion
            and decoding.
        """
        self.tokenizer = tokenizer
        self.parser_name = parser_name
        self.parser = None

        if not VLLM_AVAILABLE:
            warnings.warn(
                "vLLM is not installed — tool call parsing is disabled. "
                "Install vllm to enable structured tool call extraction from "
                "model output (pip install vllm). The translator will still "
                "handle message conversion and template rendering, but "
                "parse_model_output() will return raw text without parsing.",
                stacklevel=2,
            )
        else:
            ParserClass = ToolParserManager.get_tool_parser(parser_name)
            self.parser = ParserClass(tokenizer)

        # tool_call_id → raw text segment that produced it.
        # Used to reconstruct assistant messages when the caller sends
        # follow-up messages with tool results.
        self.call_id_to_raw_text: Dict[str, str] = {}

    def parse_model_output(
        self,
        raw_text: str,
        tool_choice: Optional[str] = None,
        tools: Optional[List[dict]] = None,
    ) -> Tuple[Optional[str], Optional[List[ToolCall]], str]:
        """Parse raw model output into OpenAI response fields.

        Args:
            raw_text: Raw model output text (may contain <tool_call> tags etc.)
            tool_choice: The tool_choice from the request. If "none", skip
                parsing entirely. If "required", force tool_calls interpretation.
            tools: Tool definitions from the request (needed for vLLM request obj).

        Returns:
            Tuple of (content, tool_calls, finish_reason):
                content: Text content (before tool calls, or full text if no tools)
                tool_calls: List of ToolCall objects, or None
                finish_reason: "stop", "tool_calls", or "length"
        """
        # If tool_choice is "none" or no tools defined, don't even try parsing
        if tool_choice == "none" or not tools:
            return raw_text, None, "stop"

        # If vLLM isn't available, can't parse — return raw text
        if self.parser is None:
            return raw_text, None, "stop"

        # Build a minimal ChatCompletionRequest for the parser
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": ""}],
            model="proxy",
            tools=tools,
            tool_choice=tool_choice,
        )

        result: ExtractedToolCallInformation = self.parser.extract_tool_calls(
            raw_text, request
        )

        if result.tools_called and result.tool_calls:
            # Store mapping for reverse direction
            for tc in result.tool_calls:
                self.call_id_to_raw_text[tc.id] = raw_text

            return result.content, result.tool_calls, "tool_calls"
        else:
            return raw_text, None, "stop"

    def reconstruct_raw_text_from_tool_calls(self, tool_calls: List[dict]) -> str:
        """Reconstruct raw model text from OpenAI-format tool_calls.

        When a caller sends an assistant message with tool_calls (e.g. in a
        multi-turn conversation), we need to convert it back to the raw text
        the model actually generated so the chat template produces the right
        tokens.

        First tries the lookup table (exact reconstruction). Falls back to
        rebuilding from the structured data (best-effort).

        Args:
            tool_calls: List of tool call dicts from OpenAI format, each with
                'id', 'type', 'function' (containing 'name' and 'arguments').

        Returns:
            Raw text with <tool_call> tags (or whatever format the parser uses).
        """
        if not tool_calls:
            return ""

        # Try lookup table first — if the first call's ID is in the table,
        # we can return the exact raw text
        first_id = tool_calls[0].get("id", "")
        if first_id in self.call_id_to_raw_text:
            return self.call_id_to_raw_text[first_id]

        # Fallback: reconstruct from structured data
        # This is best-effort — the exact formatting may differ from what
        # the model originally generated, but it's close enough for the
        # chat template to handle.
        # TODO: make the tag format configurable per parser (hermes uses
        # <tool_call>, others may differ)
        parts = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "{}")
            # Parse arguments string back to dict for clean formatting
            try:
                args_dict = (
                    json.loads(arguments) if isinstance(arguments, str) else arguments
                )
            except (json.JSONDecodeError, TypeError):
                args_dict = arguments
            call_obj = {"name": name, "arguments": args_dict}
            parts.append(f"<tool_call>{json.dumps(call_obj)}</tool_call>")

        return "\n".join(parts)

    def convert_messages_for_template(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI messages to raw format suitable for apply_chat_template.

        Handles three cases:
        1. Regular messages (user, system): pass through unchanged
        2. Assistant messages with tool_calls: replace with raw text content
        3. Tool result messages (role=tool): pass through (chat template handles them)

        Args:
            messages: OpenAI-format messages list.

        Returns:
            Messages list with tool_call assistant messages reconstructed to raw text.
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "")

            if role == "assistant" and msg.get("tool_calls"):
                # Reconstruct raw text from tool_calls
                first_id = (
                    msg["tool_calls"][0].get("id", "") if msg["tool_calls"] else ""
                )
                is_exact = first_id in self.call_id_to_raw_text
                raw_text = self.reconstruct_raw_text_from_tool_calls(msg["tool_calls"])

                if not is_exact:
                    # Fallback reconstruction — prepend any content (e.g. <think>)
                    # that came before the tool calls
                    content = msg.get("content") or ""
                    if content:
                        raw_text = content + "\n" + raw_text
                # When exact lookup succeeds, raw_text already contains
                # the full model output (including any <think> blocks)

                converted.append(
                    {
                        "role": "assistant",
                        "content": raw_text,
                    }
                )
            else:
                # Pass through as-is (user, system, tool, regular assistant)
                converted.append(msg)

        return converted

    def decode_with_tool_awareness(
        self,
        token_ids: List[int],
        has_tools: bool = False,
    ) -> str:
        """Decode token IDs, preserving tool call tags when tools are active.

        Some tokenizers mark <tool_call> as a special token. If we decode with
        skip_special_tokens=True (the default), the tags vanish before the
        parser ever sees them. When tools are in play, we decode with
        skip_special_tokens=False to preserve them.

        Args:
            token_ids: Token IDs to decode.
            has_tools: Whether tools are active for this request.

        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=not has_tools,
        )
