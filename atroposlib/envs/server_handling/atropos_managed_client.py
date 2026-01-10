"""
AtroposManagedClient: AsyncOpenAI-compatible client backed by ManagedServer.

This module provides a drop-in replacement for AsyncOpenAI that uses Atropos's
ManagedServer for inference, enabling token tracking for multi-turn RL training
with the Verifiers library.

Usage:
    async with server_manager.managed_server(tokenizer=tokenizer) as managed:
        client = AtroposManagedClient(managed_server=managed, model="model-name")

        # Use like AsyncOpenAI - tokens are tracked automatically
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100
        )

        # Token data is available on the response:
        # - response.prompt_token_ids
        # - response.choices[0].token_ids
        # - response.choices[0].logprobs.content[i].logprob
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai.types.chat.chat_completion_message import ChatCompletionMessage

from atroposlib.envs.server_handling.managed_server import ManagedServer, SequenceNode

# =============================================================================
# Enhanced Types for Token Data Injection
# =============================================================================


@dataclass
class LogprobContent:
    """
    Single token logprob entry.

    Compatible with verifiers' parse_response_tokens() which accesses:
    - response.choices[i].logprobs.content[j].logprob
    """

    logprob: float
    token: str = ""
    token_id: int = 0
    top_logprobs: Optional[List[Any]] = None


@dataclass
class ChoiceLogprobs:
    """
    Logprobs structure compatible with verifiers expectations.

    Verifiers checks for either object or dict format:
    - Object: response.choices[i].logprobs.content[j].logprob
    - Dict: response.choices[i].logprobs["content"][j]["logprob"]

    This dataclass supports the object format.
    """

    content: List[LogprobContent] = field(default_factory=list)


@dataclass
class EnhancedChoice:
    """
    Choice with token_ids and logprobs for RL training.

    Adds the following attributes that verifiers expects:
    - token_ids: List[int] - completion token IDs
    - logprobs: ChoiceLogprobs - structured logprobs
    """

    index: int
    message: ChatCompletionMessage
    finish_reason: str
    token_ids: List[int]
    logprobs: ChoiceLogprobs


@dataclass
class EnhancedChatCompletion:
    """
    ChatCompletion with token data for RL training.

    Compatible with verifiers' parse_response_tokens() expectations:
    - prompt_token_ids: list[int]
    - choices[i].token_ids: list[int]
    - choices[i].logprobs.content[j].logprob
    """

    id: str
    created: int
    model: str
    object: str
    choices: List[EnhancedChoice]
    prompt_token_ids: List[int]
    usage: Optional[Dict[str, int]] = None


# =============================================================================
# AsyncOpenAI-Compatible Client Classes
# =============================================================================


class _CompletionsNamespace:
    """
    Mimics openai.resources.chat.completions.AsyncCompletions.

    Provides the create() method that verifiers calls.
    """

    def __init__(self, parent: "AtroposManagedClient"):
        self.parent = parent

    async def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        n: int = 1,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> EnhancedChatCompletion:
        """
        Create chat completion with token tracking.

        Returns ChatCompletion with additional attributes:
        - prompt_token_ids: list[int]
        - choices[i].token_ids: list[int]
        - choices[i].logprobs.content: list with logprob info

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (defaults to client's model)
            n: Number of completions (should be 1 for multi-turn)
            max_tokens: Max tokens in completion (legacy param)
            max_completion_tokens: Max tokens in completion (new param)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            tools: Tool definitions for function calling
            stop: Stop sequences
            **kwargs: Additional parameters passed to ManagedServer
        """
        # Use max_completion_tokens if provided, else max_tokens
        effective_max_tokens = max_completion_tokens or max_tokens

        # Build kwargs for ManagedServer
        completion_kwargs = {
            "messages": messages,
            "model": model or self.parent.model,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
        }

        if effective_max_tokens is not None:
            completion_kwargs["max_tokens"] = effective_max_tokens

        if tools is not None:
            completion_kwargs["tools"] = tools

        if stop is not None:
            completion_kwargs["stop"] = stop

        # Add any extra kwargs (like logprobs settings)
        for key, value in kwargs.items():
            if value is not None:
                completion_kwargs[key] = value

        # Call ManagedServer for inference
        completion = await self.parent.managed_server.chat_completion(
            **completion_kwargs
        )

        # Get token state from managed server
        state = self.parent.managed_server.get_state()
        nodes: List[SequenceNode] = state["nodes"]

        # Inject token data into response
        return self._enhance_completion(completion, nodes)

    def _enhance_completion(
        self, completion: Any, nodes: List[SequenceNode]
    ) -> EnhancedChatCompletion:
        """
        Convert ManagedServer output to verifiers-compatible format.

        Extracts token data from SequenceNodes and injects it into the
        ChatCompletion response in the format verifiers expects.
        """
        enhanced_choices = []
        prompt_token_ids: List[int] = []

        for i, (choice, node) in enumerate(zip(completion.choices, nodes)):
            # Find prompt/completion boundary from masked_tokens
            # -100 indicates prompt tokens, actual token IDs indicate completion
            prompt_len = sum(1 for m in node.masked_tokens if m == -100)

            # Extract prompt and completion portions
            if i == 0:
                prompt_token_ids = node.tokens[:prompt_len]

            completion_ids = node.tokens[prompt_len:]
            completion_logprobs = node.logprobs[prompt_len:]

            # Build logprobs structure verifiers expects
            logprobs_content = []
            tokenizer = self.parent.managed_server.tokenizer

            for token_id, logprob in zip(completion_ids, completion_logprobs):
                # Decode token to string if tokenizer available
                token_str = ""
                if tokenizer is not None:
                    try:
                        token_str = tokenizer.decode([token_id])
                    except Exception:
                        token_str = f"<token_{token_id}>"

                logprobs_content.append(
                    LogprobContent(
                        logprob=logprob,
                        token=token_str,
                        token_id=token_id,
                    )
                )

            # Create enhanced choice with token data
            enhanced_choice = EnhancedChoice(
                index=choice.index,
                message=choice.message,
                finish_reason=choice.finish_reason or "stop",
                token_ids=completion_ids,
                logprobs=ChoiceLogprobs(content=logprobs_content),
            )
            enhanced_choices.append(enhanced_choice)

        return EnhancedChatCompletion(
            id=completion.id,
            created=completion.created,
            model=completion.model,
            object=completion.object,
            choices=enhanced_choices,
            prompt_token_ids=prompt_token_ids,
            usage=completion.usage.model_dump() if completion.usage else None,
        )


class _ChatNamespace:
    """Mimics openai.resources.chat.AsyncChat."""

    def __init__(self, parent: "AtroposManagedClient"):
        self.completions = _CompletionsNamespace(parent)


class AtroposManagedClient:
    """
    AsyncOpenAI-compatible client backed by ManagedServer.

    This client provides the same interface as AsyncOpenAI but uses Atropos's
    ManagedServer for inference, enabling automatic token tracking for
    multi-turn RL training with the Verifiers library.

    The key feature is that responses include token data attributes that
    verifiers' parse_response_tokens() expects:
    - response.prompt_token_ids
    - response.choices[i].token_ids
    - response.choices[i].logprobs.content[j].logprob

    Usage:
        async with server_manager.managed_server(tokenizer=tokenizer) as managed:
            client = AtroposManagedClient(
                managed_server=managed,
                model="Qwen/Qwen2.5-1.5B-Instruct"
            )

            # Pass to verifiers env.rollout()
            state = await vf_env.rollout(
                input=rollout_input,
                client=client,
                model="Qwen/Qwen2.5-1.5B-Instruct",
            )

            # Token data is now in state["trajectory"][i]["tokens"]
    """

    def __init__(
        self,
        managed_server: ManagedServer,
        model: str,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the managed client.

        Args:
            managed_server: ManagedServer instance for inference and token tracking
            model: Model name to use for completions
            base_url: Optional base URL (for API compatibility, not used)
        """
        self.managed_server = managed_server
        self.model = model
        self.base_url = base_url or "http://managed-server"

        # Mimic AsyncOpenAI namespace structure
        self.chat = _ChatNamespace(self)

    def reset(self):
        """Reset token tracking state between rollouts."""
        self.managed_server.reset()

    async def close(self):
        """Compatibility method - no-op since ManagedServer handles cleanup."""
        pass

    def copy(self, **_kwargs) -> "AtroposManagedClient":
        """
        Create a copy of this client (for API compatibility).

        Verifiers may call client.copy() for certain operations.
        Returns self since we want to maintain the same ManagedServer state.
        """
        return self
