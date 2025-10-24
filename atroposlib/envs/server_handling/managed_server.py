"""
Managed server wrapper that tracks text sequences with aligned tokens and logprobs.

This wrapper maintains a tree structure of sequences, where:
- Each node represents a complete text sequence (prompt + completion)
- Tokens and logprobs are tracked with proper masking for training
- Branching occurs organically from different contexts and n > 1 completions
"""

import time
import uuid
import warnings
from typing import Any, Dict, List, Optional, Union

from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import Completion, CompletionChoice
from pydantic import BaseModel

from atroposlib.envs.server_handling.server_baseline import APIServer


class SequenceNode(BaseModel):
    """
    A node in the sequence tree representing a complete text sequence.

    Attributes:
        full_text: Complete text (prompt + completion)
        tokens: Full token sequence (actual token IDs)
        masked_tokens: Tokens with -100 for prompt positions, actual IDs for completion
        logprobs: Logprobs with 0.0 for prompt positions, actual values for completion
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
    logprobs set to 0.0).

    Uses the clean _tokens_and_logprobs_completion_wrapper interface internally.
    """

    def __init__(
        self,
        server: APIServer,
        tokenizer: Optional[Any] = None,
        track_tree: bool = False,
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
        """
        self.server = server
        self.tokenizer = tokenizer
        self.track_tree = track_tree

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

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to prompt text using tokenizer's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        if self.tokenizer is None:
            # Fallback: simple concatenation
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        if hasattr(self.tokenizer, "apply_chat_template"):
            # Only add generation prompt if last message is not from assistant
            add_generation_prompt = (
                len(messages) == 0 or messages[-1].get("role") != "assistant"
            )

            # Use the tokenizer's chat template
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        else:
            # Fallback for tokenizers without chat template
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

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
    ) -> SequenceNode:
        """
        Create a sequence node with proper masking.

        Args:
            input_text: The input prompt text
            parent_node: Parent node to extend from (if available)
            prompt_tokens: Token IDs for the prompt
            output_tokens: Token IDs for the output/completion
            output_logprobs: Logprobs for output tokens
            completion_text: The completion text
            finish_reason: Finish reason from server

        Returns:
            SequenceNode with properly masked tokens and logprobs
        """
        # Combine text
        full_text = input_text + completion_text

        # If we have a parent node, we should use its tokens as the prompt base
        if parent_node is not None:
            # Use parent's full tokens as the prompt
            prompt_tokens = parent_node.tokens.copy()

        # Combine tokens
        full_tokens = prompt_tokens + output_tokens
        prompt_len = len(prompt_tokens)

        # Create masked tokens: -100 for prompt, actual IDs for completion
        masked_tokens = [-100] * prompt_len + output_tokens

        # Create masked logprobs: 0.0 for prompt, actual for completion
        # Pad logprobs to match token length if needed
        if len(output_logprobs) < len(output_tokens):
            output_logprobs = output_logprobs + [0.0] * (
                len(output_tokens) - len(output_logprobs)
            )
        elif len(output_logprobs) > len(output_tokens):
            output_logprobs = output_logprobs[: len(output_tokens)]

        full_logprobs = [0.0] * prompt_len + output_logprobs

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

        Args:
            **kwargs: Standard chat completion kwargs (messages, n, etc.)

        Returns:
            ChatCompletion response
        """
        # Get input text
        messages = kwargs.get("messages", [])
        prompt = self._convert_messages_to_prompt(messages)

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
        ) = await self.server._tokens_and_logprobs_completion_wrapper(
            **completion_kwargs
        )

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
            choice = Choice(
                finish_reason=finish_reason,
                index=i,
                message=ChatCompletionMessage(
                    content=completion_text, role="assistant"
                ),
            )
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
        ) = await self.server._tokens_and_logprobs_completion_wrapper(**kwargs)

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
