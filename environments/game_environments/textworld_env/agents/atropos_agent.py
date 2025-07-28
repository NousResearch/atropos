import logging
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizer

from atroposlib.envs.base import ServerManager
from atroposlib.type_definitions import Message

from ..utils.memory_parser import extract_memory_block, validate_memory_content
from .agent_types import (
    AtroposAgentAction,
    AtroposAgentActionLog,
    AtroposAgentTurn,
)
from .atropos_memory_manager import (
    MEMORY_SYSTEM_PREREQUISITES_AVAILABLE,
)
from .atropos_memory_manager import (
    AtroposMemoryManager as TextWorldMemoryManager,
)

logger = logging.getLogger(__name__)


class AtroposAgentConfig(BaseModel):
    """Configuration for AtroposAgent."""

    # General LLM parameters
    model_id: str = Field(
        default="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        description="LLM model ID for server calls.",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature for the LLM."
    )
    max_tokens_per_completion: int = Field(
        default=16384,
        description="Maximum tokens to generate for each action alternative.",
    )
    max_retries_on_error: int = Field(
        default=3, description="Maximum retries for LLM calls on failure."
    )

    system_prompt: str = Field(
        default=(
            "You are a helpful AI assistant playing a text adventure game. "
            "Think step-by-step and then call the required tool."
        ),
        description="A general system prompt for the agent.",
    )

    # Context Management
    max_history_tokens: int = Field(
        default=20000,
        description="Maximum tokens to keep in conversation history (sliding window).",
    )

    # Memory System Configuration (related to how agent *uses* memory)
    enable_memory: bool = Field(
        default=True,
        description="Whether the agent should use its memory system. "
        "Actual memory system functionality depends on TextWorldMemoryManager.",
    )
    embedding_dim: int = Field(
        default=384,
        description=(
            "Default dimension of embeddings for TextWorldMemoryManager. "
            "Will be overridden by actual model."
        ),
    )
    memory_top_k: int = Field(
        default=3,
        description="Number of relevant memories to retrieve and prepend to the observation.",
    )
    memory_generation_system_prompt: str = Field(
        default=(
            "You are a helpful assistant. Based on the provided game history snippet, "
            "summarize the key events, insights, player actions, and outcomes concisely. "
            "This summary will be used as a memory for a game-playing agent. "
            "Focus on information that would be useful for making future decisions in the game. "
            "Output only the summary. Be brief and focus on the most important facts."
        ),
        description="System prompt for the LLM call to generate a memory summary.",
    )
    max_tokens_for_memory_summary: int = Field(
        default=150, description="Maximum tokens to generate for a memory summary."
    )

    player_id_for_logging: str = Field(
        default="AtroposAgent", description="Identifier for the agent in logs."
    )

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


def _convert_messages_for_api(messages: List[Message]) -> List[Dict[str, str]]:
    """Converts a list of Atropos Message objects to API-compatible dicts."""
    api_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content_str = msg.get("content", "")
        if content_str is None:
            content_str = ""
        api_messages.append({"role": role, "content": content_str})
    return api_messages


class AtroposAgent:
    """
    An agent that interacts with an LLM for game playing, managing its own dialogue history
    and using a memory system.
    """

    # Define tools for tokenizers that require them (e.g., Qwen)
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "Execute a command in the game",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute",
                        },
                        "expected_outcome": {
                            "type": "string",
                            "description": "Expected result of the command",
                        },
                    },
                    "required": ["command", "expected_outcome"],
                },
            },
        }
    ]

    def __init__(
        self,
        server_client: ServerManager,
        tokenizer: PreTrainedTokenizer,
        config: Optional[AtroposAgentConfig] = None,
        memory_manager: Optional[TextWorldMemoryManager] = None,
    ):
        self.config = config if config is not None else AtroposAgentConfig()
        self.model_id = self.config.model_id

        self.server_client = server_client
        self.tokenizer = tokenizer
        self.system_prompt_content = self.config.system_prompt

        self.game_log: AtroposAgentActionLog = AtroposAgentActionLog(turn=[])

        if memory_manager:
            self.memory_manager = memory_manager
        elif self.config.enable_memory and MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            logger.info(
                f"AtroposAgent[{self.config.player_id_for_logging}] "
                "Initializing default TextWorldMemoryManager.",
            )
            self.memory_manager = TextWorldMemoryManager(
                embedding_dim_config_val=self.config.embedding_dim,
                player_id_for_logging=f"{self.config.player_id_for_logging}_Memory",
            )
        else:
            self.memory_manager = None
            if self.config.enable_memory and not MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
                logger.warning(
                    f"AtroposAgent[{self.config.player_id_for_logging}] "
                    "Memory is enabled in config, but TextWorldMemoryManager "
                    "could not be initialized due to missing prerequisites. "
                    "Memory system will be OFF.",
                )
            else:
                logger.info(
                    f"AtroposAgent[{self.config.player_id_for_logging}] "
                    "Memory system is disabled or no manager provided.",
                )

        self.memory_generation_system_prompt = (
            self.config.memory_generation_system_prompt
        )

        self.last_input_token_count: Optional[int] = None
        self.last_output_token_count: Optional[int] = None

    def new_game(self) -> None:
        """Clears all game-specific logs, resets turn number, and resets memory."""
        self.game_log = AtroposAgentActionLog(turn=[])
        logger.info(
            f"AtroposAgent[{self.config.player_id_for_logging}] "
            "New game started. All logs cleared, turn number reset.",
        )

        if self.memory_manager and self.config.enable_memory:
            self.memory_manager.reset_memory()
            logger.info(
                f"AtroposAgent[{self.config.player_id_for_logging}] "
                "Memory manager reset for new game.",
            )
        elif self.config.enable_memory:
            logger.warning(
                f"AtroposAgent[{self.config.player_id_for_logging}] "
                "New game: Memory enabled in config, but no active memory "
                "manager to reset.",
            )

    def get_final_canonical_dialogue(self) -> List[Message]:
        """Reconstructs and returns the complete canonical
        dialogue history for the finished game."""
        return self._reconstruct_canonical_history()

    async def generate(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[str]:
        api_kwargs = {
            "model": self.config.model_id,
            "n": kwargs.get("n", 1),
            "max_tokens": kwargs.get(
                "max_tokens", self.config.max_tokens_per_completion
            ),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stop": stop_sequences,
        }
        if api_kwargs["stop"] is None:
            del api_kwargs["stop"]

        log_prompt_snippet = (
            messages[-1]["content"][:200]
            if messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("content")
            else "EMPTY_HISTORY_CONTENT"
        )
        logger.debug(
            f"AtroposAgent[{self.config.player_id_for_logging}] (generate) "
            f"LLM Chat Prompt (last message content, first 200 chars): {log_prompt_snippet}"
        )

        try:
            # Apply chat template with tools
            prompt = self.tokenizer.apply_chat_template(
                messages, tools=self.TOOLS, add_generation_prompt=True, tokenize=False
            )

            # Update api_kwargs for completion endpoint
            api_kwargs.pop("model", None)  # model is not used in completion endpoint
            raw_response = await self.server_client.completion(
                prompt=prompt, **api_kwargs
            )

            content = None

            if raw_response and raw_response.choices:
                first_choice = raw_response.choices[0]
                if hasattr(first_choice, "text") and first_choice.text is not None:
                    content = first_choice.text.strip()
                else:
                    logger.warning(
                        f"AtroposAgent[{self.config.player_id_for_logging}] (generate) "
                        f"LLM first choice had no text content. Choice: {first_choice}"
                    )
            else:
                logger.warning(
                    f"AtroposAgent[{self.config.player_id_for_logging}] (generate) "
                    f"chat_completion returned no choices or empty response. "
                    f"History snippet: {log_prompt_snippet}"
                )

            if hasattr(raw_response, "usage") and raw_response.usage:
                self.last_input_token_count = raw_response.usage.prompt_tokens
                self.last_output_token_count = raw_response.usage.completion_tokens
                logger.debug(
                    f"AtroposAgent[{self.config.player_id_for_logging}] (generate) "
                    f"Token usage: input={self.last_input_token_count}, "
                    f"output={self.last_output_token_count}"
                )
            else:
                self.last_input_token_count = None
                self.last_output_token_count = None

            return content

        except Exception as e:
            logger.error(
                f"AtroposAgent[{self.config.player_id_for_logging}] (generate) "
                f"LLM API (chat_completion) error: {e}. "
                f"History snippet: {log_prompt_snippet}",
                exc_info=True,
            )
            raise ValueError(f"LLM API error during generate: {e}")

    async def generate_action(
        self,
        observation_content: str,
        n: int = 1,
    ) -> List[AtroposAgentAction]:
        current_history_atropos: List[Message] = self._reconstruct_canonical_history()

        retrieved_memories_str = ""
        if (
            self.config.enable_memory
            and self.memory_manager
            and self.memory_manager.is_active
        ):
            relevant_memories = await self.memory_manager.retrieve_relevant_memories(
                query_text=observation_content, k=self.config.memory_top_k
            )
            if relevant_memories:
                retrieved_memories_str = "\n".join(
                    [f"- {mem}" for mem in relevant_memories]
                )
                logger.info(
                    f"AtroposAgent[{self.config.player_id_for_logging}] "
                    f"Retrieved {len(relevant_memories)} memories for current observation."
                )
                observation_content = (
                    f"Relevant Memories:\n{retrieved_memories_str}\n\n"
                    f"Original Observation:\n{observation_content}"
                )
            else:
                logger.info(
                    f"AtroposAgent[{self.config.player_id_for_logging}] "
                    "No relevant memories found for current observation."
                )

        observation_message = Message(
            role="user", content=observation_content, reward=None
        )
        current_history_atropos.append(observation_message)

        history_for_llm_call = _convert_messages_for_api(current_history_atropos)

        log_prompt_snippet = (
            history_for_llm_call[-1]["content"][:200]
            if history_for_llm_call
            and isinstance(history_for_llm_call[-1], dict)
            and history_for_llm_call[-1].get("content")
            else "EMPTY_HISTORY_CONTENT"
        )
        logger.debug(
            f"AtroposAgent[{self.config.player_id_for_logging}] (generate_action) "
            f"LLM Chat Prompt (last message content, first 200 chars): {log_prompt_snippet}"
        )

        llm_generated_actions: List[AtroposAgentAction] = []

        try:
            # Apply chat template with tools
            prompt = self.tokenizer.apply_chat_template(
                history_for_llm_call,
                tools=self.TOOLS,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Count tokens and warn if approaching limits
            try:
                prompt_tokens = len(self.tokenizer.encode(prompt))
                total_expected_tokens = (
                    prompt_tokens + self.config.max_tokens_per_completion
                )

                logger.info(
                    f"Token usage: {prompt_tokens} input tokens "
                    f"+ {self.config.max_tokens_per_completion} max completion "
                    f"= {total_expected_tokens} total"
                )

                if total_expected_tokens > 40960:  # Model's context limit
                    logger.error(
                        f"Total tokens ({total_expected_tokens}) exceeds model "
                        f"context limit (40,960)! Consider reducing "
                        f"max_history_tokens (currently {self.config.max_history_tokens})"
                    )
                elif prompt_tokens > self.config.max_history_tokens:
                    logger.warning(
                        f"Input tokens ({prompt_tokens}) exceeds "
                        f"max_history_tokens ({self.config.max_history_tokens}). "
                        "Sliding window may not be working correctly."
                    )
                elif prompt_tokens > self.config.max_history_tokens * 0.9:
                    logger.warning(
                        f"Input tokens ({prompt_tokens}) approaching "
                        f"max_history_tokens limit ({self.config.max_history_tokens}). "
                        "Context window will start dropping old messages soon."
                    )
            except Exception as e:
                logger.warning(f"Failed to count prompt tokens: {e}")

            completions = await self.server_client.completion(
                prompt=prompt,
                n=n,
                max_tokens=self.config.max_tokens_per_completion,
                temperature=self.config.temperature,
                top_p=0.9,
                stop=["</tool_call>", "<|im_end|>", "<|endoftext|>"],
            )

            if completions and completions.choices:
                for choice_idx, choice in enumerate(completions.choices):
                    if hasattr(choice, "text") and choice.text is not None:
                        # Append </tool_call> if there's an unclosed tool_call block
                        action_text = choice.text.strip()
                        tool_call_open_count = action_text.count("<tool_call>")
                        tool_call_close_count = action_text.count("</tool_call>")

                        # Only append closing tag if we have more opens than closes
                        if tool_call_open_count > tool_call_close_count:
                            action_text += "</tool_call>"

                        llm_generated_actions.append(
                            AtroposAgentAction(
                                action_text=action_text,
                                api_error=False,
                                score=0.0,
                                logprobs=None,
                            )
                        )
                    else:
                        logger.warning(
                            f"AtroposAgent[{self.config.player_id_for_logging}] "
                            f"(generate_action) LLM choice {choice_idx} had no text content. "
                            f"Choice: {choice}"
                        )
                        llm_generated_actions.append(
                            AtroposAgentAction(
                                action_text="",
                                api_error=True,
                                score=0.0,
                                logprobs=None,
                            )
                        )
                if hasattr(completions, "usage") and completions.usage:
                    logger.debug(
                        f"AtroposAgent[{self.config.player_id_for_logging}] "
                        f"(generate_action) Token usage (n={n}): "
                        f"input={completions.usage.prompt_tokens}, "
                        f"output={completions.usage.completion_tokens}"
                    )

            else:
                logger.warning(
                    f"AtroposAgent[{self.config.player_id_for_logging}] "
                    f"(generate_action) completion returned no choices or empty response. "
                    f"History snippet: {log_prompt_snippet}"
                )

        except Exception as e:
            logger.error(
                f"AtroposAgent[{self.config.player_id_for_logging}] "
                f"(generate_action) LLM API (chat_completion) error: {e}. "
                f"History snippet: {log_prompt_snippet}",
                exc_info=True,
            )
            for _ in range(n):
                llm_generated_actions.append(
                    AtroposAgentAction(
                        action_text="<ERROR_GENERATING_ACTION>",
                        api_error=True,
                        score=-1.0,
                        logprobs=None,
                    )
                )

        while len(llm_generated_actions) < n:
            logger.warning(
                f"AtroposAgent[{self.config.player_id_for_logging}] "
                f"(generate_action) LLM returned fewer actions than requested "
                f"({len(llm_generated_actions)} vs {n}). Appending error placeholders."
            )
            llm_generated_actions.append(
                AtroposAgentAction(
                    action_text="<MISSING_ACTION_FROM_LLM>",
                    api_error=True,
                    score=-1.0,
                    logprobs=None,
                )
            )

        new_turn = AtroposAgentTurn(
            turn_number=len(self.game_log["turn"]) + 1,
            observation_message=observation_message,
            alternatives=llm_generated_actions,
            selected_alternative=None,
        )
        self.game_log["turn"].append(new_turn)

        return llm_generated_actions[:n]

    async def record_selected_action_and_learn_from_turn(
        self, selected_action_index: int
    ) -> None:
        if not self.game_log["turn"]:
            logger.error(
                f"AtroposAgent[{self.config.player_id_for_logging}] "
                "No turns to select from. Cannot record or learn."
            )
            return

        last_turn_data = self.game_log["turn"][-1]
        if not (0 <= selected_action_index < len(last_turn_data["alternatives"])):
            logger.error(
                f"AtroposAgent[{self.config.player_id_for_logging}] "
                f"Invalid selected_action_index {selected_action_index} "
                f"for {len(last_turn_data['alternatives'])} alternatives. "
                "Cannot record."
            )
            return

        last_turn_data["selected_alternative"] = selected_action_index
        logger.info(
            f"AtroposAgent[{self.config.player_id_for_logging}] "
            f"Selected action {selected_action_index} for turn "
            f"{last_turn_data['turn_number']}."
        )

        if (
            self.config.enable_memory
            and self.memory_manager
            and self.memory_manager.is_active
        ):
            selected_action_obj = last_turn_data["alternatives"][selected_action_index]

            if (
                not selected_action_obj["api_error"]
                and selected_action_obj["action_text"]
            ):
                # Extract memory from the agent's response
                response_text = selected_action_obj["action_text"]
                memory_content = extract_memory_block(response_text)

                if memory_content and validate_memory_content(memory_content):
                    logger.info(
                        f"AtroposAgent[{self.config.player_id_for_logging}] "
                        f"Extracted inline memory for turn {last_turn_data['turn_number']}: "
                        f"{memory_content[:50]}..."
                    )
                    await self.memory_manager.add_memory(memory_content)

        elif self.config.enable_memory:
            logger.debug(
                f"AtroposAgent[{self.config.player_id_for_logging}] "
                f"Memory enabled but manager not active. Skipping memory storage for turn "
                f"{last_turn_data['turn_number']}."
            )

    def _count_tokens_in_messages(self, messages: List[Message]) -> int:
        """Count tokens in a list of messages by converting to chat format."""
        try:
            # Convert messages to the format expected by apply_chat_template
            formatted_messages = _convert_messages_for_api(messages)

            # Apply chat template to get the full formatted text
            prompt = self.tokenizer.apply_chat_template(
                formatted_messages,
                tools=self.TOOLS,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Count tokens in the prompt
            tokens = self.tokenizer.encode(prompt)
            return len(tokens)
        except Exception as e:
            # Testing fallback re tokenizer issues, qwen template SHOULD be fixed now
            logger.warning(
                f"Failed to count tokens accurately: {e}. " "Using rough estimate."
            )
            # Rough estimate: 4 characters per token
            char_count = sum(len(msg.content or "") for msg in messages)
            return char_count // 4

    def _reconstruct_canonical_history(self) -> List[Message]:
        """Reconstruct conversation history with sliding window to manage context length."""
        history: List[Message] = []

        # Always include system prompt
        if self.system_prompt_content:
            history.append(
                Message(role="system", content=self.system_prompt_content, reward=None)
            )

        # Get all turns
        all_turns = []
        for turn_data in self.game_log["turn"]:
            turn_messages = [turn_data["observation_message"]]
            if turn_data["selected_alternative"] is not None:
                selected_idx = turn_data["selected_alternative"]
                if 0 <= selected_idx < len(turn_data["alternatives"]):
                    choice = turn_data["alternatives"][selected_idx]
                    if not choice["api_error"] and choice["action_text"]:
                        action_text = choice["action_text"]
                        # Strip thinking blocks from assistant messages to save tokens
                        # Memory blocks left in, idea is LLM learns to use them correctly
                        action_text_stripped = re.sub(
                            r"<think>.*?</think>",
                            "",
                            action_text,
                            flags=re.DOTALL,
                        ).strip()
                        turn_messages.append(
                            Message(
                                role="assistant",
                                content=action_text_stripped,
                                reward=None,
                            )
                        )
                else:
                    logger.error(
                        f"AtroposAgent[{self.config.player_id_for_logging}] "
                        f"Invalid selected_alternative index {selected_idx} "
                        f"in turn {turn_data['turn_number']} during history reconstruction."
                    )
            all_turns.append(turn_messages)

        # Apply sliding window: keep most recent turns that fit within token budget
        if all_turns:
            # Start with just system prompt
            current_tokens = self._count_tokens_in_messages(history)
            logger.debug(f"System prompt uses {current_tokens} tokens")
            included_turn_indices = []

            # Check turns from most recent to oldest
            for i in range(len(all_turns) - 1, -1, -1):
                turn_messages = all_turns[i]
                test_history = history.copy()
                for j in sorted(included_turn_indices + [i]):
                    test_history.extend(all_turns[j])

                test_tokens = self._count_tokens_in_messages(test_history)

                if test_tokens > self.config.max_history_tokens:
                    # We've hit the limit! Break the loop
                    break

                # This turn fits, include it in the history
                included_turn_indices.append(i)
                current_tokens = test_tokens

            # add included turns in chronological order
            for i in sorted(included_turn_indices):
                history.extend(all_turns[i])

        return history

    def get_last_token_usage(self) -> Tuple[Optional[int], Optional[int]]:
        """Returns the last recorded input and output token counts."""
        return self.last_input_token_count, self.last_output_token_count
