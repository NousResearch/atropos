import logging
from typing import List, Optional, Tuple, Any
from atroposlib.type_definitions import Message

logger = logging.getLogger(__name__)

class AtroposAgent:
    """
    An agent that interacts with an LLM for game playing, managing its own dialogue history.
    It uses a chat-based completion model currently (for compatibility with existing LLMs and servers).
    """
    def __init__(
        self,
        system_prompt: str,
        tokenizer: Any, # Tokenizer instance (e.g., from HuggingFace) for token counting
        temperature: float,
        max_context_token_length: int, # Overall maximum token length for the context window
        max_tokens_for_llm_output: int = 256, # Max tokens the LLM should generate for its response
        player_id_for_logging: Optional[int] = None # Optional player ID for more specific logging
    ):
        self.system_prompt_content = system_prompt
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_context_token_length = max_context_token_length
        self.max_tokens_for_llm_output = max_tokens_for_llm_output
        self.player_id_for_logging = str(player_id_for_logging) if player_id_for_logging is not None else "Agent"

        self.current_game_messages: List[Message] = []

    def start_new_game_dialogue(self) -> None:
        """Clears the current dialogue history and starts with the system prompt."""
        self.current_game_messages = [
            {"role": "system", "content": self.system_prompt_content}
        ]

    def get_final_game_dialogue(self) -> List[Message]:
        """Returns the complete dialogue history for the finished game."""
        return list(self.current_game_messages) # Return a copy

    async def _agent_sample_llm_response(
        self, 
        history_for_llm_call: List[Message], 
        server_client: Any
    ) -> Tuple[Optional[str], bool]:
        """
        Internal method to sample a response from the LLM using chat completions.
        Returns a tuple: (llm_content_or_empty_tool_call, api_error_occurred_flag)
        """
        log_prompt_snippet = history_for_llm_call[-1]['content'][:200] if history_for_llm_call else "EMPTY_HISTORY"
        logger.debug(f"AtroposAgent[{self.player_id_for_logging}] LLM Chat Prompt (last message content, first 200 chars): {log_prompt_snippet}")

        try:
            chat_completions = await server_client.chat_completion(
                messages=history_for_llm_call,
                n=1,
                max_tokens=self.max_tokens_for_llm_output,
                temperature=self.temperature
            )
            
            llm_generated_content = ""
            if chat_completions.choices and \
               chat_completions.choices[0].message and \
               hasattr(chat_completions.choices[0].message, 'content') and \
               chat_completions.choices[0].message.content is not None:
                llm_generated_content = chat_completions.choices[0].message.content.strip()
            
            if not llm_generated_content:
                logger.warning(f"AtroposAgent[{self.player_id_for_logging}] returned empty or None content from chat_completion. Last user message (start): {log_prompt_snippet}")
                return "<tool_call>\n\n</tool_call>", False # Expected to be parsed as an error by the environment
            
            logger.debug(f"AtroposAgent[{self.player_id_for_logging}] raw content output via chat_completion: '{llm_generated_content}'")
            return llm_generated_content, False # No API error

        except Exception as e:
            logger.error(f"AtroposAgent[{self.player_id_for_logging}] LLM API (chat_completion) error: {e}. Last user message (start): {log_prompt_snippet}")
            return None, True # API error occurred

    async def _generate_memory(
        self,
        observation_content: str, # The game-specific augmented observation from the environment
        server_client: Any,
    ) -> Tuple[Optional[str], bool, bool]:
        """
        Generates a memory based on the observation, managing history and LLM interaction.
        """
        pass

    async def generate_action(
        self,
        observation_content: str, # The game-specific augmented observation from the environment
        server_client: Any,
        # Context for logging from the environment
        game_seed_for_logging: Optional[int] = None,
        group_idx_for_logging: Optional[int] = None,
        turn_idx_for_logging: Optional[int] = None,
        is_evaluation_context: bool = False,
        eval_episode_idx_for_logging: Optional[int] = None,

    ) -> Tuple[Optional[str], bool, bool]:
        """
        Generates an action based on the observation, managing history and LLM interaction.

        Returns:
            A tuple: (raw_llm_response_str, api_error_occurred, token_limit_exceeded)
        """
        log_prefix_parts = []
        if is_evaluation_context:
            log_prefix_parts.append(f"Eval Ep {eval_episode_idx_for_logging or 'N/A'}")
        else:
            log_prefix_parts.append(f"Train Seed {game_seed_for_logging or 'N/A'}")
            if group_idx_for_logging is not None: log_prefix_parts.append(f"Grp {group_idx_for_logging}")
        log_prefix_parts.append(f"Agent {self.player_id_for_logging}")
        if turn_idx_for_logging is not None: log_prefix_parts.append(f"Turn {turn_idx_for_logging}")
        log_prefix_base = f"[{', '.join(log_prefix_parts)}]"

        # Append the environment's observation as a user message
        user_message: Message = {"role": "user", "content": observation_content}
        self.current_game_messages.append(user_message)

        # Token length check before calling LLM
        # The history for token check includes the latest user message.
        # The tokenizer.apply_chat_template is used here only for an accurate token count
        # of how the messages would be formatted by the LLM, including generation prompts.
        prompt_str_for_token_check = self.tokenizer.apply_chat_template(
            self.current_game_messages, # Current history up to and including the latest user message
            tokenize=False,
            add_generation_prompt=True # Crucial for an accurate count of input tokens to LLM
        )
        current_prompt_token_count = len(self.tokenizer.encode(prompt_str_for_token_check))
        
        safety_buffer_tokens = 64 # Safety margin

        if current_prompt_token_count + self.max_tokens_for_llm_output + safety_buffer_tokens > self.max_context_token_length:
            logger.warning(
                f"{log_prefix_base} Token count for prompt+output "
                f"({current_prompt_token_count + self.max_tokens_for_llm_output}) "
                f"approaching max context token length ({self.max_context_token_length}). Marking token limit exceeded."
            )
            # The user message was already added. The environment will handle this state.
            return None, False, True # parsed_action=None, api_error=False, token_limit_exceeded=True

        # LLM sees the full history up to this point (system, previous turns, current user message)
        raw_llm_response_str, api_error = await self._agent_sample_llm_response(
            history_for_llm_call=self.current_game_messages, 
            server_client=server_client
        )

        if api_error:
            # Error already logged by _agent_sample_llm_response
            # Append a placeholder for the assistant message in case of API error, so history is balanced
            self.current_game_messages.append({"role": "assistant", "content": "<AGENT_LLM_API_ERROR>"})
            return None, True, False

        if raw_llm_response_str is None: # Should be handled by API error, but as a safeguard
             logger.error(f"{log_prefix_base} _agent_sample_llm_response returned None without API error flag. Treating as API error.")
             self.current_game_messages.append({"role": "assistant", "content": "<AGENT_LLM_UNEXPECTED_NONE_RESPONSE>"})
             return None, True, False # Treat as API error

        # Append LLM's successful response as an assistant message
        self.current_game_messages.append({"role": "assistant", "content": raw_llm_response_str})
        
        return raw_llm_response_str, False, False # api_error=False, token_limit_exceeded=False 