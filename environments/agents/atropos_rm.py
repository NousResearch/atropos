import logging
import re
from typing import List, Optional, Tuple, Any, Dict
import json
from transformers import PreTrainedTokenizer
from pydantic import BaseModel, Field

# A common representation for chat messages
Message = Dict[str, str]

logger = logging.getLogger(__name__)

# --- Start of AtroposRMConfig Definition ---
class AtroposRMConfig(BaseModel):
    """Configuration for AtroposRM."""
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0, 
        description="Sampling temperature for the RM LLM."
    )
    max_context_token_length: int = Field(
        default=4096, # Example, should align with the model used for RM
        description="Maximum context token length the RM LLM should consider."
    )
    max_tokens_for_llm_output: int = Field(
        default=512, # RM might need more tokens for thinking + Q-value
        description="Maximum tokens the RM LLM should generate for one judgement."
    )
    thinking: bool = Field(
        default=True,
        description="If True, RM is instructed to use <thinking> tags. If False, only Q-value."
    )
    q_value_extraction_pattern: str = Field(
        default=r"\\boxed{\\s*([+-]?\\d*\\.?\\d+)\\s*}",
        description="Regex pattern to extract the Q-value from the RM's response. Must capture the number."
    )
    thinking_block_extraction_pattern: str = Field(
        default=r"<thinking>(.*?)</thinking>",
        description="Regex pattern to extract the thinking block. Must capture content within tags."
    )
    rm_id_for_logging: str = Field(
        default="RM", 
        description="Identifier for the RM in logs."
    )

    class Config:
        extra = 'forbid'
# --- End of AtroposRMConfig Definition ---

class AtroposRM:
    """
    Atropos Reward Model (RM).
    This LLM-based agent evaluates a policy agent\'s proposed action (which includes thinking)
    from a given game state. It generates G distinct "judgements", where each judgement consists
    of its own thinking process and a numerical Q-value for the policy's action.
    The RM is designed to be trainable via GRPO, using the final game outcome to score its judgements.
    """

    def __init__(
        self,
        server_client: Any, # LLM server client (e.g., APIServer instance)
        tokenizer: PreTrainedTokenizer,  # Tokenizer instance (e.g., from HuggingFace)
        config: Optional[AtroposRMConfig] = None,
    ):
        """
        Initializes the AtroposRM.

        Args:
            server_client: The API client for the LLM server.
            tokenizer: The tokenizer used for token counting if necessary (e.g., by a
                       client library or for context length checks, though context checks
                       are not explicitly implemented in this version of _sample_one_judgement).
            config: The configuration for the RM.
        """
        self.config = config if config is not None else AtroposRMConfig()
        self.server_client = server_client
        self.tokenizer = tokenizer
        
        # Construct the system prompt based on config.thinking
        base_prompt = (
            "You are an expert reward model. Your task is to analyze a given trajectory of an "
            "AI agent interacting with an environment and provide a numerical Q-value estimate "
            "representing the expected future return from the current state-action pair. "
            "The Q-value should be a single floating-point number."
        )
        
        thinking_instructions = (
            "Before providing your final Q-value, you should engage in a detailed chain of thought. "
            "Enclose your entire reasoning process, step-by-step analysis, and any intermediate "
            "calculations or considerations within <thinking> and </thinking> tags. This thinking "
            "process can be as long and detailed as necessary to arrive at an accurate estimate. "
            "After the </thinking> tag, you must provide your final Q-value estimate."
        )
        
        no_thinking_instructions = (
            "You must provide ONLY the Q-value estimate as a single floating-point number."
        )

        q_value_format_instruction = (
            f"Your final Q-value estimate must be a single floating-point number enclosed in "
            f"'boxed{{}}', like so: {self.config.q_value_extraction_pattern.replace('(.*?)','Q_ESTIMATE')}. For example: boxed{{17.35}}."
        )

        if self.config.thinking:
            self.system_prompt_content = f"{base_prompt}\\n\\n{thinking_instructions}\\n\\n{q_value_format_instruction}"
        else:
            self.system_prompt_content = f"{base_prompt}\\n\\n{no_thinking_instructions}\\n\\n{q_value_format_instruction}"
        
        self.q_value_pattern = re.compile(self.config.q_value_extraction_pattern)
        self.thinking_block_pattern = re.compile(self.config.thinking_block_extraction_pattern, re.DOTALL)


    def _construct_evaluation_user_prompt_content(
        self,
        game_history_window: List[Message], # Window of game history, last message is action to evaluate
        # current_player_id: int # This was in the original, but not directly used by current logic. Re-add if needed.
    ) -> str:
        """
        Constructs the content for the 'user' message that will be sent to the RM LLM.
        This prompt asks the RM to evaluate the policy agent's proposed move, using provided game history.

        Args:
            game_history_window: A list of Message dicts. The last message in this list
                                 is assumed to be the policy agent's response (thinking + action)
                                 that the RM must evaluate. Earlier messages provide context,
                                 and may include system messages intended for the policy agent.

        Returns:
            A string representing the content of the user message for the RM.
        """
        env_system_prompt_contents = []
        formatted_history_parts = []
        action_to_evaluate_str = "Error: Action to evaluate not found or not in expected format in history window."

        if not game_history_window:
            logger.error(f"[{self.config.rm_id_for_logging}] _construct_evaluation_user_prompt_content called with empty game_history_window.")
            return "Critical Error: No game history provided for evaluation. Cannot proceed."

        history_context = game_history_window[:-1]
        action_message = game_history_window[-1]

        if action_message.get('role') == 'assistant':
            action_to_evaluate_str = action_message['content']
        else:
            logger.warning(
                f"[{self.config.rm_id_for_logging}] Last message in game_history_window (expected policy action) has role '{action_message.get('role')}' instead of 'assistant'. "
                f"Using its content directly: {action_message.get('content', '')[:100]}..."
            )
            action_to_evaluate_str = action_message.get('content', action_to_evaluate_str)

        for msg in history_context:
            if msg.get("role") == "system":
                env_system_prompt_contents.append(msg.get("content", ""))
            else:
                role_display = msg.get("role", "unknown_role").capitalize()
                content_display = msg.get("content", "[No content]")
                formatted_history_parts.append(f"{role_display}: {content_display}")

        prompt_sections = ["You are evaluating a policy agent's move based on the following context and proposed action."] # current_player_id removed for now

        if env_system_prompt_contents:
            prompt_sections.append("\n--- Environment System Prompt (Instructions for the Policy Agent) ---")
            prompt_sections.append("\n\n".join(filter(None, env_system_prompt_contents)))
            prompt_sections.append("--- End of Environment System Prompt ---")

        if formatted_history_parts:
            prompt_sections.append("\n--- Recent Game History (excluding the move to be evaluated) ---")
            prompt_sections.append("\n".join(formatted_history_parts))
            prompt_sections.append("--- End of Recent Game History ---")
        
        prompt_sections.append("\n--- Policy Agent's Proposed Move (to be evaluated) ---")
        prompt_sections.append(action_to_evaluate_str)
        prompt_sections.append("--- End of Proposed Move ---")

        # Use the q_value_extraction_pattern to inform the RM of the expected output format
        example_q_value_format = self.config.q_value_extraction_pattern.replace("([+-]?\\d*\\.?\\d+)", "Q_VALUE")
        prompt_sections.append(f"\nPlease provide your detailed evaluation of this move, including your own thinking process (if enabled in your setup) and a final Q-value score in {example_q_value_format} format.")
        
        return "\n\n".join(prompt_sections)

    def _parse_q_value_from_response(self, llm_response_content: str) -> Optional[float]:
        """
        Parses the Q-value from the LLM's response string using regex.
        Looks for \\boxed{float_value}.

        Args:
            llm_response_content: The raw text content from the LLM's response.

        Returns:
            The parsed float Q-value, or None if parsing fails.
        """
        match = self.q_value_pattern.search(llm_response_content)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                logger.warning(f"[{self.config.rm_id_for_logging}] Could not convert matched Q-value '{match.group(1)}' to float. Response: '{llm_response_content[:200]}...'")
                return None
        logger.debug(f"[{self.config.rm_id_for_logging}] Q-value pattern not found in response: '{llm_response_content[:200]}...'")
        return None

    def _extract_thinking_block(self, llm_response_content: str) -> Optional[str]:
        """
        Extracts the content of the <thinking>...</thinking> block from the LLM's response.

        Args:
            llm_response_content: The raw text content from the LLM's response.

        Returns:
            The content of the thinking block, or None if not found.
        """
        match = self.thinking_block_pattern.search(llm_response_content)
        if match:
            return match.group(1).strip()
        logger.debug(f"[{self.config.rm_id_for_logging}] Thinking block pattern not found in response: '{llm_response_content[:200]}...'")
        return None

    async def _sample_one_judgement(
        self,
        messages_for_llm_call: List[Message],
        # server_client: Any, # Already have self.server_client
        # For logging context
        log_prefix_base: str = ""
    ) -> Tuple[Optional[str], bool]:
        """
        Makes a single call to the LLM to get one judgement.
        Assumes messages_for_llm_call is correctly formatted by the caller (generate_g_judgements)
        typically as [RM_system_prompt_message, fully_constructed_user_prompt_message].

        Args:
            messages_for_llm_call: The list of messages to send to the LLM.
            # server_client: The API client for the LLM server. (Now uses self.server_client)
            log_prefix_base: Base prefix for logging.

        Returns:
            A tuple: (raw_llm_content_or_none, api_error_occurred_flag)
        """
        
        if not messages_for_llm_call or not any(msg.get('role') == 'user' for msg in messages_for_llm_call):
            logger.error(f"{log_prefix_base} RM LLM call attempted with no messages or no user message. Messages: {messages_for_llm_call}")
            return None, True # Error

        log_prompt_snippet = "PROMPT_SNIPPET_UNAVAILABLE"
        for msg in messages_for_llm_call:
            if msg.get('role') == 'user':
                log_prompt_snippet = msg.get('content', '')[:200]
                break
        
        effective_log_prefix = f"{log_prefix_base} [{self.config.rm_id_for_logging}]"
        logger.debug(f"{effective_log_prefix} LLM Chat Prompt for RM (first user message snippet): {log_prompt_snippet}")

        try:
            chat_completions = await self.server_client.chat_completion(
                messages=messages_for_llm_call,
                n=1,
                max_tokens=self.config.max_tokens_for_llm_output,
                temperature=self.config.temperature,
            )
            
            llm_generated_content = None
            if chat_completions.choices and \
               chat_completions.choices[0].message and \
               hasattr(chat_completions.choices[0].message, 'content') and \
               chat_completions.choices[0].message.content is not None:
                llm_generated_content = chat_completions.choices[0].message.content.strip()
            
            if not llm_generated_content:
                logger.warning(f"{effective_log_prefix} RM LLM returned empty or None content. Snippet: {log_prompt_snippet}")
                # Returning None, True indicates an issue, but not necessarily an API error if the call succeeded but content was empty
                return None, False # LLM call succeeded but no content, not an API error but problematic
            
            logger.debug(f"{effective_log_prefix} Raw content output from RM LLM: '{llm_generated_content[:300]}...'")
            return llm_generated_content, False # No API error

        except Exception as e:
            logger.error(f"{effective_log_prefix} RM LLM API (chat_completion) error: {e}. Snippet: {log_prompt_snippet}", exc_info=True)
            return None, True # API error occurred

    async def generate_g_judgements(
        self,
        num_judgements_g: int,
        game_history_window: List[Message],
        # current_player_id: int, # Removed from here, _construct_evaluation_user_prompt_content doesn't use it now
        # server_client: Any, # Already have self.server_client
        # Context for logging from the environment
        game_seed_for_logging: Optional[int] = None,
        turn_idx_for_logging: Optional[int] = None,
        policy_action_candidate_idx_for_logging: Optional[int] = None 
    ) -> List[Tuple[Optional[str], Optional[float], Optional[str]]]:
        """
        Generates G distinct judgements for a given policy agent's action.
        Each judgement includes the RM's thinking (if enabled) and a Q-value.

        Args:
            num_judgements_g: The number of judgements (G) to generate.
            game_history_window: The history of the game so far, where the last message is
                                 the policy agent's response (action) to be evaluated.
            # server_client: The API client for LLM calls.
            # Logging context arguments...

        Returns:
            A list of G tuples. Each tuple contains:
            (raw_llm_response_content, q_value, thinking_block_content)
            Values can be None if errors occur or content is not found.
        """
        log_prefix_parts = []
        if game_seed_for_logging is not None: log_prefix_parts.append(f"Seed {game_seed_for_logging}")
        if turn_idx_for_logging is not None: log_prefix_parts.append(f"Turn {turn_idx_for_logging}")
        if policy_action_candidate_idx_for_logging is not None: 
            log_prefix_parts.append(f"PolActIdx {policy_action_candidate_idx_for_logging}")
        log_prefix_base = f"[{self.config.rm_id_for_logging}] [{', '.join(log_prefix_parts)}] JudgementGen:"

        if not game_history_window:
            logger.error(f"{log_prefix_base} Called with empty game_history_window.")
            return [(None, None, None)] * num_judgements_g # Return G error results

        rm_system_prompt_message = Message(role="system", content=self.system_prompt_content)
        
        # Construct the user prompt content once, as it's the same for all G judgements
        # The current_player_id was removed from _construct_evaluation_user_prompt_content signature
        user_prompt_content = self._construct_evaluation_user_prompt_content(
            game_history_window=game_history_window,
            # current_player_id=current_player_id 
        )
        if "Critical Error:" in user_prompt_content:
            logger.error(f"{log_prefix_base} Failed to construct user prompt content: {user_prompt_content}")
            return [(None, None, None)] * num_judgements_g
        
        user_prompt_message = Message(role="user", content=user_prompt_content)
        messages_for_llm_call = [rm_system_prompt_message, user_prompt_message]

        tasks = []
        for i in range(num_judgements_g):
            current_log_prefix = f"{log_prefix_base} Sample {i+1}/{num_judgements_g}"
            tasks.append(self._sample_one_judgement(
                messages_for_llm_call=messages_for_llm_call, 
                # server_client=server_client, # Uses self.server_client
                log_prefix_base=current_log_prefix
            ))
        
        llm_responses_with_errors = await asyncio.gather(*tasks)

        results: List[Tuple[Optional[str], Optional[float], Optional[str]]] = []
        for raw_response_content, api_error in llm_responses_with_errors:
            if api_error or raw_response_content is None:
                logger.warning(f"{log_prefix_base} LLM call failed or returned None for a judgement sample.")
                results.append((raw_response_content, None, None))
                continue

            q_value = self._parse_q_value_from_response(raw_response_content)
            thinking_block = None
            if self.config.thinking: # Only try to extract thinking if it was enabled
                thinking_block = self._extract_thinking_block(raw_response_content)
            
            if q_value is None:
                 logger.warning(f"{log_prefix_base} Failed to parse Q-value for a judgement. Response snippet: {raw_response_content[:100]}...")
            
            results.append((raw_response_content, q_value, thinking_block))
        
        return results

# Example Usage (Conceptual - would be called from the Environment)
async def example_rm_usage(server_client: Any, tokenizer: PreTrainedTokenizer):
    # Dummy data for illustration
    current_obs = "Player X to move. Board: ... (detailed UTTT board) ..."
    policy_think = "I should try to win microboard 3, as it sets me up for a global win."
    policy_action = '<tool_call>\n{"arguments": {"micro_board": 3, "row": 1, "col": 1}, "name": "submit_move"}\n</tool_call>'
    player_id = 0 # Player X

    num_judgements = 3 # G value
    
    rm_config = AtroposRMConfig(thinking=True, temperature=0.2)
    rm_agent = AtroposRM(server_client=server_client, tokenizer=tokenizer, config=rm_config)
    
    judgements = await rm_agent.generate_g_judgements(
        num_judgements_g=num_judgements,
        current_game_observation_str=current_obs,
        policy_agent_thinking_str=policy_think,
        policy_agent_action_str=policy_action,
        current_player_id=player_id,
        server_client=server_client,
        game_seed_for_logging=123,
        turn_idx_for_logging=5,
        policy_action_candidate_idx_for_logging=1
    )

    effective_q_values_for_policy = []
    for i, judgement_result in enumerate(judgements):
        logger.info(f"Judgement {i+1}:")
        logger.info(f"  RM Thinking: {judgement_result[2]}")
        logger.info(f"  Parsed Q: {judgement_result[1]}")
        logger.info(f"  API Error: {judgement_result[0] is None}")
        if judgement_result[1] is not None and not judgement_result[0] is None:
            effective_q_values_for_policy.append(judgement_result[1])

    if effective_q_values_for_policy:
        mean_q_for_policy = sum(effective_q_values_for_policy) / len(effective_q_values_for_policy)
        logger.info(f"Mean Q-value from RM for this policy action: {mean_q_for_policy}")
    else:
        logger.warning("No valid Q-values obtained from RM for this policy action.")

    # The `judgements` list (specifically `rm_dialogue_history_this_judgement` and the final game outcome)
    # would then be used to create ScoredDataGroups for training the RM itself via GRPO.
