import logging
import re
from typing import List, Optional, Tuple, Any, Dict, TypedDict
import json
import asyncio
from transformers import PreTrainedTokenizer
from pydantic import BaseModel, Field
from atroposlib.type_definitions import Message
from atroposlib.utils.boxed_parser import extract_boxed_content

# A common representation for chat messages
# Message = Dict[str, str] # This local incorrect definition will be removed

logger = logging.getLogger(__name__)

# --- Define RMJudgementLog locally ---
class RMJudgementLog(TypedDict):
    """
    Represents a single judgement made by the AtroposRM for one sample,
    often one of G judgements for a specific policy agent's action.
    """
    # Contextual information
    game_seed_for_logging: Optional[int]
    turn_idx_for_logging: Optional[int]
    policy_action_candidate_idx_for_logging: Optional[int]

    # Input to the RM's LLM
    rm_input_messages: List[Message]

    # Output from the RM's LLM and parsing results
    raw_rm_response_content: Optional[str]
    parsed_q_value: Optional[float]
    parsed_thinking_block: Optional[str]

    # Status/Error Flags
    api_error: bool
    q_value_parse_error: bool
    thinking_block_parse_error: bool
# --- End of RMJudgementLog Definition ---

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

        policy_instruction_awareness_prompt = (
            "The game history provided to you in the user message will contain the policy agent's own system "
            "instructions, typically at the beginning of the dialogue history, clearly marked with "
            "'--- Policy Agent System Instructions ---' and '--- End of Policy Agent System Instructions ---'. "
            "You MUST consider these instructions when evaluating the policy agent's adherence, reasoning, and proposed action."
        )
        
        thinking_instructions = (
            "Before providing your final Q-value, you should engage in a detailed chain of thought. "
            "Enclose your entire reasoning process, step-by-step analysis, and any intermediate "
            "calculations or considerations within <thinking> and </thinking> tags. This thinking "
            "process can be as long and detailed as necessary to arrive at an accurate estimate. "
            "After the </thinking> tag, YOU MUST IMMEDIATELY provide your final Q-value estimate. No other text should follow the </thinking> tag before the Q-value."
        )
        
        no_thinking_instructions = (
            "You must provide ONLY the Q-value estimate as a single floating-point number. Nothing else."
        )

        q_value_format_instruction = (
            "Your final Q-value estimate MUST be a single floating-point number. "
            "This number MUST be enclosed in \\boxed{}. For example: \\boxed{17.35} or \\boxed{-2.5}. "
            "This \\boxed{} expression is MANDATORY and MUST be the absolute final part of your response. "
            "If thinking is enabled, it appears immediately after the </thinking> tag. If thinking is disabled, it is your entire response."
        )

        if self.config.thinking:
            self.system_prompt_content = f"{base_prompt}\n\n{policy_instruction_awareness_prompt}\n\n{thinking_instructions}\n\n{q_value_format_instruction}"
        else:
            self.system_prompt_content = f"{base_prompt}\n\n{policy_instruction_awareness_prompt}\n\n{no_thinking_instructions}\n\n{q_value_format_instruction}"
        
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
        # env_system_prompt_contents = [] # No longer needed as a separate list
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

        for msg in history_context: # This includes the policy agent's system prompt if it's first
            role_display = msg.get("role", "unknown_role").capitalize()
            content_display = msg.get("content", "[No content]")
            if msg.get("role") == "system":
                # Embed the policy agent's system prompt with clear markers
                formatted_history_parts.append(f"--- Policy Agent System Instructions ---\n{content_display}\n--- End of Policy Agent System Instructions ---")
            else:
                formatted_history_parts.append(f"{role_display}: {content_display}")

        prompt_sections = ["You are evaluating a policy agent's move based on the following context and proposed action."]

        # The policy system prompt is now part of formatted_history_parts
        # if env_system_prompt_contents:
        #     prompt_sections.append("\n--- Environment System Prompt (Instructions for the Policy Agent) ---")
        #     prompt_sections.append("\n\n".join(filter(None, env_system_prompt_contents)))
        #     prompt_sections.append("--- End of Environment System Prompt ---")

        if formatted_history_parts:
            prompt_sections.append("\n--- Recent Game History (including Policy Agent Instructions if provided, and excluding the move to be evaluated) ---")
            prompt_sections.append("\n".join(formatted_history_parts))
            prompt_sections.append("--- End of Recent Game History ---")
        
        prompt_sections.append("\n--- Policy Agent's Proposed Move (to be evaluated) ---")
        prompt_sections.append(action_to_evaluate_str)
        prompt_sections.append("--- End of Proposed Move ---")

        # Simplified instruction, as the exact format details are now handled by the parser and RM prompt.
        prompt_sections.append(f"\nPlease provide your detailed evaluation of this move, including your own thinking process (if enabled in your setup) and a final Q-value score in \\boxed{{Q_VALUE}} format.")
        
        return "\n\n".join(prompt_sections)

    def _parse_q_value_from_response(self, llm_response_content: str) -> Optional[float]:
        """
        Parses the Q-value from the LLM's response string using the extract_boxed_content utility.
        
        Args:
            llm_response_content: The raw text content from the LLM's response.

        Returns:
            The parsed float Q-value, or None if parsing fails.
        """
        logger.debug(f"[{self.config.rm_id_for_logging}] Attempting to parse Q-value from raw response (repr): {repr(llm_response_content)}")
        
        boxed_content = extract_boxed_content(llm_response_content)
        
        if boxed_content is not None:
            try:
                return float(boxed_content)
            except ValueError:
                logger.warning(f"[{self.config.rm_id_for_logging}] Could not convert extracted boxed content '{boxed_content}' to float. Response: '{llm_response_content[:200]}...'")
                return None
        
        logger.debug(f"[{self.config.rm_id_for_logging}] No boxed Q-value found in response: '{llm_response_content[:200]}...'")
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
        game_seed_for_logging: Optional[int] = None,
        turn_idx_for_logging: Optional[int] = None,
        policy_action_candidate_idx_for_logging: Optional[int] = None
    ) -> List[RMJudgementLog]:
        """
        Generates G distinct judgements for a given policy agent's action.
        Each judgement includes the RM's thinking (if enabled), a Q-value, and detailed logging info.

        Args:
            num_judgements_g: The number of judgements (G) to generate.
            game_history_window: The history of the game so far.
            # ... (other args remain the same)

        Returns:
            A list of G RMJudgementLog dictionaries.
        """
        log_prefix_parts = []
        if game_seed_for_logging is not None: log_prefix_parts.append(f"Seed {game_seed_for_logging}")
        if turn_idx_for_logging is not None: log_prefix_parts.append(f"Turn {turn_idx_for_logging}")
        if policy_action_candidate_idx_for_logging is not None:
            log_prefix_parts.append(f"PolActIdx {policy_action_candidate_idx_for_logging}")
        log_prefix_base = f"[{self.config.rm_id_for_logging}] [{', '.join(log_prefix_parts)}] JudgementGen:"

        results: List[RMJudgementLog] = []

        if not game_history_window:
            logger.error(f"{log_prefix_base} Called with empty game_history_window.")
            # Still return G items, but marked with errors
            for _ in range(num_judgements_g):
                results.append(RMJudgementLog(
                    game_seed_for_logging=game_seed_for_logging,
                    turn_idx_for_logging=turn_idx_for_logging,
                    policy_action_candidate_idx_for_logging=policy_action_candidate_idx_for_logging,
                    rm_input_messages=[], # No input messages could be formed
                    raw_rm_response_content=None,
                    parsed_q_value=None,
                    parsed_thinking_block=None,
                    api_error=True, # Treat as an API error equivalent
                    q_value_parse_error=True,
                    thinking_block_parse_error=self.config.thinking # True if thinking was expected
                ))
            return results

        rm_system_prompt_message = Message(role="system", content=self.system_prompt_content)
        user_prompt_content = self._construct_evaluation_user_prompt_content(
            game_history_window=game_history_window,
        )

        if "Critical Error:" in user_prompt_content:
            logger.error(f"{log_prefix_base} Failed to construct user prompt content: {user_prompt_content}")
            for _ in range(num_judgements_g):
                results.append(RMJudgementLog(
                    game_seed_for_logging=game_seed_for_logging,
                    turn_idx_for_logging=turn_idx_for_logging,
                    policy_action_candidate_idx_for_logging=policy_action_candidate_idx_for_logging,
                    rm_input_messages=[rm_system_prompt_message], # Only system prompt could be formed
                    raw_rm_response_content=None,
                    parsed_q_value=None,
                    parsed_thinking_block=None,
                    api_error=True, 
                    q_value_parse_error=True,
                    thinking_block_parse_error=self.config.thinking
                ))
            return results
        
        user_prompt_message = Message(role="user", content=user_prompt_content)
        messages_for_llm_call = [rm_system_prompt_message, user_prompt_message]

        tasks = []
        for i in range(num_judgements_g):
            current_log_prefix = f"{log_prefix_base} Sample {i+1}/{num_judgements_g}"
            tasks.append(self._sample_one_judgement(
                messages_for_llm_call=messages_for_llm_call,
                log_prefix_base=current_log_prefix
            ))
        
        llm_responses_with_errors_flags = await asyncio.gather(*tasks)

        for raw_response_content, api_error_occurred in llm_responses_with_errors_flags:
            q_value = None
            thinking_block = None
            q_parse_error_flag = False
            tb_parse_error_flag = False

            if not api_error_occurred and raw_response_content is not None:
                q_value = self._parse_q_value_from_response(raw_response_content)
                if q_value is None:
                    q_parse_error_flag = True
                    logger.warning(f"{log_prefix_base} Failed to parse Q-value for a judgement. Response snippet: {raw_response_content[:100]}...")

                if self.config.thinking:
                    thinking_block = self._extract_thinking_block(raw_response_content)
                    if thinking_block is None:
                        tb_parse_error_flag = True
                        # Optional: log if thinking block parsing fails when expected
                        logger.debug(f"{log_prefix_base} Thinking block expected but not found/parsed. Response snippet: {raw_response_content[:100]}...")
            
            elif api_error_occurred: # If API error, implies parsing errors for q-val and thinking
                q_parse_error_flag = True
                if self.config.thinking:
                    tb_parse_error_flag = True
            
            elif raw_response_content is None and not api_error_occurred: # LLM returned empty content
                q_parse_error_flag = True # Cannot parse Q from None
                if self.config.thinking:
                    tb_parse_error_flag = True # Cannot parse thinking from None


            results.append(RMJudgementLog(
                game_seed_for_logging=game_seed_for_logging,
                turn_idx_for_logging=turn_idx_for_logging,
                policy_action_candidate_idx_for_logging=policy_action_candidate_idx_for_logging,
                rm_input_messages=list(messages_for_llm_call), # Store a copy
                raw_rm_response_content=raw_response_content,
                parsed_q_value=q_value,
                parsed_thinking_block=thinking_block,
                api_error=api_error_occurred,
                q_value_parse_error=q_parse_error_flag,
                thinking_block_parse_error=tb_parse_error_flag and self.config.thinking # only true if thinking was on
            ))
        
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
        logger.info(f"  RM Thinking: {judgement_result['parsed_thinking_block']}")
        logger.info(f"  Parsed Q: {judgement_result['parsed_q_value']}")
        logger.info(f"  API Error: {judgement_result['api_error']}")
        if judgement_result['parsed_q_value'] is not None and not judgement_result['api_error']:
            effective_q_values_for_policy.append(judgement_result['parsed_q_value'])

    if effective_q_values_for_policy:
        mean_q_for_policy = sum(effective_q_values_for_policy) / len(effective_q_values_for_policy)
        logger.info(f"Mean Q-value from RM for this policy action: {mean_q_for_policy}")
    else:
        logger.warning("No valid Q-values obtained from RM for this policy action.")

    # The `judgements` list (specifically `rm_dialogue_history_this_judgement` and the final game outcome)
    # would then be used to create ScoredDataGroups for training the RM itself via GRPO.
