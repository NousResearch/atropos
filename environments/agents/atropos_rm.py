import logging
import re
from typing import List, Optional, Tuple, Any, Dict
import json
from transformers import PreTrainedTokenizer

# A common representation for chat messages
Message = Dict[str, str]

logger = logging.getLogger(__name__)

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
        tokenizer: PreTrainedTokenizer,  # Tokenizer instance (e.g., from HuggingFace)
        temperature: float,
        max_context_token_length: int, # Max tokens for the LLM's context window
        max_tokens_for_llm_output: int, # Max tokens the RM LLM should generate for one judgement
        thinking: bool = True, # Whether the RM should use <thinking> tags
        rm_id_for_logging: Optional[str] = "RM"
    ):
        """
        Initializes the AtroposRM.

        Args:
            tokenizer: The tokenizer used for token counting if necessary (e.g., by a
                       client library or for context length checks, though context checks
                       are not explicitly implemented in this version of _sample_one_judgement).
            temperature: The sampling temperature for the LLM.
            max_context_token_length: The maximum token length for the LLM's context window.
            max_tokens_for_llm_output: The maximum number of tokens the LLM should generate
                                       for each of its G judgements.
            thinking: If True, RM is instructed to use <thinking> tags. If False,
                      it's instructed to output only the Q-value.
            rm_id_for_logging: An identifier for logging purposes.
        """
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_context_token_length = max_context_token_length
        self.max_tokens_for_llm_output = max_tokens_for_llm_output
        self.enable_thinking = thinking
        self.rm_id = rm_id_for_logging
        
        # Construct the system prompt
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
            "Your final Q-value estimate must be a single floating-point number enclosed in "
            "\\boxed{{}}, like so: \\boxed{{Q_ESTIMATE}}. For example: \\boxed{{17.35}}."
        )

        if self.enable_thinking:
            self.system_prompt_content = f"{base_prompt}\\n\\n{thinking_instructions}\\n\\n{q_value_format_instruction}"
        else:
            self.system_prompt_content = f"{base_prompt}\\n\\n{no_thinking_instructions}\\n\\n{q_value_format_instruction}"
        
        # Regex to extract Q-value, e.g., \\boxed{0.5} or \\boxed{-0.23}
        self.q_value_pattern = re.compile(r"\\boxed{\\s*([+-]?\\d*\\.?\\d+)\\s*}")
        # Regex to extract content within <thinking>...</thinking> tags
        self.thinking_block_pattern = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)


    def _construct_evaluation_user_prompt_content(
        self,
        game_history_window: List[Message], # Window of game history, last message is action to evaluate
        current_player_id: int
    ) -> str:
        """
        Constructs the content for the 'user' message that will be sent to the RM LLM.
        This prompt asks the RM to evaluate the policy agent's proposed move, using provided game history.

        Args:
            game_history_window: A list of Message dicts. The last message in this list
                                 is assumed to be the policy agent's response (thinking + action)
                                 that the RM must evaluate. Earlier messages provide context,
                                 and may include system messages intended for the policy agent.
            current_player_id: The ID of the player whose move is being evaluated.

        Returns:
            A string representing the content of the user message for the RM.
        """
        env_system_prompt_contents = []
        formatted_history_parts = []
        action_to_evaluate_str = "Error: Action to evaluate not found or not in expected format in history window."

        if not game_history_window:
            logger.error(f"[{self.rm_id}] _construct_evaluation_user_prompt_content called with empty game_history_window.")
            # Return a prompt that indicates an error, or handle as per desired strictness
            return "Critical Error: No game history provided for evaluation. Cannot proceed."

        # The last message is the action to be evaluated.
        # Messages before the last one form the context (history and potential env system prompts).
        history_context = game_history_window[:-1]
        action_message = game_history_window[-1]

        # Extract the content of the action to be evaluated
        # We assume the policy agent's output (what the RM judges) has the role 'assistant'
        # or contains the thinking/action structure we expect.
        if action_message.get('role') == 'assistant': # Standard role for LLM agent responses
            action_to_evaluate_str = action_message['content']
        else:
            logger.warning(
                f"[{self.rm_id}] Last message in game_history_window (expected policy action) has role '{action_message.get('role')}' instead of 'assistant'. "
                f"Using its content directly: {action_message.get('content', '')[:100]}..."
            )
            # Fallback: use the content of the last message directly, regardless of role, if it's not 'assistant'.
            # This might happen if the environment uses different role conventions for the policy agent's turn.
            action_to_evaluate_str = action_message.get('content', action_to_evaluate_str)

        # Process the history context for environment system prompts and game turns
        for msg in history_context:
            if msg.get("role") == "system":
                env_system_prompt_contents.append(msg.get("content", ""))
            else:
                role_display = msg.get("role", "unknown_role").capitalize()
                content_display = msg.get("content", "[No content]")
                formatted_history_parts.append(f"{role_display}: {content_display}")

        # Build the final prompt string sections
        prompt_sections = [f"You are evaluating a move for Player {current_player_id}."]

        if env_system_prompt_contents:
            prompt_sections.append("\n--- Environment System Prompt (Instructions for the Policy Agent) ---")
            prompt_sections.append("\n\n".join(filter(None, env_system_prompt_contents))) # Join non-empty prompts
            prompt_sections.append("--- End of Environment System Prompt ---")

        if formatted_history_parts:
            prompt_sections.append("\n--- Recent Game History (excluding the move to be evaluated) ---")
            prompt_sections.append("\n".join(formatted_history_parts))
            prompt_sections.append("--- End of Recent Game History ---")
        
        prompt_sections.append("\n--- Policy Agent's Proposed Move (to be evaluated) ---")
        prompt_sections.append(action_to_evaluate_str)
        prompt_sections.append("--- End of Proposed Move ---")

        prompt_sections.append("\nPlease provide your detailed evaluation of this move, including your own thinking process and a final Q-value score in \\boxed{{Q_VALUE}} format.")
        
        return "\n\n".join(prompt_sections) # Join all major sections with double newlines

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
                logger.warning(f"[{self.rm_id}] Could not convert matched Q-value '{match.group(1)}' to float. Response: '{llm_response_content[:200]}...'")
                return None
        logger.debug(f"[{self.rm_id}] Q-value pattern not found in response: '{llm_response_content[:200]}...'")
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
        logger.debug(f"[{self.rm_id}] Thinking block pattern not found in response: '{llm_response_content[:200]}...'")
        return None

    async def _sample_one_judgement(
        self,
        messages_for_llm_call: List[Message],
        server_client: Any,
        # For logging context
        log_prefix_base: str = ""
    ) -> Tuple[Optional[str], bool]:
        """
        Makes a single call to the LLM to get one judgement.
        Assumes messages_for_llm_call is correctly formatted by the caller (generate_g_judgements)
        typically as [RM_system_prompt_message, fully_constructed_user_prompt_message].

        Args:
            messages_for_llm_call: The list of messages to send to the LLM.
            server_client: The API client for the LLM server.
            log_prefix_base: Base prefix for logging.

        Returns:
            A tuple: (raw_llm_content_or_none, api_error_occurred_flag)
        """
        
        if not messages_for_llm_call or not any(msg.get('role') == 'user' for msg in messages_for_llm_call):
            logger.error(f"{log_prefix_base} RM LLM call attempted with no messages or no user message. Messages: {messages_for_llm_call}")
            return None, True # Error

        # Log a snippet of the first user message for context
        log_prompt_snippet = "PROMPT_SNIPPET_UNAVAILABLE"
        for msg in messages_for_llm_call:
            if msg.get('role') == 'user':
                log_prompt_snippet = msg.get('content', '')[:200]
                break
        
        logger.debug(f"{log_prefix_base} LLM Chat Prompt for RM (first user message snippet): {log_prompt_snippet}")
        # For very detailed debugging, you might log the full messages_for_llm_call, but be wary of verbosity.
        # logger.debug(f"{log_prefix_base} Final messages being sent to LLM: {json.dumps(messages_for_llm_call, indent=2)}")

        try:
            chat_completions = await server_client.chat_completion(
                messages=messages_for_llm_call,
                n=1,
                max_tokens=self.max_tokens_for_llm_output,
                temperature=self.temperature,
            )
            
            llm_generated_content = None
            if chat_completions.choices and \
               chat_completions.choices[0].message and \
               hasattr(chat_completions.choices[0].message, 'content') and \
               chat_completions.choices[0].message.content is not None:
                llm_generated_content = chat_completions.choices[0].message.content.strip()
            
            if not llm_generated_content:
                logger.warning(f"{log_prefix_base} RM LLM returned empty or None content. User prompt snippet: {log_prompt_snippet}")
                return None, False 
            
            logger.debug(f"{log_prefix_base} RM raw content output: '{llm_generated_content[:300]}...'")
            return llm_generated_content, False

        except Exception as e:
            logger.error(f"{log_prefix_base} RM LLM API (chat_completion) error: {e}. User prompt snippet: {log_prompt_snippet}")
            return None, True

    async def generate_g_judgements(
        self,
        num_judgements_g: int,
        game_history_window: List[Message], # This includes env system prompts, history, and the action to be evaluated.
        current_player_id: int,             # Player ID whose move is being evaluated
        server_client: Any,
        # Context for logging from the environment
        game_seed_for_logging: Optional[int] = None,
        turn_idx_for_logging: Optional[int] = None,
        policy_action_candidate_idx_for_logging: Optional[int] = None # If N > 1 for policy
    ) -> List[Dict[str, Any]]:
        """
        Generates G distinct judgements from the RM for a single proposed policy action.
        Each judgement includes the RM's thinking process and a Q-value.

        Args:
            num_judgements_g: The number of judgements (G) to generate.
            game_history_window: A list of Message dicts representing the game context.
                                 The last message is the policy agent's action to be evaluated.
                                 May contain environment system prompts for the policy agent.
            current_player_id: The ID of the player whose action is being evaluated.
            server_client: The API client for the LLM server.
            game_seed_for_logging, turn_idx_for_logging, policy_action_candidate_idx_for_logging: For logging.

        Returns:
            A list of G dictionaries, where each dictionary represents one judgement and contains:
            - "rm_thinking_block": Optional[str], The RM's thinking.
            - "parsed_q_value": Optional[float], The Q-value parsed from the RM's response.
            - "raw_llm_response_content": Optional[str], The full raw response from the RM.
            - "rm_dialogue_history_this_judgement": List[Message], The dialogue (system, user, assistant)
                                                     for this specific judgement call, for GRPO training of RM.
            - "api_error": bool, True if an API error occurred for this judgement.
            - "parse_error": bool, True if Q-value parsing failed for this judgement.
        """
        
        results: List[Dict[str, Any]] = []

        user_prompt_content = self._construct_evaluation_user_prompt_content(
            game_history_window=game_history_window,
            current_player_id=current_player_id
        )

        for i in range(num_judgements_g):
            log_prefix_parts = [f"RM[{self.rm_id}]"]
            if game_seed_for_logging is not None: log_prefix_parts.append(f"Seed {game_seed_for_logging}")
            if turn_idx_for_logging is not None: log_prefix_parts.append(f"Turn {turn_idx_for_logging}")
            if policy_action_candidate_idx_for_logging is not None: log_prefix_parts.append(f"PolActCand {policy_action_candidate_idx_for_logging}")
            log_prefix_parts.append(f"JudgeNum {i+1}/{num_judgements_g}")
            current_judgement_log_prefix = f"[{' '.join(log_prefix_parts)}]"

            # Each judgement is an independent LLM call with the same initial setup (system prompt, user prompt).
            # The dialogue history for *this specific judgement* is what's needed for GRPO training of the RM.
            messages_for_this_judgement_call: List[Message] = [
                {"role": "system", "content": self.system_prompt_content},
                {"role": "user", "content": user_prompt_content}
            ]
            
            # TODO: Add token length check for messages_for_this_judgement_call + max_tokens_for_llm_output
            # against self.max_context_token_length. This requires tokenizer.apply_chat_template
            # and tokenizer.encode. For brevity in this initial implementation, it's omitted but crucial for robustness.
            # Example check (conceptual):
            # prompt_str_for_token_check = self.tokenizer.apply_chat_template(messages_for_this_judgement_call, tokenize=False, add_generation_prompt=True)
            # current_prompt_token_count = len(self.tokenizer.encode(prompt_str_for_token_check))
            # if current_prompt_token_count + self.max_tokens_for_llm_output > self.max_context_token_length:
            #     logger.warning(f"{current_judgement_log_prefix} Token limit for RM judgement. Skipping.")
            #     results.append({ ... "api_error": True, "token_limit_exceeded_flag_for_rm_itself": True ... }) # Special flag
            #     continue


            raw_llm_response_content, api_error = await self._sample_one_judgement(
                messages_for_llm_call=messages_for_this_judgement_call,
                server_client=server_client,
                log_prefix_base=current_judgement_log_prefix
            )

            # Append the RM's response to its dialogue history for this specific judgement
            # This forms the complete exchange for this RM "action" (judgement).
            if raw_llm_response_content is not None and not api_error:
                messages_for_this_judgement_call.append({"role": "assistant", "content": raw_llm_response_content})
            elif api_error:
                 messages_for_this_judgement_call.append({"role": "assistant", "content": "<RM_LLM_API_ERROR>"})
            else: # raw_llm_response_content is None but no API error (e.g. empty from LLM)
                 messages_for_this_judgement_call.append({"role": "assistant", "content": "<RM_LLM_EMPTY_RESPONSE>"})


            parsed_q_value = None
            rm_thinking_block = None
            parse_error = False

            if raw_llm_response_content and not api_error:
                rm_thinking_block = self._extract_thinking_block(raw_llm_response_content)
                parsed_q_value = self._parse_q_value_from_response(raw_llm_response_content)
                if parsed_q_value is None:
                    logger.warning(f"{current_judgement_log_prefix} Failed to parse Q-value from RM response.")
                    parse_error = True
            
            results.append({
                "rm_thinking_block": rm_thinking_block, # Can use this for feedback to the policy agent, self-critique, etc.
                "parsed_q_value": parsed_q_value,
                "raw_llm_response_content": raw_llm_response_content,
                "rm_dialogue_history_this_judgement": messages_for_this_judgement_call, # History for this specific judgement
                "api_error": api_error,
                "parse_error": parse_error
            })
            
        return results

# Example Usage (Conceptual - would be called from the Environment)
async def example_rm_usage(rm_agent: AtroposRM, server_client: Any):
    # Dummy data for illustration
    current_obs = "Player X to move. Board: ... (detailed UTTT board) ..."
    policy_think = "I should try to win microboard 3, as it sets me up for a global win."
    policy_action = '<tool_call>\n{"arguments": {"micro_board": 3, "row": 1, "col": 1}, "name": "submit_move"}\n</tool_call>'
    player_id = 0 # Player X

    num_judgements = 3 # G value
    
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
        logger.info(f"  RM Thinking: {judgement_result['rm_thinking_block']}")
        logger.info(f"  Parsed Q: {judgement_result['parsed_q_value']}")
        logger.info(f"  API Error: {judgement_result['api_error']}")
        logger.info(f"  Parse Error: {judgement_result['parse_error']}")
        # logger.info(f"  RM Dialogue: {judgement_result['rm_dialogue_history_this_judgement']}") # Can be verbose
        if judgement_result['parsed_q_value'] is not None and not judgement_result['api_error']:
            effective_q_values_for_policy.append(judgement_result['parsed_q_value'])

    if effective_q_values_for_policy:
        mean_q_for_policy = sum(effective_q_values_for_policy) / len(effective_q_values_for_policy)
        logger.info(f"Mean Q-value from RM for this policy action: {mean_q_for_policy}")
    else:
        logger.warning("No valid Q-values obtained from RM for this policy action.")

    # The `judgements` list (specifically `rm_dialogue_history_this_judgement` and the final game outcome)
    # would then be used to create ScoredDataGroups for training the RM itself via GRPO.
