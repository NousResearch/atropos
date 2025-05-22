import logging
import re
from typing import List, Optional, Tuple, Any, TypedDict, Dict
import json
import asyncio

# Smol Gents imports
from smolagents.models import Model, ChatMessage, MessageRole

from transformers import PreTrainedTokenizer
from pydantic import BaseModel, Field
from atroposlib.type_definitions import Message
from atroposlib.envs.base import ServerManager
from atroposlib.utils.boxed_parser import extract_boxed_content

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
    rm_input_messages: List[Dict[str, str]]

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
    model_id: str = Field(default="gpt-4-turbo", description="LLM model ID for smolagents.Model and server calls.")
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0, 
        description="Sampling temperature for the RM LLM."
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
        default="AtroposRM",
        description="Identifier for the RM in logs."
    )

    class Config:
        extra = 'forbid'
# --- End of AtroposRMConfig Definition ---

class AtroposRM(Model):
    """
    Atropos Reward Model (RM).
    Evaluates a policy agent's proposed action, generating G judgements (thinking + Q-value).
    Implements the smolagents.Model interface.
    """

    def __init__(
        self,
        server_client: ServerManager, 
        tokenizer: PreTrainedTokenizer,  
        config: Optional[AtroposRMConfig] = None,
    ):
        self.config = config if config is not None else AtroposRMConfig()
        super().__init__(model_id=self.config.model_id)

        self.server_client = server_client
        self.tokenizer = tokenizer
        
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

        # For smolagents.Model interface compliance
        self.last_input_token_count: Optional[int] = None
        self.last_output_token_count: Optional[int] = None

    def _construct_evaluation_user_prompt_content(
        self,
        game_history_window: List[Message],
    ) -> str:
        """
        Constructs the content for the 'user' message that will be sent to the RM LLM.
        Args:
            game_history_window: A list of Atropos Message dicts.
        Returns:
            A string representing the content of the user message for the RM.
        """
        formatted_history_parts = []
        action_to_evaluate_str = "Error: Action to evaluate not found or not in expected format in history window."

        if not game_history_window:
            logger.error(f"[{self.config.rm_id_for_logging}] _construct_evaluation_user_prompt_content called with empty game_history_window.")
            return "Critical Error: No game history provided for evaluation. Cannot proceed."

        history_context = game_history_window[:-1]
        action_message_dict = game_history_window[-1]

        if action_message_dict.get('role') == 'assistant':
            action_to_evaluate_str = action_message_dict['content']
        else:
            logger.warning(
                f"[{self.config.rm_id_for_logging}] Last message in game_history_window (expected policy action) has role '{action_message_dict.get('role')}' instead of 'assistant'. "
                f"Using its content directly: {action_message_dict.get('content', '')[:100]}..."
            )
            action_to_evaluate_str = action_message_dict.get('content', action_to_evaluate_str)

        for msg_dict in history_context:
            role_display = msg_dict.get("role", "unknown_role").capitalize()
            content_display = msg_dict.get("content", "[No content]")
            if msg_dict.get("role") == "system":
                formatted_history_parts.append(f"--- Policy Agent System Instructions ---\n{content_display}\n--- End of Policy Agent System Instructions ---")
            else:
                formatted_history_parts.append(f"{role_display}: {content_display}")

        prompt_sections = ["You are evaluating a policy agent's move based on the following context and proposed action."]
        if formatted_history_parts:
            prompt_sections.append("\n--- Recent Game History (including Policy Agent Instructions if provided, and excluding the move to be evaluated) ---")
            prompt_sections.append("\n".join(formatted_history_parts))
            prompt_sections.append("--- End of Recent Game History ---")
        
        prompt_sections.append("\n--- Policy Agent's Proposed Move (to be evaluated) ---")
        prompt_sections.append(action_to_evaluate_str)
        prompt_sections.append("--- End of Proposed Move ---")
        prompt_sections.append(f"\nPlease provide your detailed evaluation of this move, including your own thinking process (if enabled in your setup) and a final Q-value score in \\boxed{{Q_VALUE}} format.")
        
        return "\n\n".join(prompt_sections)

    def _parse_q_value_from_response(self, llm_response_content: str) -> Optional[float]:
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
        match = self.thinking_block_pattern.search(llm_response_content)
        if match:
            return match.group(1).strip()
        logger.debug(f"[{self.config.rm_id_for_logging}] Thinking block pattern not found in response: '{llm_response_content[:200]}...'")
        return None

    async def generate(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> ChatMessage:
        api_kwargs = {
            "model": self.config.model_id,
            "n": kwargs.get("n", 1),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens_for_llm_output),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stop": stop_sequences,
        }
        if api_kwargs["stop"] is None:
            del api_kwargs["stop"]

        log_prompt_snippet = (
            messages[-1]['content'][:200]
            if messages and isinstance(messages[-1], dict) and messages[-1].get('content') 
            else "EMPTY_HISTORY_CONTENT"
        )
        logger.debug(f"[{self.config.rm_id_for_logging}] (smol.generate) LLM Chat Prompt (last message snippet): {log_prompt_snippet}")

        try:
            raw_response = await self.server_client.chat_completion(
                messages=messages, 
                **api_kwargs
            )
            content = "No response content or error."
            role = MessageRole.ASSISTANT
            
            if raw_response and raw_response.choices:
                first_choice = raw_response.choices[0]
                if first_choice.message and first_choice.message.content is not None:
                    content = first_choice.message.content.strip()
                else:
                    logger.warning(f"[{self.config.rm_id_for_logging}] (smol.generate) LLM first choice had no message content.")
            else:
                 logger.warning(f"[{self.config.rm_id_for_logging}] (smol.generate) chat_completion returned no choices. Snippet: {log_prompt_snippet}")

            if hasattr(raw_response, "usage") and raw_response.usage:
                self.last_input_token_count = raw_response.usage.prompt_tokens
                self.last_output_token_count = raw_response.usage.completion_tokens
            else:
                self.last_input_token_count = None
                self.last_output_token_count = None
            
            return ChatMessage(role=role, content=content, raw=raw_response)

        except Exception as e:
            logger.error(f"[{self.config.rm_id_for_logging}] (smol.generate) LLM API error: {e}. Snippet: {log_prompt_snippet}", exc_info=True)
            raise ValueError(f"LLM API error during RM smol.generate: {e}")

    async def generate_g_judgements(
        self,
        num_judgements_g: int,
        game_history_window: List[Message],
        game_seed_for_logging: Optional[int] = None,
        turn_idx_for_logging: Optional[int] = None,
        policy_action_candidate_idx_for_logging: Optional[int] = None
    ) -> List[RMJudgementLog]:
        log_prefix_parts = []
        if game_seed_for_logging is not None: log_prefix_parts.append(f"Seed {game_seed_for_logging}")
        if turn_idx_for_logging is not None: log_prefix_parts.append(f"Turn {turn_idx_for_logging}")
        if policy_action_candidate_idx_for_logging is not None:
            log_prefix_parts.append(f"PolActIdx {policy_action_candidate_idx_for_logging}")
        log_prefix_base = f"[{self.config.rm_id_for_logging}] [{', '.join(log_prefix_parts)}] JudgementGen:"

        results: List[RMJudgementLog] = []

        if not game_history_window:
            logger.error(f"{log_prefix_base} Called with empty game_history_window.")
            for _ in range(num_judgements_g):
                results.append(self._create_error_judgement_log(
                    game_seed_for_logging, turn_idx_for_logging, policy_action_candidate_idx_for_logging, []
                ))
            return results

        rm_system_prompt_dict = {"role": "system", "content": self.system_prompt_content}
        user_prompt_content_str = self._construct_evaluation_user_prompt_content(
            game_history_window=game_history_window,
        )

        if "Critical Error:" in user_prompt_content_str:
            logger.error(f"{log_prefix_base} Failed to construct user prompt content: {user_prompt_content_str}")
            for _ in range(num_judgements_g):
                results.append(self._create_error_judgement_log(
                    game_seed_for_logging, turn_idx_for_logging, policy_action_candidate_idx_for_logging, [rm_system_prompt_dict]
                ))
            return results
        
        user_prompt_dict = {"role": "user", "content": user_prompt_content_str}
        messages_for_llm_call = [rm_system_prompt_dict, user_prompt_dict]
        
        raw_rm_outputs_content: List[Optional[str]] = []
        api_error_overall = False

        try:
            logger.debug(f"{log_prefix_base} Making single LLM call for {num_judgements_g} judgements. User prompt snippet: {user_prompt_content_str[:200]}...")
            chat_completions = await self.server_client.chat_completion(
                messages=messages_for_llm_call,
                model=self.config.model_id,
                n=num_judgements_g,
                max_tokens=self.config.max_tokens_for_llm_output,
                temperature=self.config.temperature,
            )

            if chat_completions and chat_completions.choices:
                for choice_idx, choice in enumerate(chat_completions.choices):
                    if choice.message and hasattr(choice.message, 'content') and choice.message.content is not None:
                        raw_rm_outputs_content.append(choice.message.content.strip())
                    else:
                        logger.warning(f"{log_prefix_base} Choice {choice_idx} had no message content. Appending None.")
                        raw_rm_outputs_content.append(None)
                
                if hasattr(chat_completions, "usage") and chat_completions.usage:
                    logger.debug(
                        f"{log_prefix_base} Token usage (n={num_judgements_g}): "
                        f"input={chat_completions.usage.prompt_tokens}, output={chat_completions.usage.completion_tokens}"
                    )
            else:
                logger.warning(f"{log_prefix_base} chat_completion returned no choices. Will mark all as API error.")
                api_error_overall = True
        
        except Exception as e:
            logger.error(f"{log_prefix_base} LLM API (chat_completion) error for G judgements: {e}. User prompt snippet: {user_prompt_content_str[:200]}...", exc_info=True)
            api_error_overall = True

        while len(raw_rm_outputs_content) < num_judgements_g:
            raw_rm_outputs_content.append(None)

        for i, raw_content_this_judgement in enumerate(raw_rm_outputs_content):
            q_value = None
            thinking_block = None
            q_parse_error_flag = False
            tb_parse_error_flag = False
            current_judgement_api_error = api_error_overall or (raw_content_this_judgement is None and not api_error_overall)

            if not current_judgement_api_error and raw_content_this_judgement is not None:
                q_value = self._parse_q_value_from_response(raw_content_this_judgement)
                if q_value is None:
                    q_parse_error_flag = True
                    logger.warning(f"{log_prefix_base} Sample {i+1}: Failed to parse Q-value. Response snippet: {raw_content_this_judgement[:100]}...")

                if self.config.thinking:
                    thinking_block = self._extract_thinking_block(raw_content_this_judgement)
                    if thinking_block is None:
                        tb_parse_error_flag = True
                        logger.debug(f"{log_prefix_base} Sample {i+1}: Thinking block expected but not found. Snippet: {raw_content_this_judgement[:100]}...")
            else:
                q_parse_error_flag = True
                if self.config.thinking:
                    tb_parse_error_flag = True
            
            results.append(RMJudgementLog(
                game_seed_for_logging=game_seed_for_logging,
                turn_idx_for_logging=turn_idx_for_logging,
                policy_action_candidate_idx_for_logging=policy_action_candidate_idx_for_logging,
                rm_input_messages=list(messages_for_llm_call), 
                raw_rm_response_content=raw_content_this_judgement,
                parsed_q_value=q_value,
                parsed_thinking_block=thinking_block,
                api_error=current_judgement_api_error,
                q_value_parse_error=q_parse_error_flag,
                thinking_block_parse_error=tb_parse_error_flag and self.config.thinking
            ))
        return results

    def _create_error_judgement_log(
        self, 
        gs: Optional[int], ti: Optional[int], pai: Optional[int], 
        inp_msg: List[Dict[str, str]]
    ) -> RMJudgementLog:
        return RMJudgementLog(
            game_seed_for_logging=gs,
            turn_idx_for_logging=ti,
            policy_action_candidate_idx_for_logging=pai,
            rm_input_messages=inp_msg,
            raw_rm_response_content=None,
            parsed_q_value=None,
            parsed_thinking_block=None,
            api_error=True,
            q_value_parse_error=True,
            thinking_block_parse_error=self.config.thinking
        )
    
    def get_last_token_usage(self) -> Tuple[Optional[int], Optional[int]]:
        return self.last_input_token_count, self.last_output_token_count

# Example Usage (Conceptual)
async def example_rm_usage(server_client: ServerManager, tokenizer: PreTrainedTokenizer):
    rm_config = AtroposRMConfig(thinking=True, temperature=0.2, model_id="gpt-3.5-turbo")
    rm_agent = AtroposRM(server_client=server_client, tokenizer=tokenizer, config=rm_config)
    
    policy_agent_system_prompt = "You are a UTTT playing agent. Your goal is to win."
    current_game_observation = "Player X to move. Board: ... (detailed UTTT board) ..."
    policy_agent_response_content = (
        "<thinking>I should try to win microboard 3</thinking>"
        '<tool_call>{"arguments": {"micro_board": 3, "row": 1, "col": 1}, "name": "submit_move"}</tool_call>'
    )

    game_history_for_rm: List[Message] = [
        Message(role="system", content=policy_agent_system_prompt),
        Message(role="user", content=current_game_observation),
        Message(role="assistant", content=policy_agent_response_content)
    ]

    judgements = await rm_agent.generate_g_judgements(
        num_judgements_g=3,
        game_history_window=game_history_for_rm,
        game_seed_for_logging=123,
        turn_idx_for_logging=5,
        policy_action_candidate_idx_for_logging=1
    )

    for i, judgement_result in enumerate(judgements):
        logger.info(f"Judgement {i+1}: Q={judgement_result['parsed_q_value']}, API Error: {judgement_result['api_error']}")
