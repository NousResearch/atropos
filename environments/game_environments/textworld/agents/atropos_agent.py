import logging
from typing import List, Optional, Tuple, Any, Dict

# Smol Gents imports
from smolagents.models import Model, ChatMessage, MessageRole

from atroposlib.type_definitions import Message, AtroposAgentActionLog, AtroposAgentAction, AtroposAgentTurn
import numpy as np
from transformers import PreTrainedTokenizer
from pydantic import BaseModel, Field
from atroposlib.envs.base import ServerManager

# Import the new memory manager
from environments.game_environments.textworld.agents.textworld_memory_manager import TextWorldMemoryManager, MEMORY_SYSTEM_PREREQUISITES_AVAILABLE

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Keep this commented or control via TextWorldEnv config

class AtroposAgentConfig(BaseModel):
    """Configuration for AtroposAgent."""
    # General LLM parameters
    model_id: str = Field(default="gpt-4-turbo", description="LLM model ID for smolagents.Model and server calls.")
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0,
        description="Sampling temperature for the LLM."
    )
    max_tokens_per_completion: int = Field(
        default=1024,
        description="Maximum tokens to generate for each action alternative."
    )
    max_retries_on_error: int = Field(default=3, description="Maximum retries for LLM calls on failure (not directly used by smol Model).")

    system_prompt: str = Field(
        default="You are a helpful AI assistant playing a text adventure game. Think step-by-step and then call the required tool.",
        description="A general system prompt for the agent."
    )

    # Memory System Configuration (related to how agent *uses* memory)
    enable_memory: bool = Field(
        default=True,
        description="Whether the agent should use its memory system. "
                    "Actual memory system functionality depends on TextWorldMemoryManager."
    )
    embedding_dim: int = Field(
        default=384, 
        description="Default dimension of embeddings for TextWorldMemoryManager. Will be overridden by actual model."
    )
    memory_top_k: int = Field(
        default=3,
        description="Number of relevant memories to retrieve and prepend to the observation."
    )
    memory_generation_system_prompt: str = Field(
        default=(
            "You are a helpful assistant. Based on the provided game history snippet, "
            "summarize the key events, insights, player actions, and outcomes concisely. "
            "This summary will be used as a memory for a game-playing agent. "
            "Focus on information that would be useful for making future decisions in the game. "
            "Output only the summary. Be brief and focus on the most important facts."
        ),
        description="System prompt for the LLM call to generate a memory summary."
    )
    max_tokens_for_memory_summary: int = Field(
        default=150,
        description="Maximum tokens to generate for a memory summary."
    )
    
    player_id_for_logging: str = Field(
        default="AtroposAgent",
        description="Identifier for the agent in logs."
    )

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True


def _convert_atropos_to_smolagent_messages(messages: List[Message]) -> List[Dict[str, str]]:
    """Converts a list of Atropos Message objects to smolagent message dicts."""
    smol_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            smol_role = "system"
        elif role == "user":
            smol_role = "user"
        elif role == "assistant" or role == "agent":
            smol_role = "assistant"
        elif role == "tool_call":
            smol_role = "assistant"
        elif role == "tool_response":
            smol_role = "tool"
        else:
            smol_role = "user"

        content_str = msg.get("content", "")
        if content_str is None: content_str = ""

        smol_messages.append({"role": smol_role, "content": content_str})
    return smol_messages

def _convert_smolagent_to_atropos_message(smol_message: ChatMessage) -> Message:
    """Converts a smolagent ChatMessage to an Atropos Message."""
    role_map = {
        MessageRole.SYSTEM: "system",
        MessageRole.USER: "user",
        MessageRole.ASSISTANT: "assistant",
        MessageRole.TOOL: "tool_response",
    }
    atropos_role = role_map.get(smol_message.role, "assistant")
    
    content = smol_message.content
    if isinstance(content, list):
        content = "\n".join(item["text"] for item in content if item.get("type") == "text")

    return Message(role=atropos_role, content=str(content) if content is not None else "", reward=None)


class AtroposAgent(Model):
    """
    An agent that interacts with an LLM for game playing, managing its own dialogue history
    and using a memory system. Implements the smolagents.Model interface.
    """
    def __init__(
        self,
        server_client: ServerManager,
        tokenizer: PreTrainedTokenizer,
        config: Optional[AtroposAgentConfig] = None,
        memory_manager: Optional[TextWorldMemoryManager] = None,
    ):
        self.config = config if config is not None else AtroposAgentConfig()
        super().__init__(model_id=self.config.model_id)

        self.server_client = server_client
        self.tokenizer = tokenizer
        self.system_prompt_content = self.config.system_prompt
        
        self.game_log: AtroposAgentActionLog = AtroposAgentActionLog(turn=[])
        
        if memory_manager:
            self.memory_manager = memory_manager
        elif self.config.enable_memory and MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Initializing default TextWorldMemoryManager.")
            self.memory_manager = TextWorldMemoryManager(
                embedding_dim_config_val=self.config.embedding_dim,
                player_id_for_logging=f"{self.config.player_id_for_logging}_Memory"
            )
        else:
            self.memory_manager = None
            if self.config.enable_memory and not MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
                logger.warning(
                    f"AtroposAgent[{self.config.player_id_for_logging}] Memory is enabled in config, but TextWorldMemoryManager "
                    f"could not be initialized due to missing prerequisites. Memory system will be OFF."
                )
            else:
                 logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Memory system is disabled or no manager provided.")

        self.memory_generation_system_prompt = self.config.memory_generation_system_prompt
        
        self.last_input_token_count: Optional[int] = None
        self.last_output_token_count: Optional[int] = None
    
    def new_game(self) -> None:
        """Clears all game-specific logs, resets turn number, and resets memory."""
        self.game_log = AtroposAgentActionLog(turn=[])
        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] New game started. All logs cleared, turn number reset.")

        if self.memory_manager and self.config.enable_memory:
            self.memory_manager.reset_memory()
            logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Memory manager reset for new game.")
        elif self.config.enable_memory:
            logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] New game: Memory enabled in config, but no active memory manager to reset.")

    def get_final_canonical_dialogue(self) -> List[Message]:
        """Reconstructs and returns the complete canonical dialogue history for the finished game."""
        return self._reconstruct_canonical_history()

    async def summarize_turn_for_memory(
        self,
        game_history_window: List[Message],
    ) -> Optional[str]: 
        """
        Uses the agent's LLM (via self.generate) to summarize a game history window for memory.
        """
        if not self.config.enable_memory or self.memory_manager is None or not self.memory_manager.is_active:
            logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] summarize_turn_for_memory: Memory system not active. Skipping summary generation.")
            return None

        if not game_history_window:
            logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] summarize_turn_for_memory: Game history window is empty.")
            return None

        formatted_history_for_smol = _convert_atropos_to_smolagent_messages(game_history_window)
        
        memory_prompt_messages_for_smol = [
            {"role": "system", "content": self.memory_generation_system_prompt}
        ]
        combined_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in formatted_history_for_smol])
        memory_prompt_messages_for_smol.append({"role": "user", "content": combined_history_text})
        
        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Generating memory summary with prompt (first 100 chars of user content): {combined_history_text[:100]}")
        
        try:
            chat_message_response = await self.generate(
                messages=memory_prompt_messages_for_smol,
                max_tokens=self.config.max_tokens_for_memory_summary,
                temperature=self.config.temperature
            )
            
            if chat_message_response and chat_message_response.content:
                summary_text = chat_message_response.content.strip()
                logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Generated memory summary: '{summary_text[:100]}...'")
                return summary_text
            else:
                logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] summarize_turn_for_memory: LLM call for summary resulted in empty content.")
                return None
        except Exception as e:
            logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] summarize_turn_for_memory: Error during LLM call: {e}", exc_info=True)
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
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens_per_completion),
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
        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] (smol.generate) LLM Chat Prompt (last message content, first 200 chars): {log_prompt_snippet}")

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
                    logger.warning(
                        f"AtroposAgent[{self.config.player_id_for_logging}] (smol.generate) LLM first choice had no message content. Choice: {first_choice}"
                    )
            else:
                 logger.warning(
                    f"AtroposAgent[{self.config.player_id_for_logging}] (smol.generate) chat_completion returned no choices or empty response. History snippet: {log_prompt_snippet}"
                )

            if hasattr(raw_response, "usage") and raw_response.usage:
                self.last_input_token_count = raw_response.usage.prompt_tokens
                self.last_output_token_count = raw_response.usage.completion_tokens
                logger.debug(
                    f"AtroposAgent[{self.config.player_id_for_logging}] (smol.generate) Token usage: input={self.last_input_token_count}, output={self.last_output_token_count}"
                )
            else:
                self.last_input_token_count = None
                self.last_output_token_count = None
            
            return ChatMessage(role=role, content=content, raw=raw_response)

        except Exception as e:
            logger.error(
                f"AtroposAgent[{self.config.player_id_for_logging}] (smol.generate) LLM API (chat_completion) error: {e}. History snippet: {log_prompt_snippet}", 
                exc_info=True
            )
            raise ValueError(f"LLM API error during smol.generate: {e}")


    async def generate_action(
        self,
        observation_content: str,
        n: int = 1,
    ) -> List[AtroposAgentAction]:
        current_history_atropos: List[Message] = self._reconstruct_canonical_history()
        
        retrieved_memories_str = ""
        if self.config.enable_memory and self.memory_manager and self.memory_manager.is_active:
            relevant_memories = await self.memory_manager.retrieve_relevant_memories(
                query_text=observation_content,
                k=self.config.memory_top_k
            )
            if relevant_memories:
                retrieved_memories_str = "\n".join([f"- {mem}" for mem in relevant_memories])
                logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Retrieved {len(relevant_memories)} memories for current observation.")
                observation_content = f"Relevant Memories:\n{retrieved_memories_str}\n\nOriginal Observation:\n{observation_content}"
            else:
                logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] No relevant memories found for current observation.")
        
        observation_message = Message(role="user", content=observation_content, reward=None)
        current_history_atropos.append(observation_message)

        history_for_llm_call = _convert_atropos_to_smolagent_messages(current_history_atropos)
        
        log_prompt_snippet = (
            history_for_llm_call[-1]['content'][:200] 
            if history_for_llm_call and isinstance(history_for_llm_call[-1], dict) and history_for_llm_call[-1].get('content') 
            else "EMPTY_HISTORY_CONTENT"
        )
        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] (generate_action) LLM Chat Prompt (last message content, first 200 chars): {log_prompt_snippet}")

        llm_generated_actions: List[AtroposAgentAction] = []
        
        try:
            chat_completions = await self.server_client.chat_completion(
                messages=history_for_llm_call,
                model=self.config.model_id,
                n=n,
                max_tokens=self.config.max_tokens_per_completion,
                temperature=self.config.temperature
            )
            
            if chat_completions and chat_completions.choices:
                for choice_idx, choice in enumerate(chat_completions.choices):
                    if choice.message and hasattr(choice.message, 'content') and choice.message.content is not None:
                        llm_generated_actions.append(
                            AtroposAgentAction(action_text=choice.message.content.strip(), api_error=False, score=0.0)
                        )
                    else:
                        logger.warning(
                            f"AtroposAgent[{self.config.player_id_for_logging}] (generate_action) LLM choice {choice_idx} had no message content. Choice: {choice}"
                        )
                        llm_generated_actions.append(
                            AtroposAgentAction(action_text="", api_error=True, score=0.0)
                        )
                if hasattr(chat_completions, "usage") and chat_completions.usage:
                    logger.debug(
                        f"AtroposAgent[{self.config.player_id_for_logging}] (generate_action) Token usage (n={n}): "
                        f"input={chat_completions.usage.prompt_tokens}, output={chat_completions.usage.completion_tokens}"
                    )

            else:
                 logger.warning(
                    f"AtroposAgent[{self.config.player_id_for_logging}] (generate_action) chat_completion returned no choices or empty response. History snippet: {log_prompt_snippet}"
                )

        except Exception as e:
            logger.error(
                f"AtroposAgent[{self.config.player_id_for_logging}] (generate_action) LLM API (chat_completion) error: {e}. History snippet: {log_prompt_snippet}", 
                exc_info=True
            )
            for _ in range(n): 
                llm_generated_actions.append(
                    AtroposAgentAction(action_text="<ERROR_GENERATING_ACTION>", api_error=True, score=-1.0)
                )
        
        while len(llm_generated_actions) < n:
            logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] (generate_action) LLM returned fewer actions than requested ({len(llm_generated_actions)} vs {n}). Appending error placeholders.")
            llm_generated_actions.append(
                AtroposAgentAction(action_text="<MISSING_ACTION_FROM_LLM>", api_error=True, score=-1.0)
            )
        
        for i, action in enumerate(llm_generated_actions):
            logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] (generate_action) Generated Action Candidate {i+1}/{n}: '{action['action_text'][:100]}...' (Error: {action['api_error']})")
        
        new_turn = AtroposAgentTurn(
            turn_number=len(self.game_log['turn']) + 1,
            observation_message=observation_message,
            alternatives=llm_generated_actions,
            selected_alternative=None
        )
        self.game_log['turn'].append(new_turn)
        
        return llm_generated_actions[:n]


    async def record_selected_action_and_learn_from_turn(
        self,
        selected_action_index: int
    ) -> None:
        if not self.game_log['turn']:
            logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] No turns to select from. Cannot record or learn.")
            return
        
        last_turn_data = self.game_log['turn'][-1]
        if not (0 <= selected_action_index < len(last_turn_data['alternatives'])):
            logger.error(
                f"AtroposAgent[{self.config.player_id_for_logging}] Invalid selected_action_index {selected_action_index} "
                f"for {len(last_turn_data['alternatives'])} alternatives. Cannot record."
            )
            return
        
        last_turn_data['selected_alternative'] = selected_action_index
        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Selected action {selected_action_index} for turn {last_turn_data['turn_number']}.")

        if self.config.enable_memory and self.memory_manager and self.memory_manager.is_active:
            turn_obs_message = last_turn_data['observation_message']
            selected_action_obj = last_turn_data['alternatives'][selected_action_index]
            
            if not selected_action_obj['api_error'] and selected_action_obj['action_text']:
                turn_assistant_message = Message(role="assistant", content=selected_action_obj['action_text'], reward=None)
                
                history_window_for_memory: List[Message] = []
                
                if self.system_prompt_content:
                     history_window_for_memory.append(Message(role="system", content=self.system_prompt_content, reward=None))
                history_window_for_memory.append(turn_obs_message)
                history_window_for_memory.append(turn_assistant_message)

                logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Attempting to summarize and store memory for turn {last_turn_data['turn_number']}.")
                
                memory_summary = await self.summarize_turn_for_memory(history_window_for_memory)
                
                if memory_summary:
                    await self.memory_manager.add_memory(memory_summary)
                else:
                    logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] Failed to generate memory summary for turn {last_turn_data['turn_number']}.")
            else:
                logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] Selected action was an error or empty. Skipping memory generation for turn {last_turn_data['turn_number']}.")
        elif self.config.enable_memory:
            logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Memory enabled but manager not active. Skipping memory storage for turn {last_turn_data['turn_number']}.")


    def _reconstruct_canonical_history(self) -> List[Message]:
        history: List[Message] = []
        if self.system_prompt_content:
            history.append(Message(role="system", content=self.system_prompt_content, reward=None))
        
        for turn_data in self.game_log['turn']:
            history.append(turn_data['observation_message'])
            if turn_data['selected_alternative'] is not None:
                selected_idx = turn_data['selected_alternative']
                if 0 <= selected_idx < len(turn_data['alternatives']):
                    choice = turn_data['alternatives'][selected_idx]
                    if not choice['api_error'] and choice['action_text']:
                        history.append(Message(role="assistant", content=choice['action_text'], reward=None))
                else:
                    logger.error(
                        f"AtroposAgent[{self.config.player_id_for_logging}] Invalid selected_alternative index {selected_idx} "
                        f"in turn {turn_data['turn_number']} during history reconstruction."
                    )
        return history

    def get_last_token_usage(self) -> Tuple[Optional[int], Optional[int]]:
        """Returns the last recorded input and output token counts."""
        return self.last_input_token_count, self.last_output_token_count