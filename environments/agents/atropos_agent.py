import logging
from typing import List, Optional, Tuple, Any
from atroposlib.type_definitions import Message
import numpy as np
from transformers import PreTrainedTokenizer
from pydantic import BaseModel, Field
from environments.agents.atropos_agent_types import (
    AtroposAgentActionLog, 
    AtroposAgentAction
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Conditional imports for the memory system
MEMORY_SYSTEM_PREREQUISITES_AVAILABLE = False
try:
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss # For FAISS (vector similarity search)
    MEMORY_SYSTEM_PREREQUISITES_AVAILABLE = True
    logger.info("Memory system prerequisites (torch, sentence-transformers, faiss) found.")
except ImportError as e:
    logger.warning(
        f"Memory system prerequisites not fully met (torch, sentence-transformers, or faiss missing: {e}). "
        f"AtroposAgent memory features will be disabled."
    )
    torch = None
    SentenceTransformer = None
    faiss = None

class AtroposAgentConfig(BaseModel):
    """Configuration for AtroposAgent."""
    # General LLM parameters
    model_name: str = Field(default="gpt-4-turbo", description="LLM model name to use.")
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0,
        description="Sampling temperature for the LLM."
    )
    max_tokens_per_completion: int = Field(
        default=16384,
        description="Maximum tokens to generate for the agent's action or LLM response."
    )
    max_tokens_for_llm_output: int = Field(
        default=24576,
        description="Maximum tokens to generate for the LLM output."
    )
    max_retries_on_error: int = Field(default=3, description="Maximum retries for LLM calls on failure.")

    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="A general system prompt for the agent."
    )

    # Memory System Configuration
    enable_memory: bool = Field(
        default=True,
        description="Whether to enable the FAISS-based memory system. "
                    "Gracefully disables if prerequisites (faiss, sentence-transformers, torch) are missing."
    )
    embedding_dim: int = Field(
        default=384, # Default for all-MiniLM-L6-v2
        description="Dimension of embeddings. This will be updated by SentenceEmbeddingHelper upon its initialization "
                    "based on the actual model loaded by it, so this value mainly serves as a placeholder/default."
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
            "Output only the summary."
        ),
        description="System prompt for the LLM call to generate a memory summary."
    )
    
    player_id_for_logging: str = Field(
        default="AtroposAgent", # Changed from "Agent" to be more specific
        description="Identifier for the agent in logs."
    )

    class Config:
        extra = 'forbid' # Ensure no unexpected fields are passed
        arbitrary_types_allowed = True # Needed if FAISS index or model objects are stored in instances with this config


class SentenceEmbeddingHelper:
    """
    A singleton helper class for sentence embeddings using SentenceTransformer.
    It only loads the model once and reuses it for all subsequent embeddings.
    Uses a very small model (all-MiniLM-L6-v2) which is fast and has minimal memory 
    footprint. This isn't intended for a full RAG system, use a larger model for that
    if your environment requires it.
    """
    _instance = None
    _model = None
    _device = None
    _embedding_dim = 384 # Default for all-MiniLM-L6-v2

    def __new__(cls, *args, **kwargs):
        if not MEMORY_SYSTEM_PREREQUISITES_AVAILABLE: # Broader check now
            # Logger warning already issued at import time
            return None 

        # only load the model once (singleton helper)
        if cls._instance is None:
            cls._instance = super(SentenceEmbeddingHelper, cls).__new__(cls)
            try:
                cls._device = "cpu"
                logger.info(f"SentenceEmbeddingHelper: Using device: {cls._device}")
                
                # 22M parameters - small enough to run on CPU with minimal overhead
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                cls._model = SentenceTransformer(model_name, device=cls._device)
                dummy_embedding = cls._model.encode(["test"], device=cls._device)
                actual_dim = dummy_embedding.shape[1]
                if actual_dim != cls._embedding_dim:
                    logger.warning(
                        f"SentenceEmbeddingHelper: Expected embedding dimension {cls._embedding_dim} "
                        f"for {model_name}, but got {actual_dim}. Using {actual_dim}."
                    )
                    cls._embedding_dim = actual_dim
                logger.info(f"SentenceEmbeddingHelper: Model {model_name} loaded successfully. Embedding dim: {cls._embedding_dim}")

            except Exception as e:
                logger.error(f"SentenceEmbeddingHelper: Error loading SentenceTransformer model: {e}", exc_info=True)
                cls._instance = None
        return cls._instance

    def get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        if self._model is None or not MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            logger.error("SentenceEmbeddingHelper: Model not loaded or prerequisites unavailable. Cannot get embeddings.")
            return None
        if not texts:
            return np.array([]).reshape(0, self._embedding_dim)
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True, device=self.device, show_progress_bar=False)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"SentenceEmbeddingHelper: Error encoding texts: {e}", exc_info=True)
            return None

    @property
    def device(self):
        return self._device

    @property
    def embedding_dim(self):
        return self._embedding_dim

class AtroposAgent:
    """
    An agent that interacts with an LLM for game playing, managing its own dialogue history.
    It uses a chat-based completion model currently (for compatibility with existing LLMs and servers).
    """
    def __init__(
        self,
        server_client: Any, # LLM server client (e.g., APIServer instance)
        tokenizer: PreTrainedTokenizer, # Tokenizer from the environment/BaseEnv
        # max_context_token_length: int, # Now part of config, but usually from env. For now, agent doesn't directly use it for truncation
        config: Optional[AtroposAgentConfig] = None,
    ):
        self.config = config if config is not None else AtroposAgentConfig()
        self.server_client = server_client
        self.tokenizer = tokenizer
        self.system_prompt_content = self.config.system_prompt
        
        # Initialize structured game log
        self.game_log: AtroposAgentActionLog = AtroposAgentActionLog()
        
        # --- Memory System Initialization ---
        self.embedding_helper = None
        self.faiss_index = None
        # Use embedding_dim from config as initial, SentenceEmbeddingHelper might update it
        self.embedding_dim = self.config.embedding_dim 

        if self.config.enable_memory and MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            self.embedding_helper = SentenceEmbeddingHelper()
            if self.embedding_helper and self.embedding_helper._model is not None:
                self.embedding_dim = self.embedding_helper.embedding_dim # Use actual dim from helper
                logger.info(
                    f"AtroposAgent[{self.config.player_id_for_logging}] SentenceEmbeddingHelper active. "
                    f"Embedding dim set to {self.embedding_dim}."
                )
                try:
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                    logger.info(
                        f"AtroposAgent[{self.config.player_id_for_logging}] FAISS index initialized (dim={self.embedding_dim}). "
                        "Memory system enabled."
                    )
                except Exception as e:
                    logger.error(
                        f"AtroposAgent[{self.config.player_id_for_logging}] Failed to initialize FAISS index "
                        f"(dim={self.embedding_dim}): {e}. Memory system will be disabled.", exc_info=True
                    )
                    self.faiss_index = None 
                    self.embedding_helper = None # If FAISS fails, disable helper too
            else:
                logger.warning(
                    f"AtroposAgent[{self.config.player_id_for_logging}] SentenceEmbeddingHelper instantiated "
                    f"but model not loaded. Memory system will be disabled."
                )
                self.embedding_helper = None 
                self.faiss_index = None
        elif self.config.enable_memory and not MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
             logger.info(
                f"AtroposAgent[{self.config.player_id_for_logging}] Memory system configured to be enabled, "
                f"but prerequisites (torch, sentence-transformers, faiss) not met. "
                f"Memory features will be disabled."
            )
        else: # memory_system_enabled is False
            logger.info(
                f"AtroposAgent[{self.config.player_id_for_logging}] Memory system explicitly disabled in config."
            )
            # self.embedding_helper and self.faiss_index are already None or will remain so
        
        # stores CANONICAL memories only (not every alternative), for use with FAISS
        self.memory_texts: List[str] = []
        self.memory_generation_system_prompt = self.config.memory_generation_system_prompt
    
    def new_game(self) -> None:
        """Clears all game-specific logs and resets turn number."""
        self.game_log = AtroposAgentActionLog()
        
        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] New game started. All logs cleared, turn number reset.")

        # Memory system reset
        if self.faiss_index is not None:
            self.faiss_index.reset()
            logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] FAISS index reset for new game.")
        else: # Also log if memory was intended but not active
            if self.config.enable_memory:
                 logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] New game started, memory system was intended but not active (no FAISS index).")

        self.memory_texts = []
        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Memory texts cleared for new game.")

    def get_final_canonical_dialogue(self) -> List[Message]:
        """Reconstructs and returns the complete canonical dialogue history for the finished game."""
        return self._reconstruct_canonical_history()

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Helper method to get an embedding for a given text string using local SentenceEmbeddingHelper."""
        if not self.embedding_helper or self.embedding_helper._model is None:
            logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] _get_embedding: SentenceEmbeddingHelper not available. Cannot get embedding.")
            return None
        if not text:
            logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] _get_embedding: Empty text provided.")
            return None 
        
        embeddings_batch = self.embedding_helper.get_embeddings([text])
        if embeddings_batch is not None and embeddings_batch.shape[0] == 1:
            return embeddings_batch[0].reshape(1, -1) # Ensure (1, dim) shape
        elif embeddings_batch is not None: 
             logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] _get_embedding: Expected 1 embedding, got {embeddings_batch.shape[0]}.")
        return None

    def _format_history_for_memory_prompt(self, history_window: List[Message]) -> str:
        """Formats a list of message dicts into a single string for the memory generation prompt."""
        formatted_lines = []
        for msg in history_window:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "[No content]")
            formatted_lines.append(f"{role}: {content}")
        return "\n".join(formatted_lines)

    async def _sample_llm_response(
        self, 
        history_for_llm_call: List[Message],
        n: int = 1
    ) -> List[AtroposAgentAction]:
        """
        Internal method to sample a response from the LLM using chat completions.
        Returns a list of AtroposAgentAction objects.
        """
        log_prompt_snippet = (
            history_for_llm_call[-1]['content'][:200] 
            if history_for_llm_call and isinstance(history_for_llm_call[-1], dict) and history_for_llm_call[-1].get('content') 
            else "EMPTY_HISTORY_CONTENT"
        )
        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] LLM Chat Prompt (last message content, first 200 chars): {log_prompt_snippet}")

        llm_generated_actions: List[AtroposAgentAction] = []
        try:
            chat_completions = await self.server_client.chat_completion(
                messages=history_for_llm_call,
                n=n,
                max_tokens=self.config.max_tokens_per_completion, # Use per-completion token limit
                temperature=self.config.temperature
            )
            
            if chat_completions and chat_completions.choices:
                for choice in chat_completions.choices:
                    if choice.message and hasattr(choice.message, 'content') and choice.message.content is not None:
                        llm_generated_actions.append(
                            AtroposAgentAction(action_text=choice.message.content.strip(), api_error=False, score=0.0)
                        )
                    else:
                        logger.warning(
                            f"AtroposAgent[{self.config.player_id_for_logging}] LLM choice had no message content. Choice: {choice}"
                        )
                        llm_generated_actions.append(
                            AtroposAgentAction(action_text="", api_error=True, score=0.0)
                        )
            else:
                 logger.warning(
                    f"AtroposAgent[{self.config.player_id_for_logging}] chat_completion returned no choices or empty response. History snippet: {log_prompt_snippet}"
                )

        except Exception as e:
            logger.error(
                f"AtroposAgent[{self.config.player_id_for_logging}] LLM API (chat_completion) error: {e}. History snippet: {log_prompt_snippet}", 
                exc_info=True
            )
            # If n responses were requested, append n error actions
            for _ in range(n): 
                llm_generated_actions.append(
                    AtroposAgentAction(action_text="<ERROR_GENERATING_ACTION>", api_error=True, score=-1.0)
                )
        
        # Ensure we always return `n` actions, even if some are error placeholders
        while len(llm_generated_actions) < n:
            logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] LLM returned fewer actions than requested ({len(llm_generated_actions)} vs {n}). Appending error placeholders.")
            llm_generated_actions.append(
                AtroposAgentAction(action_text="<MISSING_ACTION_FROM_LLM>", api_error=True, score=-1.0)
            )
        
        # Log generated actions
        for i, action in enumerate(llm_generated_actions):
            logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Generated Action Candidate {i+1}/{n}: '{action.action_text[:100]}...' (Error: {action.api_error})")

        return llm_generated_actions[:n] # Return up to n actions

    async def _generate_memory(
        self,
        game_history_window: List[Message],
        # server_client: Any, # Already have self.server_client
    ) -> Tuple[Optional[str], bool]: # Returns (generated_memory_text, error_occurred)
        """
        Uses an LLM call to generate a concise memory string from a window of game history.
        Returns the memory string if successful, otherwise None. Also returns error flag.
        """
        if not self.config.enable_memory or self.embedding_helper is None or self.faiss_index is None:
            logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] _generate_memory: Memory system not active. Skipping memory generation.")
            return None, False # Not an error, just skipped

        formatted_history = self._format_history_for_memory_prompt(game_history_window)
        if not formatted_history:
            logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] _generate_memory: Formatted history for memory prompt is empty.")
            return None, False

        memory_prompt_messages = [
            Message(role="system", content=self.memory_generation_system_prompt), # Use renamed var
            Message(role="user", content=formatted_history)
        ]
        
        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Generating memory with prompt: {memory_prompt_messages}")
        # Use a slightly higher temperature for memory generation to encourage diverse summaries? Or lower for factual?
        # For now, using agent's main temperature. Could be a separate config.
        original_temp = self.config.temperature # Store original temperature
        try:
            # Potentially use a different temperature for memory generation if desired
            # self.config.temperature = 0.5 # Example: more factual memory summary
            
            # This part needs adjustment if _sample_llm_response is used, as it returns List[AtroposAgentAction]
            # For now, assuming memory generation is complex and will be refactored later with memory focus.
            # The original call was:
            # memory_text, error_occurred = await self._sample_llm_response(
            # history_for_llm_call=memory_prompt_messages,
            # )
            # This implies _sample_llm_response would need to be adapted or a different method used for single text generation.
            # For now, we keep the structure but acknowledge this will need a fix when memory is enabled.
            # A temporary placeholder for the call, knowing it's currently not active due to enable_memory checks / faiss_index.
            
            # Simulating the expected outcome for the existing logic flow if it were active:
            temp_actions = await self._sample_llm_response(history_for_llm_call=memory_prompt_messages, n=1)
            memory_text_candidate: Optional[str] = None
            error_occurred: bool = True
            if temp_actions:
                action_result = temp_actions[0]
                if not action_result.api_error and action_result.action_text:
                    memory_text_candidate = action_result.action_text.strip()
                    error_occurred = False
                else:
                    logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] _generate_memory: LLM call for memory resulted in error or empty text.")
            else:
                logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] _generate_memory: LLM call for memory returned no actions.")

        finally:
            pass # self.config.temperature = original_temp # Restore if changed

        if error_occurred or not memory_text_candidate: # Check against memory_text_candidate
            logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] _generate_memory: Failed to generate memory text from LLM.")
            return None, True
        
        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Generated memory: '{memory_text_candidate}'")
        return memory_text_candidate, False

    async def generate_action(
        self,
        observation_content: str, # The game-specific augmented observation from the environment
        is_evaluation_context: bool = False, # Informs if this action is for evaluation
        n: int = 1,
    ) -> List[AtroposAgentAction]:  # Returns list of action alternatives
        """
        Generates 'n' action alternatives based on the reconstructed canonical history.
        Stores the alternatives and the observation internally for later logging.
        Ignores memory system for now.
        """
        # Reconstruct history and append this observation
        history = self._reconstruct_canonical_history()
        observation_message = Message(role="user", content=observation_content)
        history.append(observation_message)
        # Sample action alternatives
        action_alternatives = await self._sample_llm_response(
            history_for_llm_call=history,
            n=n
        )
        # Append new turn without a selected alternative
        new_turn = AtroposAgentTurn(
            turn_number=len(self.game_log.turn) + 1,
            observation_message=observation_message,
            alternatives=action_alternatives,
            selected_alternative=None
        )
        self.game_log.turn.append(new_turn)
        return action_alternatives

    def record_selected_action(
        self,
        selected_action_index: int
    ) -> None:
        """
        Records the selected action for the most recent generate_action call.
        Creates a turn entry with the stored observation and alternatives.
        """
        # Update selected_alternative on the most recent turn
        if not self.game_log.turn:
            logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] No turns to select from.")
            return
        last_turn = self.game_log.turn[-1]
        if not (0 <= selected_action_index < len(last_turn.alternatives)):
            logger.error(
                f"AtroposAgent[{self.config.player_id_for_logging}] Invalid selected_action_index {selected_action_index} "
                f"for {len(last_turn.alternatives)} alternatives. Cannot record turn."
            )
            return
        last_turn.selected_alternative = selected_action_index
        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Selected action {selected_action_index} for turn {last_turn.turn_number}.")

    def _reconstruct_canonical_history(self) -> List[Message]:
        """
        Reconstructs the full conversation history including system prompt, all observations and selected responses.
        """
        history: List[Message] = []
        # System prompt
        if self.system_prompt_content:
            history.append(Message(role="system", content=self.system_prompt_content))
        # Each past turn
        for turn in self.game_log.turn:
            history.append(turn.observation_message)
            if turn.selected_alternative is not None:
                choice = turn.alternatives[turn.selected_alternative]
                history.append(Message(role="assistant", content=choice.action_text))
        return history