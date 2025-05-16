import logging
from typing import List, Optional, Tuple, Any, Dict
from atroposlib.type_definitions import Message
import numpy as np
from transformers import PreTrainedTokenizer
from pydantic import BaseModel, Field

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

class AtroposAgentAction(BaseModel):
    action_text: str
    api_error: bool
    score: float

class AtroposAgentActionLog(BaseModel):
    turn_number: int
    alternatives: List[AtroposAgentAction]


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
        self.current_game_messages: List[Message] = []

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
        
        self.memory_texts: List[str] = []

        # Use memory_generation_system_prompt from config, or the default
        if not hasattr(self.config, 'memory_generation_system_prompt') or self.config.memory_generation_system_prompt is None:
            self.memory_generation_system_prompt_content = ( # Renamed variable for clarity
                "You are a memory creation assistant. Based on the provided game history window, "
                "which includes an agent's thoughts, actions, and the resulting observations, "
                "create a concise memory summary. Focus on:\\n"
                "1. The core of the agent's plan or intention during that turn.\\n"
                "2. Key learnings or important observations from the outcome of the agent's action.\\n"
                "3. Any critical changes to the agent's state (e.g., new inventory items, "
                "significant score changes, new locations discovered that seemed important).\\n"
                "Provide only the summarized memory text. Do not include any other conversational filler or explanation."
            )
        else:
            self.memory_generation_system_prompt_content = self.config.memory_generation_system_prompt # Corrected attribute name
        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Memory generation system prompt set.")

    def start_new_game_dialogue(self) -> None:
        """Clears the current dialogue history and starts with the system prompt."""
        self.current_game_messages = [
            Message(role="system", content=self.system_prompt_content)
        ]
        # Clear memories for the new game
        if self.faiss_index is not None:
            self.faiss_index.reset()
            logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] FAISS index reset for new game.")
        else: # Also log if memory was intended but not active
            if self.config.enable_memory:
                 logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] New game started, memory system was intended but not active (no FAISS index).")

        self.memory_texts = []
        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Memory texts cleared for new game.")

    def get_final_game_dialogue(self) -> List[Message]:
        """Returns the complete dialogue history for the finished game."""
        return list(self.current_game_messages) # Return a copy

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
        log_prompt_snippet = history_for_llm_call[-1]['content'][:200] if history_for_llm_call else "EMPTY_HISTORY"
        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] LLM Chat Prompt (last message content, first 200 chars): {log_prompt_snippet}")

        try:
            chat_completions = await self.server_client.chat_completion( # Use self.server_client
                messages=history_for_llm_call,
                n=n,
                max_tokens=self.config.max_tokens_for_llm_output, # Use from config
                temperature=self.config.temperature # Use from config
            )
            llm_generated_content = []
            # Error checks
            for choice in chat_completions.choices:
                if choice.message and \
                hasattr(choice.message, 'content') and \
                choice.message.content is not None:
                    llm_generated_content.append(AtroposAgentAction(action_text=choice.message.content.strip(), api_error=False, score=0.0))
                else:
                    llm_generated_content.append(AtroposAgentAction(action_text="", api_error=True, score=0.0))
                
                if not llm_generated_content:
                    logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] returned empty or None content from chat_completion. Last user message (start): {log_prompt_snippet}")
                    llm_generated_content.append(AtroposAgentAction(action_text="<tool_call>\n\n</tool_call>", api_error=False, score=-1.0)) # Track error for scoring
                
                logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] raw content output via chat_completion: '{llm_generated_content}'")
        except Exception as e:
            logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] LLM API (chat_completion) error: {e}. Last user message (start): {log_prompt_snippet}")
            llm_generated_content.append(AtroposAgentAction(action_text="<tool_call>\n\n</tool_call>", api_error=True, score=-1.0)) # API error occurred
        
        return llm_generated_content

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
            Message(role="system", content=self.memory_generation_system_prompt_content), # Use renamed var
            Message(role="user", content=formatted_history)
        ]
        
        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Generating memory with prompt: {memory_prompt_messages}")
        # Use a slightly higher temperature for memory generation to encourage diverse summaries? Or lower for factual?
        # For now, using agent's main temperature. Could be a separate config.
        original_temp = self.config.temperature # Store original temperature
        try:
            # Potentially use a different temperature for memory generation if desired
            # self.config.temperature = 0.5 # Example: more factual memory summary
            memory_text, error_occurred = await self._sample_llm_response(
                history_for_llm_call=memory_prompt_messages,
                # server_client=server_client # Handled by self.server_client
            )
        finally:
            pass # self.config.temperature = original_temp # Restore if changed

        if error_occurred or not memory_text:
            logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] _generate_memory: Failed to generate memory text from LLM.")
            return None, True
        
        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Generated memory: '{memory_text}'")
        return memory_text.strip(), False

    async def generate_action(
        self,
        observation_content: str, # The game-specific augmented observation from the environment
        game_history_window: List[Message], # Canonical history for the agent
        is_evaluation_context: bool = False, # Informs if this action is for evaluation (e.g., to disable learning/memory updates)
        n: int = 1,

    ) -> Tuple[Optional[str], List[Message]]: # Returns (action_text, updated_game_messages_with_action)
                                              # API error flag removed as it's handled internally / by server_client
        """
        Generates actions based on the current observation and dialogue history.
        Optionally retrieves and uses memories.
        Updates its internal dialogue history with the observation and its action.
        Generates a new memory if the system is enabled and not in an evaluation context.
        """
        
        # Step 1: Prepend memories if system is active
        final_observation_content = observation_content
        retrieved_memories_texts: List[str] = []

        if False and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Attempting to retrieve memories for observation.")
            observation_embedding = await self._get_embedding(observation_content)
            
            if observation_embedding is not None:
                try:
                    # Ensure query embedding is float32 and correctly shaped for FAISS
                    query_embedding_faiss = observation_embedding.astype(np.float32).reshape(1, -1)
                    
                    distances, indices = self.faiss_index.search(query_embedding_faiss, self.config.top_k_memories)
                    
                    # Filter out invalid indices (e.g., -1 if fewer than top_k memories exist)
                    # and map to actual memory texts
                    retrieved_memories_texts = [
                        self.memory_texts[i] for i in indices[0] if i != -1 and 0 <= i < len(self.memory_texts)
                    ]
                    
                    if retrieved_memories_texts:
                        memory_header = "--- Relevant Memories (from past experiences) ---"
                        formatted_memories = "\n".join([f"- {mem}" for mem in retrieved_memories_texts])
                        final_observation_content = f"{memory_header}\n{formatted_memories}\n--- Current Observation ---\n{observation_content}"
                        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Prepended {len(retrieved_memories_texts)} memories to observation.")
                        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Memories: {retrieved_memories_texts}")
                    else:
                        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] No relevant memories found or FAISS search returned no valid indices.")
                except Exception as e:
                    logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] Error during FAISS search or memory formatting: {e}", exc_info=True)
            else:
                logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] Could not get embedding for observation, skipping memory retrieval.")
        elif self.config.enable_memory and (self.faiss_index is None or self.faiss_index.ntotal == 0):
            logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Memory system enabled, but no memories in FAISS index yet or index not initialized.")


        # Step 2: Construct LLM prompt
        llm_call_history = list(game_history_window) # Ensure it's a copy
        llm_call_history.append(Message(role="user", content=final_observation_content))

        # TODO: Truncate here or env?

        # Step 3: Call LLM
        action_list, api_error = await self._sample_llm_response(
            history_for_llm_call=llm_call_history,
            n=n
        )
        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] LLM response: {action_list}")

        # Step 4: Update internal messages and prepare messages for RM/next turn
        # The history for the RM (and for the next turn's agent) should include the action taken.
        history_after_action = llm_call_history + [Message(role="assistant", content=action_list)]
        
        # self.current_game_messages is the agent's own canonical history.
        # This might be updated more carefully by an environment managing multiple perspectives.
        # For this standalone agent, we'll update it directly.
        self.current_game_messages = history_after_action


        # Step 5: Generate and store memory for this turn (action + observation outcome)
        # Only generate memory if not in evaluation context and system is enabled.
        if False and self.faiss_index is not None and not is_evaluation_context:
            # The "outcome" is implicitly the next observation, but we generate memory based on action taken in current context
            # For memory generation, we use the history *including* the action just taken.
            memory_context_for_generation = history_after_action # History including the system prompt, user obs, and assistant action
            
            generated_memory_text, mem_gen_error = await self._generate_memory(
                game_history_window=memory_context_for_generation,
                # server_client=server_client # Handled by self.server_client
            )
            if not mem_gen_error and generated_memory_text:
                memory_embedding = await self._get_embedding(generated_memory_text)
                if memory_embedding is not None:
                    try:
                        self.faiss_index.add(memory_embedding.astype(np.float32).reshape(1, -1))
                        self.memory_texts.append(generated_memory_text)
                        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Added new memory to FAISS. Total memories: {self.faiss_index.ntotal}")
                    except Exception as e:
                        logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] Error adding memory to FAISS: {e}", exc_info=True)
                else:
                    logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] Could not generate embedding for new memory text. Memory not added.")
            elif mem_gen_error:
                 logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] Error occurred during memory generation. Memory not added.")

        return action_list, history_after_action 