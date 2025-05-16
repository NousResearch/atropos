import logging
from typing import List, Optional, Tuple, Any, Dict
from atroposlib.type_definitions import Message
import numpy as np
from transformers import PreTrainedTokenizer
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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

# --- Start of AtroposAgentConfig Definition ---
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
        default=256,
        description="Maximum tokens to generate for the agent's action or LLM response."
    )
    max_retries_on_error: int = Field(default=3, description="Maximum retries for LLM calls on failure.")

    # Base system prompt (optional, can be a general persona)
    # system_prompt: str = Field(
    #     default="You are a helpful AI assistant.",
    #     description="A general system prompt for the agent."
    # )

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
    
    # Task-Specific Prompt for TextWorld Action Generation (primarily used by TextWorldEnv)
    action_generation_system_prompt: str = Field(
        default=(
            "You are an expert text adventure game player. You are playing a game in TextWorld. "
            "Based on the game history and your current situation (observation), "
            "determine the best single command to execute next to progress in the game and achieve the objective. "
            "Output ONLY this command and nothing else. Do not add any conversational fluff, explanation, or formatting. "
            "For example, if you decide to go north, output: go north"
        ),
        description="System prompt specifically for TextWorld action generation. "
                    "This is typically passed by the TextWorld environment to the agent's history."
    )
    
    player_id_for_logging: str = Field(
        default="AtroposAgent", # Changed from "Agent" to be more specific
        description="Identifier for the agent in logs."
    )

    class Config:
        extra = 'forbid' # Ensure no unexpected fields are passed
        arbitrary_types_allowed = True # Needed if FAISS index or model objects are stored in instances with this config

# --- End of AtroposAgentConfig Definition ---


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
                if torch.cuda.is_available():
                    cls._device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    cls._device = "mps"
                else:
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
        # self.max_context_token_length = max_context_token_length # Agent doesn't directly enforce this; relies on server/env for now

        # Use the action_generation_system_prompt as the default system prompt for this agent's behavior
        # if a more general 'system_prompt' field isn't specifically defined in the config.
        if hasattr(self.config, 'system_prompt') and self.config.system_prompt:
            self.system_prompt_content = self.config.system_prompt
        elif hasattr(self.config, 'action_generation_system_prompt') and self.config.action_generation_system_prompt:
            self.system_prompt_content = self.config.action_generation_system_prompt
            logger.info(f"AtroposAgent using 'action_generation_system_prompt' as main system prompt.")
        else:
            # Fallback if neither is defined, though AtroposAgentConfig should provide one.
            self.system_prompt_content = "You are a helpful AI assistant."
            logger.warning("AtroposAgent falling back to a generic system prompt as none specific was found in config.")

        self.current_game_messages: List[Message] = []

        # --- Memory System Initialization ---
        self.embedding_helper = None
        self.faiss_index = None
        # Use embedding_dim from config as initial, SentenceEmbeddingHelper might update it
        self.embedding_dim = self.config.embedding_dim 

        if self.config.memory_system_enabled and MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
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
        elif self.config.memory_system_enabled and not MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
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

        # Use memory_generation_prompt_template from config, or the default
        if self.config.memory_generation_prompt_template is None:
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
            self.memory_generation_system_prompt_content = self.config.memory_generation_prompt_template
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
            if self.config.memory_system_enabled:
                 logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] New game started, memory system was intended but not active (no FAISS index).")

        self.memory_texts = []
        logger.info(f"AtroposAgent[{self.config.player_id_for_logging}] Memory texts cleared for new game.")

    def get_final_game_dialogue(self) -> List[Message]:
        """Returns the complete dialogue history for the finished game."""
        return list(self.current_game_messages) # Return a copy

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]: # server_client removed
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

    async def _agent_sample_llm_response(
        self, 
        history_for_llm_call: List[Message],
        # server_client: Any # Already have self.server_client
    ) -> Tuple[Optional[str], bool]:
        """
        Internal method to sample a response from the LLM using chat completions.
        Returns a tuple: (llm_content_or_empty_tool_call, api_error_occurred_flag)
        """
        log_prompt_snippet = history_for_llm_call[-1]['content'][:200] if history_for_llm_call else "EMPTY_HISTORY"
        logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] LLM Chat Prompt (last message content, first 200 chars): {log_prompt_snippet}")

        try:
            chat_completions = await self.server_client.chat_completion( # Use self.server_client
                messages=history_for_llm_call,
                n=1,
                max_tokens=self.config.max_tokens_for_llm_output, # Use from config
                temperature=self.config.temperature # Use from config
            )
            
            llm_generated_content = ""
            if chat_completions.choices and \
               chat_completions.choices[0].message and \
               hasattr(chat_completions.choices[0].message, 'content') and \
               chat_completions.choices[0].message.content is not None:
                llm_generated_content = chat_completions.choices[0].message.content.strip()
            
            if not llm_generated_content:
                logger.warning(f"AtroposAgent[{self.config.player_id_for_logging}] returned empty or None content from chat_completion. Last user message (start): {log_prompt_snippet}")
                return "<tool_call>\n\n</tool_call>", False # Expected to be parsed as an error by the environment
            
            logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] raw content output via chat_completion: '{llm_generated_content}'")
            return llm_generated_content, False # No API error

        except Exception as e:
            logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] LLM API (chat_completion) error: {e}. Last user message (start): {log_prompt_snippet}")
            return None, True # API error occurred

    async def _generate_memory(
        self,
        game_history_window: List[Message],
        # server_client: Any, # Already have self.server_client
    ) -> Tuple[Optional[str], bool]: # Returns (generated_memory_text, error_occurred)
        """
        Uses an LLM call to generate a concise memory string from a window of game history.
        Returns the memory string if successful, otherwise None. Also returns error flag.
        """
        if not self.config.memory_system_enabled or self.embedding_helper is None or self.faiss_index is None:
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
            memory_text, error_occurred = await self._agent_sample_llm_response(
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
        game_history_window: List[Message], # Pass the relevant history window
        # server_client: Any, # Already have self.server_client
        # Context for logging from the environment (optional, passed through if provided)
        # These are not used by the agent logic itself but can be useful for richer logging if the calling env provides them
        game_seed_for_logging: Optional[int] = None,
        group_idx_for_logging: Optional[int] = None,
        turn_idx_for_logging: Optional[int] = None,
        is_evaluation_context: bool = False, # Informs if this action is for evaluation (e.g., to disable learning/memory updates)
        eval_episode_idx_for_logging: Optional[int] = None,

    ) -> Tuple[Optional[str], List[Message]]: # Returns (action_text, updated_game_messages_with_action)
                                              # API error flag removed as it's handled internally / by server_client
        """
        Generates an action based on the current observation and dialogue history.
        Optionally retrieves and uses memories.
        Updates its internal dialogue history with the observation and its action.
        Generates a new memory if the system is enabled and not in an evaluation context.
        """
        
        # Ensure a clean copy for this turn's processing if needed, though current_game_messages is usually the source of truth
        # For now, we assume game_history_window is the up-to-date history for this agent.
        # self.current_game_messages should be aligned by the calling environment if managing multiple agents/perspectives.
        # For a single agent scenario like in testing, game_history_window will be self.current_game_messages.
        
        # Ensure the passed game_history_window is what the agent should currently "know"
        # This might be self.current_game_messages or a subset/specific view from an environment
        # For the purpose of this agent, it uses this window to form its prompt.
        
        # Step 1: Prepend memories if system is active
        final_observation_content = observation_content
        retrieved_memories_texts: List[str] = []

        if self.config.memory_system_enabled and self.faiss_index is not None and self.faiss_index.ntotal > 0:
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
        elif self.config.memory_system_enabled and (self.faiss_index is None or self.faiss_index.ntotal == 0):
            logger.debug(f"AtroposAgent[{self.config.player_id_for_logging}] Memory system enabled, but no memories in FAISS index yet or index not initialized.")


        # Step 2: Construct LLM prompt
        # The game_history_window should be [system, user_obs1, assistant_action1, user_obs2, ...]
        # We append the (potentially augmented by memories) current observation as a new user message.
        
        # Make a mutable copy of the history window for this turn's LLM call.
        # This history should represent the state *before* the agent's current action.
        llm_call_history = list(game_history_window) # Ensure it's a copy
        
        # The last message in game_history_window should be the *previous* user observation/turn's end.
        # We are now about to act on `final_observation_content`.
        # If `game_history_window` already includes the current observation as its last user message,
        # we might be duplicating. Let's assume `game_history_window` is the history *up to* the current observation.
        # And `final_observation_content` is the *actual content* for the new user message.
        
        # Let's clarify: game_history_window should NOT contain the `observation_content`
        # that the agent is currently reacting to.
        # `observation_content` (potentially augmented to `final_observation_content`) is the NEW user message.

        # Add the current observation (potentially augmented with memories) to the history for the LLM
        llm_call_history.append(Message(role="user", content=final_observation_content))
        
        # Token length check/truncation should ideally happen here or in _agent_sample_llm_response
        # For now, relying on the server to handle if it's too long.

        # Step 3: Call LLM
        action_text, api_error = await self._agent_sample_llm_response(
            history_for_llm_call=llm_call_history,
            # server_client=server_client # Handled by self.server_client
        )

        if api_error or action_text is None: # If API error, action_text might be None from _agent_sample_llm_response
            logger.error(f"AtroposAgent[{self.config.player_id_for_logging}] Failed to get action from LLM due to API error.")
            # Return None or a specific error indicator, and the history *before* this failed attempt
            return "Error: LLM call failed.", llm_call_history # return original history before this action attempt

        # Step 4: Update internal messages and prepare messages for RM/next turn
        # The history for the RM (and for the next turn's agent) should include the action taken.
        history_after_action = llm_call_history + [Message(role="assistant", content=action_text)]
        
        # self.current_game_messages is the agent's own canonical history.
        # This might be updated more carefully by an environment managing multiple perspectives.
        # For this standalone agent, we'll update it directly.
        self.current_game_messages = history_after_action


        # Step 5: Generate and store memory for this turn (action + observation outcome)
        # Only generate memory if not in evaluation context and system is enabled.
        if self.config.memory_system_enabled and self.faiss_index is not None and not is_evaluation_context:
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

        return action_text, history_after_action 