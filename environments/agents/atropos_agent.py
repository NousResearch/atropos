import logging
from typing import List, Optional, Tuple, Any
from atroposlib.type_definitions import Message
import numpy as np
from transformers import PreTrainedTokenizer 

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

# --- Sentence Embedding Helper --- 
class SentenceEmbeddingHelper:
    _instance = None
    _model = None
    _device = None
    _embedding_dim = 384 # Default for all-MiniLM-L6-v2

    def __new__(cls, *args, **kwargs):
        if not MEMORY_SYSTEM_PREREQUISITES_AVAILABLE: # Broader check now
            # Logger warning already issued at import time
            return None 

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
             # ... (as before)
            logger.error("SentenceEmbeddingHelper: Model not loaded or prerequisites unavailable. Cannot get embeddings.")
            return None
        # ... (rest of get_embeddings as before, using self._model and np)
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
# --- End Sentence Embedding Helper ---

class AtroposAgent:
    """
    An agent that interacts with an LLM for game playing, managing its own dialogue history.
    It uses a chat-based completion model currently (for compatibility with existing LLMs and servers).
    """
    def __init__(
        self,
        system_prompt: str,
        tokenizer: PreTrainedTokenizer, 
        temperature: float,
        max_context_token_length: int, 
        max_tokens_for_llm_output: int = 256, 
        player_id_for_logging: Optional[int] = None, 
        embedding_dim: int = 384, # Default, will be overridden if helper loads
        top_k_memories: int = 3, 
        memory_generation_system_prompt: Optional[str] = None
    ):
        self.system_prompt_content = system_prompt
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_context_token_length = max_context_token_length
        self.max_tokens_for_llm_output = max_tokens_for_llm_output
        self.player_id_for_logging = str(player_id_for_logging) if player_id_for_logging is not None else "Agent"
        self.current_game_messages: List[Message] = []

        # --- Memory System Initialization ---
        self.embedding_helper = None
        self.faiss_index = None
        self.embedding_dim = embedding_dim # Start with param, may be updated

        if MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            self.embedding_helper = SentenceEmbeddingHelper()
            if self.embedding_helper and self.embedding_helper._model is not None:
                self.embedding_dim = self.embedding_helper.embedding_dim # Use actual dim
                logger.info(f"AtroposAgent[{self.player_id_for_logging}] SentenceEmbeddingHelper active. Embedding dim set to {self.embedding_dim}.")
                try:
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                    logger.info(f"AtroposAgent[{self.player_id_for_logging}] FAISS index initialized (dim={self.embedding_dim}). Memory system enabled.")
                except Exception as e:
                    logger.error(f"AtroposAgent[{self.player_id_for_logging}] Failed to initialize FAISS index (dim={self.embedding_dim}): {e}. Memory system will be disabled.", exc_info=True)
                    self.faiss_index = None # Ensure it's None
                    self.embedding_helper = None # If FAISS fails, disable helper too for consistency
            else:
                logger.warning(
                    f"AtroposAgent[{self.player_id_for_logging}] SentenceEmbeddingHelper instantiated but model not loaded. "
                    f"Memory system will be disabled."
                )
                self.embedding_helper = None # Ensure helper is None if its model isn't loaded
                self.faiss_index = None
        else:
            logger.info(
                f"AtroposAgent[{self.player_id_for_logging}] Memory system prerequisites not met. "
                f"Memory features will be disabled."
            )
            # self.embedding_helper and self.faiss_index are already None or will remain so
        
        self.top_k_memories = top_k_memories
        self.memory_texts: List[str] = []

        if memory_generation_system_prompt is None:
            self.memory_generation_system_prompt = (
                "You are a memory creation assistant. Based on the provided game history window, "
                "which includes an agent\'s thoughts, actions, and the resulting observations, "
                "create a concise memory summary. Focus on:\\n"
                "1. The core of the agent\'s plan or intention during that turn.\\n"
                "2. Key learnings or important observations from the outcome of the agent\'s action.\\n"
                "3. Any critical changes to the agent\'s state (e.g., new inventory items, "
                "significant score changes, new locations discovered that seemed important).\\n"
                "Provide only the summarized memory text. Do not include any other conversational filler or explanation."
            )
        else:
            self.memory_generation_system_prompt = memory_generation_system_prompt
        logger.debug(f"AtroposAgent[{self.player_id_for_logging}] Memory generation system prompt set.")

    def start_new_game_dialogue(self) -> None:
        """Clears the current dialogue history and starts with the system prompt."""
        self.current_game_messages = [
            {"role": "system", "content": self.system_prompt_content}
        ]
        # Clear memories for the new game
        if self.faiss_index is not None:
            self.faiss_index.reset()
        self.memory_texts = []
        logger.info(f"AtroposAgent[{self.player_id_for_logging}] Memories cleared for new game.")

    def get_final_game_dialogue(self) -> List[Message]:
        """Returns the complete dialogue history for the finished game."""
        return list(self.current_game_messages) # Return a copy

    async def _get_embedding(self, text: str, server_client: Any = None) -> Optional[np.ndarray]: # server_client is no longer used here
        """Helper method to get an embedding for a given text string using local SentenceEmbeddingHelper."""
        if not self.embedding_helper or self.embedding_helper._model is None:
            logger.warning(f"AtroposAgent[{self.player_id_for_logging}] _get_embedding: SentenceEmbeddingHelper not available. Cannot get embedding.")
            return None
        if not text:
            logger.warning(f"AtroposAgent[{self.player_id_for_logging}] _get_embedding: Empty text provided.")
            # Return None, or an empty array of the correct shape if that's more useful downstream
            return None 
        
        embeddings_batch = self.embedding_helper.get_embeddings([text])
        if embeddings_batch is not None and embeddings_batch.shape[0] == 1:
            return embeddings_batch[0].reshape(1, -1) # Ensure (1, dim) shape
        elif embeddings_batch is not None: # Should not happen if only one text is passed
             logger.error(f"AtroposAgent[{self.player_id_for_logging}] _get_embedding: Expected 1 embedding, got {embeddings_batch.shape[0]}.")
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
        game_history_window: List[Message], # Window of game history for summarization
        server_client: Any,
    ) -> Tuple[Optional[str], bool]: # Returns (generated_memory_text, error_occurred)
        """
        Generates a memory by summarizing the provided game history window using an LLM,
        then embeds and stores this memory in FAISS.

        Args:
            game_history_window: A list of Message dicts representing the game context to summarize.
            server_client: The API client for LLM and embedding calls.

        Returns:
            A tuple: (generated_memory_text, error_occurred_flag)
        """
        if self.faiss_index is None:
            logger.error(f"AtroposAgent[{self.player_id_for_logging}] _generate_memory: FAISS index not initialized. Skipping memory generation.")
            return None, True
        if not game_history_window:
            logger.warning(f"AtroposAgent[{self.player_id_for_logging}] _generate_memory: Called with empty game_history_window.")
            return None, True # Or False if empty history is not an error for memory generation

        log_prefix = f"AtroposAgent[{self.player_id_for_logging}] MemoryGen:"

        user_content_for_memory_llm = self._format_history_for_memory_prompt(game_history_window)
        messages_for_memory_llm: List[Message] = [
            {"role": "system", "content": self.memory_generation_system_prompt},
            {"role": "user", "content": user_content_for_memory_llm}
        ]

        # Use a generic LLM sampling method. _agent_sample_llm_response can be used if its
        # error handling (e.g. returning empty tool call) is acceptable or if we adapt it.
        # For now, let's assume _agent_sample_llm_response is suitable.
        # The max_tokens_for_llm_output for agent actions might be different from desired memory length.
        # Using a potentially different max_tokens for memory summary if needed, or reusing existing.
        # For simplicity, reusing existing max_tokens_for_llm_output from agent config.
        
        logger.debug(f"{log_prefix} Calling LLM for memory summarization. History snippet: {user_content_for_memory_llm[:200]}...")
        generated_memory_text, llm_api_error = await self._agent_sample_llm_response(
            history_for_llm_call=messages_for_memory_llm,
            server_client=server_client
        )

        if llm_api_error or generated_memory_text is None:
            logger.error(f"{log_prefix} LLM call failed or returned None during memory generation.")
            return None, True
        
        # _agent_sample_llm_response might return "<tool_call>\n\n</tool_call>" on empty LLM response.
        # We should treat this as an empty/failed memory if it happens.
        if not generated_memory_text.strip() or generated_memory_text == "<tool_call>\n\n</tool_call>":
            logger.warning(f"{log_prefix} LLM produced an empty or placeholder summary: '{generated_memory_text}'. Not storing memory.")
            return None, False # Not an error, but no memory generated

        logger.info(f"{log_prefix} Successfully generated memory summary: '{generated_memory_text[:150]}...'")

        # Embed and store the memory
        memory_embedding = await self._get_embedding(generated_memory_text, server_client)

        if memory_embedding is None:
            logger.error(f"{log_prefix} Failed to get embedding for the generated memory. Memory not stored.")
            return generated_memory_text, True # Memory generated, but not stored (error in embedding)

        try:
            self.faiss_index.add(memory_embedding) # .astype(np.float32) already handled in _get_embedding return
            self.memory_texts.append(generated_memory_text)
            logger.info(f"{log_prefix} Memory embedded and stored successfully. Total memories: {self.faiss_index.ntotal}")
            return generated_memory_text, False
        except Exception as e:
            logger.error(f"{log_prefix} Error adding memory to FAISS index: {e}", exc_info=True)
            # Optionally, remove the text from memory_texts if add fails, or handle inconsistency.
            return generated_memory_text, True # Memory generated, embedding obtained, but FAISS add failed.

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
        Retrieves and prepends relevant memories to the observation if available.

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

        # --- Memory Retrieval ---
        augmented_observation_content = observation_content
        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            logger.debug(f"{log_prefix_base} Attempting to retrieve memories for observation.")
            current_obs_embedding = await self._get_embedding(observation_content, server_client)
            if current_obs_embedding is not None:
                try:
                    # Ensure we don't ask for more memories than available if faiss_index.ntotal < self.top_k_memories
                    k_to_retrieve = min(self.top_k_memories, self.faiss_index.ntotal)
                    if k_to_retrieve > 0:
                        distances, indices = self.faiss_index.search(current_obs_embedding, k_to_retrieve)
                        retrieved_memory_texts = []
                        for i in range(len(indices[0])):
                            idx = indices[0][i]
                            # FAISS can return -1 if not enough neighbors or on error for some index types
                            if idx != -1 and idx < len(self.memory_texts):
                                retrieved_memory_texts.append(self.memory_texts[idx])
                        
                        if retrieved_memory_texts:
                            formatted_memories = "\n".join([f"- {mem}" for mem in retrieved_memory_texts])
                            augmented_observation_content = (
                                f"Relevant Memories from Previous Turns:\n{formatted_memories}\n\n" 
                                f"Original Observation:\n{observation_content}"
                            )
                            logger.info(f"{log_prefix_base} Prepended {len(retrieved_memory_texts)} memories to observation.")
                        else:
                            logger.debug(f"{log_prefix_base} No valid memories retrieved from FAISS indices.")
                    else:
                        logger.debug(f"{log_prefix_base} Not enough memories in index to retrieve (k_to_retrieve=0).")

                except Exception as e:
                    logger.error(f"{log_prefix_base} Error during FAISS search: {e}", exc_info=True)
            else:
                logger.warning(f"{log_prefix_base} Could not get embedding for current observation. Skipping memory retrieval.")
        else:
            logger.debug(f"{log_prefix_base} No memories in FAISS index or index not available. Skipping retrieval.")
        # --- End Memory Retrieval ---

        user_message: Message = {"role": "user", "content": augmented_observation_content}
        self.current_game_messages.append(user_message)

        # Token length check before calling LLM
        prompt_str_for_token_check = self.tokenizer.apply_chat_template(
            self.current_game_messages, 
            tokenize=False,
            add_generation_prompt=True 
        )
        current_prompt_token_count = len(self.tokenizer.encode(prompt_str_for_token_check))
        
        safety_buffer_tokens = 64 

        if current_prompt_token_count + self.max_tokens_for_llm_output + safety_buffer_tokens > self.max_context_token_length:
            logger.warning(
                f"{log_prefix_base} Token count for prompt+output "
                f"({current_prompt_token_count + self.max_tokens_for_llm_output}) "
                f"approaching max context token length ({self.max_context_token_length}). Marking token limit exceeded."
            )
            return None, False, True 

        raw_llm_response_str, api_error = await self._agent_sample_llm_response(
            history_for_llm_call=self.current_game_messages, 
            server_client=server_client
        )

        if api_error:
            self.current_game_messages.append({"role": "assistant", "content": "<AGENT_LLM_API_ERROR>"})
            return None, True, False

        if raw_llm_response_str is None: 
             logger.error(f"{log_prefix_base} _agent_sample_llm_response returned None without API error flag. Treating as API error.")
             self.current_game_messages.append({"role": "assistant", "content": "<AGENT_LLM_UNEXPECTED_NONE_RESPONSE>"})
             return None, True, False 

        self.current_game_messages.append({"role": "assistant", "content": raw_llm_response_str})
        
        return raw_llm_response_str, False, False 