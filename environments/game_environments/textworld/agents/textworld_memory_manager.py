import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Conditional imports for the memory system
MEMORY_SYSTEM_PREREQUISITES_AVAILABLE = False
try:
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss  # For FAISS (vector similarity search)
    MEMORY_SYSTEM_PREREQUISITES_AVAILABLE = True
    logger.info("Memory system prerequisites (torch, sentence-transformers, faiss) found for TextWorldMemoryManager.")
except ImportError as e:
    logger.warning(
        f"Memory system prerequisites not fully met for TextWorldMemoryManager (torch, sentence-transformers, or faiss missing: {e}). "
        f"Memory features will be disabled."
    )
    torch = None
    SentenceTransformer = None
    faiss = None


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
    _embedding_dim = 384  # Default for all-MiniLM-L6-v2

    def __new__(cls, *args, **kwargs):
        if not MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            return None

        if cls._instance is None:
            cls._instance = super(SentenceEmbeddingHelper, cls).__new__(cls)
            try:
                cls._device = "cpu"
                logger.info(f"SentenceEmbeddingHelper: Using device: {cls._device}")
                
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                cls._model = SentenceTransformer(model_name, device=cls._device)
                # Determine actual embedding dimension from the loaded model
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
                cls._instance = None # Ensure instance is None if model loading fails
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


class TextWorldMemoryManager:
    """
    Manages the memory for a TextWorld agent using sentence embeddings and FAISS.
    """
    def __init__(self, embedding_dim_config_val: int = 384, player_id_for_logging: str = "MemoryManager"):
        self.player_id_for_logging = player_id_for_logging
        self.embedding_helper = None
        self.faiss_index = None
        self.memory_texts: List[str] = []
        self.embedding_dim = embedding_dim_config_val # Start with config, update if helper loads model

        if MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            self.embedding_helper = SentenceEmbeddingHelper()
            if self.embedding_helper and self.embedding_helper._model is not None:
                self.embedding_dim = self.embedding_helper.embedding_dim # Use actual dim from loaded model
                logger.info(
                    f"{self.player_id_for_logging}: SentenceEmbeddingHelper active. "
                    f"Embedding dim set to {self.embedding_dim}."
                )
                try:
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                    logger.info(
                        f"{self.player_id_for_logging}: FAISS index initialized (dim={self.embedding_dim}). "
                        "Memory system enabled."
                    )
                except Exception as e:
                    logger.error(
                        f"{self.player_id_for_logging}: Failed to initialize FAISS index "
                        f"(dim={self.embedding_dim}): {e}. Memory system will be disabled.", exc_info=True
                    )
                    self.faiss_index = None
                    self.embedding_helper = None # Disable helper if FAISS fails
            else:
                logger.warning(
                    f"{self.player_id_for_logging}: SentenceEmbeddingHelper instantiated "
                    f"but model not loaded. Memory system will be disabled."
                )
                self.embedding_helper = None
                self.faiss_index = None # Ensure FAISS is also None
        else:
            logger.info(
                f"{self.player_id_for_logging}: Memory system prerequisites not met. "
                "Memory features will be disabled."
            )

    @property
    def is_active(self) -> bool:
        """Returns True if the memory system is initialized and usable."""
        return self.embedding_helper is not None and self.faiss_index is not None

    async def get_embedding_for_text(self, text: str) -> Optional[np.ndarray]:
        """Gets a single embedding for a text string."""
        if not self.is_active or not text:
            if not text:
                 logger.warning(f"{self.player_id_for_logging}:get_embedding_for_text: Empty text provided.")
            return None
        
        # Use the helper's get_embeddings which expects a list
        embeddings_batch = self.embedding_helper.get_embeddings([text])
        if embeddings_batch is not None and embeddings_batch.shape[0] == 1:
            return embeddings_batch[0].reshape(1, -1) # FAISS expects 2D array (1, dim)
        elif embeddings_batch is not None:
            logger.error(f"{self.player_id_for_logging}:get_embedding_for_text: Expected 1 embedding, got {embeddings_batch.shape[0]}.")
        return None

    async def add_memory(self, memory_text: str) -> bool:
        """
        Adds a new memory text to the store.
        Embeds the text and adds it to the FAISS index.
        Returns True if successful, False otherwise.
        """
        if not self.is_active:
            logger.warning(f"{self.player_id_for_logging}: Memory system not active. Cannot add memory: '{memory_text[:100]}...'")
            return False
        if not memory_text:
            logger.warning(f"{self.player_id_for_logging}: Attempted to add empty memory text.")
            return False

        embedding = await self.get_embedding_for_text(memory_text)
        if embedding is None:
            logger.error(f"{self.player_id_for_logging}: Failed to get embedding for memory: '{memory_text[:100]}...'")
            return False

        try:
            self.faiss_index.add(embedding)
            self.memory_texts.append(memory_text)
            logger.debug(f"{self.player_id_for_logging}: Added memory. Index size: {self.faiss_index.ntotal}. Memory: '{memory_text[:100]}...'")
            return True
        except Exception as e:
            logger.error(f"{self.player_id_for_logging}: Error adding memory to FAISS index: {e}", exc_info=True)
            return False

    async def retrieve_relevant_memories(self, query_text: str, k: int = 3) -> List[str]:
        """
        Retrieves the top-k most relevant memories for a given query text.
        Returns an empty list if memory system is not active, k is non-positive,
        no memories are stored, or an error occurs.
        """
        if not self.is_active:
            logger.warning(f"{self.player_id_for_logging}: Memory system not active. Cannot retrieve memories for query: '{query_text[:100]}...'")
            return []
        if k <= 0:
            logger.warning(f"{self.player_id_for_logging}: Requested k <= 0 memories. Returning empty list.")
            return []
        if self.faiss_index.ntotal == 0:
            logger.debug(f"{self.player_id_for_logging}: No memories in FAISS index to retrieve.")
            return []
        if not query_text:
            logger.warning(f"{self.player_id_for_logging}: Empty query text for memory retrieval.")
            return []


        query_embedding = await self.get_embedding_for_text(query_text)
        if query_embedding is None:
            logger.error(f"{self.player_id_for_logging}: Failed to get embedding for query: '{query_text[:100]}...'")
            return []

        try:
            # Adjust k if it's greater than the number of items in the index
            effective_k = min(k, self.faiss_index.ntotal)
            
            distances, indices = self.faiss_index.search(query_embedding, effective_k)
            
            retrieved_memories = []
            if indices.size > 0: # Check if any indices were returned
                for i in indices[0]: # indices is a 2D array, e.g., [[idx1, idx2]]
                    if 0 <= i < len(self.memory_texts):
                        retrieved_memories.append(self.memory_texts[i])
                    else:
                        logger.warning(f"{self.player_id_for_logging}: Invalid index {i} from FAISS search. Max index: {len(self.memory_texts) -1}.")
            
            logger.debug(f"{self.player_id_for_logging}: Retrieved {len(retrieved_memories)} memories for query: '{query_text[:100]}...'")
            return retrieved_memories
        except Exception as e:
            logger.error(f"{self.player_id_for_logging}: Error searching FAISS index: {e}", exc_info=True)
            return []

    def reset_memory(self) -> None:
        """Clears all stored memories and resets the FAISS index."""
        if self.is_active:
            self.faiss_index.reset()
            logger.info(f"{self.player_id_for_logging}: FAISS index reset.")
        elif MEMORY_SYSTEM_PREREQUISITES_AVAILABLE and self.embedding_helper: # Attempt to re-initialize if prerequisites were met
            logger.info(f"{self.player_id_for_logging}: FAISS index was not active. Attempting re-initialization during reset.")
            try:
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info(f"{self.player_id_for_logging}: FAISS index re-initialized during reset (dim={self.embedding_dim}).")
            except Exception as e:
                logger.error(
                    f"{self.player_id_for_logging}: Failed to re-initialize FAISS index "
                    f"(dim={self.embedding_dim}) during reset: {e}. Memory system remains disabled.", exc_info=True
                )
                self.faiss_index = None
                self.embedding_helper = None # Disable helper if FAISS init fails
        
        self.memory_texts = []
        logger.info(f"{self.player_id_for_logging}: Memory texts cleared.")

    def get_current_memory_state(self) -> Tuple[List[str], int]:
        """Returns all stored memory texts and the current count in FAISS index."""
        return list(self.memory_texts), self.faiss_index.ntotal if self.is_active else 0 