import logging
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Conditional imports for the memory system
MEMORY_SYSTEM_PREREQUISITES_AVAILABLE = False
try:
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss

    MEMORY_SYSTEM_PREREQUISITES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Memory system prerequisites not available: {e}")
    torch = None
    SentenceTransformer = None
    faiss = None


class SentenceEmbeddingHelper:
    """Singleton helper class for sentence embeddings using SentenceTransformer."""

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
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                cls._model = SentenceTransformer(model_name, device=cls._device)

                # Determine actual embedding dimension
                dummy_embedding = cls._model.encode(["test"], device=cls._device)
                actual_dim = dummy_embedding.shape[1]
                if actual_dim != cls._embedding_dim:
                    cls._embedding_dim = actual_dim

            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model: {e}")
                cls._instance = None
        return cls._instance

    def get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for a list of texts."""
        if self._model is None or not MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            return None
        if not texts:
            return np.array([]).reshape(0, self._embedding_dim)
        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                device=self.device,
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return None

    @property
    def device(self):
        return self._device

    @property
    def embedding_dim(self):
        return self._embedding_dim


class AtroposMemoryManager:
    """Manages memory for a game-playing agent using sentence embeddings and FAISS."""

    def __init__(
        self,
        embedding_dim_config_val: int = 384,
        player_id_for_logging: str = "MemoryManager",
    ):
        self.player_id_for_logging = player_id_for_logging
        self.embedding_helper = None
        self.faiss_index = None
        self.memory_texts: List[str] = []
        self.embedding_dim = embedding_dim_config_val

        if MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            self.embedding_helper = SentenceEmbeddingHelper()
            if self.embedding_helper and self.embedding_helper._model is not None:
                self.embedding_dim = self.embedding_helper.embedding_dim
                try:
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                except Exception as e:
                    logger.error(f"Failed to initialize FAISS index: {e}")
                    self.faiss_index = None
                    self.embedding_helper = None

    @property
    def is_active(self) -> bool:
        """Return True if the memory system is initialized and usable."""
        return self.embedding_helper is not None and self.faiss_index is not None

    async def get_embedding_for_text(self, text: str) -> Optional[np.ndarray]:
        """Get a single embedding for a text string."""
        if not self.is_active or not text:
            return None

        embeddings_batch = self.embedding_helper.get_embeddings([text])
        if embeddings_batch is not None and embeddings_batch.shape[0] == 1:
            return embeddings_batch[0].reshape(1, -1)
        return None

    async def add_memory(self, memory_text: str) -> bool:
        """Add a new memory text to the store."""
        if not self.is_active or not memory_text:
            return False

        embedding = await self.get_embedding_for_text(memory_text)
        if embedding is None:
            return False

        try:
            self.faiss_index.add(embedding)
            self.memory_texts.append(memory_text)
            return True
        except Exception as e:
            logger.error(f"Error adding memory to FAISS index: {e}")
            return False

    async def retrieve_relevant_memories(
        self, query_text: str, k: int = 3
    ) -> List[str]:
        """Retrieve the top-k most relevant memories for a given query text."""
        if (
            not self.is_active
            or k <= 0
            or self.faiss_index.ntotal == 0
            or not query_text
        ):
            return []

        query_embedding = await self.get_embedding_for_text(query_text)
        if query_embedding is None:
            return []

        try:
            k_actual = min(k, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(query_embedding, k_actual)

            relevant_memories = []
            for idx in indices[0]:
                if 0 <= idx < len(self.memory_texts):
                    relevant_memories.append(self.memory_texts[idx])
            return relevant_memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    async def generate_memory_summary(
        self, observation: str, action: str, outcome: str
    ) -> str:
        """Generate a concise memory summary for storage."""
        summary = f"{observation.strip()[:200]}... -> {action.strip()} -> {outcome.strip()[:100]}"
        return summary

    def reset_memory(self) -> None:
        """Reset the memory system by clearing all stored memories."""
        if not self.is_active:
            return

        try:
            self.memory_texts.clear()
            self.faiss_index.reset()
        except Exception as e:
            logger.error(f"Error resetting memory: {e}")

    def get_current_memory_state(self) -> Tuple[List[str], int]:
        """Get current memory state for debugging."""
        if not self.is_active:
            return [], 0
        return self.memory_texts.copy(), self.faiss_index.ntotal
