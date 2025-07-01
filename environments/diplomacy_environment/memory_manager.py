"""
Memory management for Diplomacy agents.

Implements episodic memory storage and retrieval using FAISS
for similarity-based memory recall.
"""

import json
import logging
import os
from typing import List, Optional, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .diplomacy_types import DiplomacyMemory

logger = logging.getLogger(__name__)


class DiplomacyMemoryManager:
    """
    Manages episodic memories for a Diplomacy agent.
    
    Uses sentence embeddings and FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        power: str,
        episode_id: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_memories: int = 1000,
    ):
        self.power = power
        self.episode_id = episode_id
        self.max_memories = max_memories
        
        # Initialize embedding model
        try:
            self.encoder = SentenceTransformer(embedding_model)
            self.embedding_dim = embedding_dim
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.encoder = None
            self.embedding_dim = embedding_dim
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Store memories
        self.memories: List[DiplomacyMemory] = []
        self.memory_embeddings: List[np.ndarray] = []
    
    def add_memory(self, memory: DiplomacyMemory) -> None:
        """Add a memory to the manager."""
        # Generate embedding
        if self.encoder:
            try:
                # Create searchable text from memory
                search_text = self._create_search_text(memory)
                embedding = self.encoder.encode([search_text])[0]
                memory.embedding = embedding.tolist()
            except Exception as e:
                logger.warning(f"Failed to create embedding: {e}")
                # Use random embedding as fallback
                embedding = np.random.randn(self.embedding_dim).astype('float32')
                memory.embedding = embedding.tolist()
        else:
            # Use random embedding if no encoder
            embedding = np.random.randn(self.embedding_dim).astype('float32')
            memory.embedding = embedding.tolist()
        
        # Add to storage
        self.memories.append(memory)
        self.memory_embeddings.append(np.array(memory.embedding, dtype='float32'))
        
        # Add to FAISS index
        self.index.add(np.array([memory.embedding], dtype='float32'))
        
        # Prune if needed
        if len(self.memories) > self.max_memories:
            self._prune_memories()
    
    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> List[DiplomacyMemory]:
        """Retrieve relevant memories for a query."""
        if not self.memories:
            return []
        
        # Create query embedding
        if self.encoder:
            try:
                query_embedding = self.encoder.encode([query])[0]
            except Exception as e:
                logger.warning(f"Failed to encode query: {e}")
                # Return most recent memories as fallback
                return sorted(
                    self.memories,
                    key=lambda m: m.turn,
                    reverse=True
                )[:top_k]
        else:
            # Return most recent memories if no encoder
            return sorted(
                self.memories,
                key=lambda m: m.turn,
                reverse=True
            )[:top_k]
        
        # Search FAISS index
        query_vec = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_vec, min(top_k * 2, len(self.memories)))
        
        # Filter by importance and get top k
        retrieved = []
        for idx in indices[0]:
            if idx < len(self.memories):
                memory = self.memories[idx]
                if memory.importance >= min_importance:
                    retrieved.append(memory)
                if len(retrieved) >= top_k:
                    break
        
        return retrieved
    
    def get_memories_by_turn_range(
        self,
        start_turn: int,
        end_turn: int,
    ) -> List[DiplomacyMemory]:
        """Get memories within a turn range."""
        return [
            m for m in self.memories
            if start_turn <= m.turn <= end_turn
        ]
    
    def get_memories_by_phase(self, phase: str) -> List[DiplomacyMemory]:
        """Get memories from a specific phase type."""
        return [m for m in self.memories if m.phase == phase]
    
    def update_memory_importance(
        self,
        memory_id: str,
        new_importance: float,
    ) -> None:
        """Update the importance score of a memory."""
        for memory in self.memories:
            if memory.memory_id == memory_id:
                memory.importance = new_importance
                break
    
    def consolidate_memories(
        self,
        turn_range: Optional[int] = 10,
    ) -> DiplomacyMemory:
        """
        Consolidate recent memories into a summary memory.
        
        This is useful for long games to compress older memories.
        """
        if not self.memories:
            return None
        
        # Get recent memories
        recent_memories = sorted(
            self.memories,
            key=lambda m: m.turn,
            reverse=True
        )[:turn_range]
        
        # Create consolidated summary
        summaries = [m.summary for m in recent_memories]
        consolidated_summary = f"Consolidated memory of turns {recent_memories[-1].turn}-{recent_memories[0].turn}: "
        consolidated_summary += " ".join(summaries[:3]) + "..."
        
        # Aggregate details
        consolidated_details = {
            "num_memories": len(recent_memories),
            "turn_range": (recent_memories[-1].turn, recent_memories[0].turn),
            "key_events": [m.details.get("key_event") for m in recent_memories if m.details.get("key_event")],
        }
        
        # Create consolidated memory with high importance
        consolidated = DiplomacyMemory(
            episode_id=self.episode_id,
            power=self.power,
            turn=recent_memories[0].turn,
            phase="consolidated",
            summary=consolidated_summary,
            details=consolidated_details,
            importance=0.8,  # High importance for consolidated memories
        )
        
        # Add to memories
        self.add_memory(consolidated)
        
        return consolidated
    
    def _create_search_text(self, memory: DiplomacyMemory) -> str:
        """Create searchable text representation of a memory."""
        parts = [
            f"Turn {memory.turn}",
            f"Phase: {memory.phase}",
            memory.summary,
        ]
        
        # Add key details
        if memory.details.get("agreements"):
            parts.append(f"Agreements: {memory.details['agreements']}")
        if memory.details.get("betrayals"):
            parts.append(f"Betrayals: {memory.details['betrayals']}")
        if memory.details.get("key_moves"):
            parts.append(f"Key moves: {memory.details['key_moves']}")
        
        return " ".join(parts)
    
    def _prune_memories(self) -> None:
        """Remove least important memories when over capacity."""
        # Sort by importance and recency
        sorted_memories = sorted(
            enumerate(self.memories),
            key=lambda x: (x[1].importance, x[1].turn),
        )
        
        # Remove lowest importance memories
        num_to_remove = len(self.memories) - self.max_memories
        indices_to_remove = [idx for idx, _ in sorted_memories[:num_to_remove]]
        
        # Remove from memories list (in reverse order to maintain indices)
        for idx in sorted(indices_to_remove, reverse=True):
            del self.memories[idx]
            del self.memory_embeddings[idx]
        
        # Rebuild FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if self.memory_embeddings:
            embeddings_array = np.vstack(self.memory_embeddings).astype('float32')
            self.index.add(embeddings_array)
    
    def save_to_file(self, filepath: str) -> None:
        """Save memories to a JSON file."""
        data = {
            "power": self.power,
            "episode_id": self.episode_id,
            "memories": [m.dict() for m in self.memories],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load memories from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Clear existing memories
        self.memories = []
        self.memory_embeddings = []
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Load memories
        for memory_data in data["memories"]:
            memory = DiplomacyMemory(**memory_data)
            
            # Restore embedding if available
            if memory.embedding:
                embedding = np.array(memory.embedding, dtype='float32')
            else:
                # Generate new embedding
                if self.encoder:
                    search_text = self._create_search_text(memory)
                    embedding = self.encoder.encode([search_text])[0]
                else:
                    embedding = np.random.randn(self.embedding_dim).astype('float32')
                memory.embedding = embedding.tolist()
            
            # Add to storage
            self.memories.append(memory)
            self.memory_embeddings.append(embedding)
            self.index.add(np.array([embedding], dtype='float32'))