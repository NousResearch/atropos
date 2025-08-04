"""
Queue Manager for Diplomacy Environment

Manages request/response queues between AtroposClient proxies and the environment.
Each game gets its own queue pair for isolation.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QueuePair:
    """A pair of queues for a single game."""
    game_id: str
    request_queue: asyncio.Queue
    response_queue: asyncio.Queue
    
    
@dataclass
class PolicyRequest:
    """Request from proxy to environment for policy sampling."""
    request_id: str
    game_id: str
    power: str
    phase: str
    prompt: str
    temperature: float
    trajectory: list  # List of previous interactions
    
    
@dataclass 
class PolicyResponse:
    """Response from environment back to proxy."""
    request_id: str
    response: str
    metadata: dict  # Can include scores, rewards, etc.


class QueueManager:
    """Manages queues for all parallel games."""
    
    def __init__(self):
        self.queue_pairs: Dict[str, QueuePair] = {}
        self._lock = asyncio.Lock()
        
    async def create_game_queues(self, game_id: str) -> QueuePair:
        """Create a new queue pair for a game."""
        async with self._lock:
            if game_id in self.queue_pairs:
                logger.warning(f"Queue pair already exists for game {game_id}")
                return self.queue_pairs[game_id]
                
            queue_pair = QueuePair(
                game_id=game_id,
                request_queue=asyncio.Queue(),
                response_queue=asyncio.Queue()
            )
            self.queue_pairs[game_id] = queue_pair
            logger.info(f"Created queue pair for game {game_id}")
            return queue_pair
            
    def get_queue_pair(self, game_id: str) -> Optional[QueuePair]:
        """Get queue pair for a game."""
        return self.queue_pairs.get(game_id)
        
    async def remove_game_queues(self, game_id: str):
        """Remove queues for a completed game."""
        async with self._lock:
            if game_id in self.queue_pairs:
                del self.queue_pairs[game_id]
                logger.info(f"Removed queue pair for game {game_id}")
                
    def get_all_request_queues(self) -> Dict[str, asyncio.Queue]:
        """Get all request queues for polling."""
        return {
            game_id: pair.request_queue 
            for game_id, pair in self.queue_pairs.items()
        }
        
    async def put_request(self, game_id: str, request: PolicyRequest):
        """Put a request on the appropriate queue."""
        queue_pair = self.get_queue_pair(game_id)
        if queue_pair:
            await queue_pair.request_queue.put(request)
        else:
            raise ValueError(f"No queue pair found for game {game_id}")
            
    async def get_response(self, game_id: str) -> PolicyResponse:
        """Get a response from the appropriate queue."""
        queue_pair = self.get_queue_pair(game_id)
        if queue_pair:
            return await queue_pair.response_queue.get()
        else:
            raise ValueError(f"No queue pair found for game {game_id}")
            
    async def put_response(self, game_id: str, response: PolicyResponse):
        """Put a response on the appropriate queue."""
        queue_pair = self.get_queue_pair(game_id)
        if queue_pair:
            await queue_pair.response_queue.put(response)
        else:
            raise ValueError(f"No queue pair found for game {game_id}")


# Global instance for easy access
_queue_manager = QueueManager()


def get_queue_manager() -> QueueManager:
    """Get the global queue manager instance."""
    return _queue_manager