"""
Minimal AtroposClient - LLM Proxy for AI_Diplomacy Integration

This is a queue-based proxy that:
- Intercepts LLM requests from AI_Diplomacy
- Puts them on a queue for the environment to process
- Waits for responses from the environment
- Returns responses to AI_Diplomacy
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Dict, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))

from environments.game_environments.diplomacy_environment.AI_Diplomacy.ai_diplomacy.clients import (  # noqa: E402
    BaseModelClient,
)

from queue_manager import (
    PolicyRequest,
    PolicyResponse,
    QueueManager,
    get_queue_manager,
)

logger = logging.getLogger(__name__)


class AtroposClientMinimal(BaseModelClient):
    """
    Queue-based proxy client that forwards LLM requests through queues.
    """

    def __init__(
        self,
        model_name: str,
        game_id: str,
        queue_manager: Optional[QueueManager] = None,
    ):
        super().__init__(model_name)
        self.game_id = game_id
        self.queue_manager = queue_manager or get_queue_manager()
        
        # Track interactions for trajectory collection
        self.interactions: List[Dict] = []
        self.current_power: Optional[str] = None
        self.current_phase: Optional[str] = None

        logger.info(
            f"Initialized AtroposClientMinimal for {model_name} in game {game_id}"
        )

    async def generate_response(
        self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True
    ) -> str:
        """
        Put request on queue and wait for response from environment.
        This is the main method AI_Diplomacy calls for all LLM interactions.
        """
        # Infer context from prompt
        task_type = self._infer_task_type(prompt)
        power = self._extract_power(prompt)
        phase = self._extract_phase(prompt)

        if power:
            self.current_power = power
        if phase:
            self.current_phase = phase

        logger.debug(f"Generating response for {self.current_power}: {task_type}")

        try:
            # Create policy request
            request_id = str(uuid.uuid4())
            request = PolicyRequest(
                request_id=request_id,
                game_id=self.game_id,
                power=self.current_power or "UNKNOWN",
                phase=self.current_phase or "UNKNOWN",
                prompt=prompt,
                temperature=temperature,
                trajectory=self.interactions.copy()  # Send current trajectory
            )
            
            # Put request on queue
            await self.queue_manager.put_request(self.game_id, request)
            logger.debug(f"Put request {request_id} on queue for game {self.game_id}")
            
            # Wait for response
            response = await self.queue_manager.get_response(self.game_id)
            
            # Verify response matches our request
            if response.request_id != request_id:
                logger.warning(f"Response ID mismatch: expected {request_id}, got {response.request_id}")
            
            response_text = response.response

            # Track interaction
            self.interactions.append(
                {
                    "power": self.current_power,
                    "phase": self.current_phase,
                    "task_type": task_type,
                    "prompt": prompt,
                    "response": response_text,
                    "metadata": response.metadata  # Store any additional info from environment
                }
            )

            return response_text

        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response from environment")
            return self._generate_fallback_response(prompt)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(prompt)

    def _infer_task_type(self, prompt: str) -> str:
        """Infer the type of task from the prompt."""
        prompt_lower = prompt.lower()

        if "orders" in prompt_lower or "submit" in prompt_lower:
            return "orders"
        elif "message" in prompt_lower or "negotiate" in prompt_lower:
            return "negotiation"
        elif "plan" in prompt_lower or "strategy" in prompt_lower:
            return "planning"
        else:
            return "general"

    def _extract_power(self, prompt: str) -> Optional[str]:
        """Extract power name from prompt if mentioned."""
        for power in [
            "AUSTRIA",
            "ENGLAND",
            "FRANCE",
            "GERMANY",
            "ITALY",
            "RUSSIA",
            "TURKEY",
        ]:
            if power in prompt.upper():
                return power
        return None

    def _extract_phase(self, prompt: str) -> Optional[str]:
        """Extract game phase from prompt if mentioned."""
        import re

        # Look for phase patterns like "Spring 1901" or "S1901M"
        phase_match = re.search(r"[SF]\d{4}[MRB]", prompt)
        if phase_match:
            return phase_match.group()

        verbose_match = re.search(r"(Spring|Fall) \d{4}", prompt)
        if verbose_match:
            return verbose_match.group()

        return None

    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a simple fallback response when server is unavailable."""
        task_type = self._infer_task_type(prompt)

        if task_type == "orders":
            return json.dumps(
                {
                    "orders": {},
                    "explanations": {"general": "Fallback - no server connected"},
                }
            )
        elif task_type == "negotiation":
            return json.dumps(
                {
                    "messages": [],
                    "explanations": {"general": "Fallback - no server connected"},
                }
            )
        else:
            return "Fallback response - server not available"

    def get_interactions(self) -> List[Dict]:
        """Get all tracked interactions for trajectory collection."""
        return self.interactions

    def clear_interactions(self):
        """Clear tracked interactions for a new game."""
        self.interactions = []
        self.current_power = None
        self.current_phase = None


def register_atropos_models(game_id: str, queue_manager: Optional[QueueManager] = None):
    """
    Register AtroposClientMinimal with AI_Diplomacy's model loading system for a specific game.

    Args:
        game_id: The game ID for this instance
        queue_manager: Optional queue manager (uses global if not provided)
    """
    from ai_diplomacy import clients

    # Save original function if not already saved
    if not hasattr(clients, '_original_load_model_client'):
        clients._original_load_model_client = clients.load_model_client

    # Store the client instances for this game
    game_clients = {}

    def load_model_client_with_atropos(
        model_id: str, prompts_dir: Optional[str] = None
    ) -> BaseModelClient:
        logger.info(f"load_model_client_with_atropos called with model_id: {model_id}")
        if model_id.startswith("atropos-"):
            # Create our queue-based proxy client
            logger.info(f"Creating AtroposClientMinimal for {model_id} in game {game_id}")
            client = AtroposClientMinimal(model_id, game_id, queue_manager)
            game_clients[model_id] = client
            return client
        else:
            # Use original loader for other models
            logger.info(f"Falling back to original loader for {model_id}")
            return clients._original_load_model_client(model_id, prompts_dir)

    clients.load_model_client = load_model_client_with_atropos
    logger.info(f"Registered AtroposClientMinimal with AI_Diplomacy for game {game_id}")
    
    return game_clients  # Return dict of created clients for this game


if __name__ == "__main__":
    # Simple test
    async def test_client():
        client = AtroposClientMinimal(
            "atropos-test",
            {"base_url": "http://localhost:8000", "model_name": "test-model"},
        )

        test_prompts = [
            "You are FRANCE. What are your orders for Spring 1901?",
            "Send a message to ENGLAND about cooperation.",
            "What is your strategic plan?",
        ]

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt[:50]}...")
            response = await client.generate_response(prompt)
            print(f"Response: {response[:100]}...")

        print(f"\nTracked {len(client.get_interactions())} interactions")
        await client.close()

    asyncio.run(test_client())
