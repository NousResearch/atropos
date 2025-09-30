"""

This is a queue-based proxy that:
- Intercepts LLM requests from AI_Diplomacy
- Puts them on a queue for the environment to process
- Waits for responses from the environment
- Returns responses to AI_Diplomacy
"""

import asyncio
import contextvars
import json
import logging
import os
import sys
import uuid
from typing import Dict, List, Optional

from environments.game_environments.diplomacy_environment.AI_Diplomacy.ai_diplomacy import (
    clients,
)

from environments.game_environments.diplomacy_environment.queue_manager import (
    PolicyRequest,
    QueueManager,
    get_queue_manager,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))

from environments.game_environments.diplomacy_environment.AI_Diplomacy.ai_diplomacy.clients import (  # noqa: E402
    BaseModelClient,
)

logger = logging.getLogger(__name__)

current_game_context = contextvars.ContextVar("current_game_id", default=None)
_game_interactions = {}


class AtroposClientMinimal(BaseModelClient):
    """
    Queue-based proxy client that forwards LLM requests through queues.
    """

    def __init__(
        self,
        model_name: str,
        queue_manager: Optional[QueueManager] = None,
    ):
        super().__init__(model_name)
        self.game_id = current_game_context.get()
        if not self.game_id:
            raise ValueError("AtroposClientMinimal created without game context set")

        self.queue_manager = queue_manager or get_queue_manager()

        self.interactions: List[Dict] = []
        self.current_power: Optional[str] = None
        self.current_phase: Optional[str] = None

        logger.info(
            f"Initialized AtroposClientMinimal for {model_name} in game {self.game_id}"
        )

    async def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Put request on queue and wait for response from environment.
        This is the main method AI_Diplomacy calls for all LLM interactions.
        """
        task_type = self._infer_task_type(prompt)
        power = self._extract_power(prompt)
        phase = self._extract_phase(prompt)

        if power:
            self.current_power = power
        if phase:
            self.current_phase = phase

        logger.debug(f"Generating response for {self.current_power}: {task_type}")

        try:
            request_id = str(uuid.uuid4())
            request = PolicyRequest(
                request_id=request_id,
                game_id=self.game_id,
                power=self.current_power or "UNKNOWN",
                phase=self.current_phase or "UNKNOWN",
                prompt=prompt,
                temperature=temperature,
                trajectory=self.interactions.copy(),
            )

            await self.queue_manager.put_request(self.game_id, request)
            logger.debug(f"Put request {request_id} on queue for game {self.game_id}")

            response = await self.queue_manager.get_response(self.game_id)

            if response.request_id != request_id:
                logger.warning(
                    f"Response ID mismatch: expected {request_id}, got {response.request_id}"
                )

            response_text = response.response

            # Track interaction
            interaction = {
                "power": self.current_power,
                "phase": self.current_phase,
                "task_type": task_type,
                "prompt": prompt,
                "response": response_text,
                "metadata": response.metadata,  # Store any additional info from environment
            }
            self.interactions.append(interaction)

            if self.game_id not in _game_interactions:
                _game_interactions[self.game_id] = []
            _game_interactions[self.game_id].append(interaction)

            return response_text

        except asyncio.TimeoutError:
            logger.error("Timeout waiting for response from environment")
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

        phase_match = re.search(r"[SF]\d{4}[MRB]", prompt)
        if phase_match:
            return phase_match.group()

        verbose_match = re.search(r"(Spring|Fall) \d{4}", prompt)
        if verbose_match:
            return verbose_match.group()

        return None

    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a simple fallback response if there's an issue."""
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


def get_game_interactions(game_id: str) -> List[Dict]:
    """Get all interactions for a specific game."""
    return _game_interactions.get(game_id, [])


def clear_game_interactions(game_id: str):
    """Clear interactions for a specific game."""
    if game_id in _game_interactions:
        del _game_interactions[game_id]


def register_atropos_models_globally(queue_manager: Optional[QueueManager] = None):
    """
    Register AtroposClientMinimal with AI_Diplomacy's model loading system globally.
    This should be called ONCE during environment setup.

    Args:
        queue_manager: Optional queue manager (uses global if not provided)
    """

    if hasattr(clients, "_atropos_registered"):
        logger.info("AtroposClientMinimal already registered globally")
        return

    clients._original_load_model_client = clients.load_model_client
    clients._atropos_queue_manager = queue_manager or get_queue_manager()

    def load_model_client_with_atropos(
        model_id: str, prompts_dir: Optional[str] = None
    ) -> BaseModelClient:
        if model_id.startswith("atropos-"):
            logger.info(f"Creating context-aware AtroposClientMinimal for {model_id}")
            return AtroposClientMinimal(model_id, clients._atropos_queue_manager)
        else:
            logger.info(f"Falling back to original loader for {model_id}")
            return clients._original_load_model_client(model_id, prompts_dir)

    clients.load_model_client = load_model_client_with_atropos
    clients._atropos_registered = True

    logger.info("Registered AtroposClientMinimal globally with AI_Diplomacy")


if __name__ == "__main__":

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
