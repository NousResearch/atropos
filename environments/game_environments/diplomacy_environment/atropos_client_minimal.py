"""
Minimal AtroposClient - LLM Proxy for AI_Diplomacy Integration

This is a simplified version that:
- Intercepts LLM requests from AI_Diplomacy
- Forwards them to the Atropos policy server
- Collects trajectory data for training
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import httpx

sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))

from environments.game_environments.diplomacy_environment.AI_Diplomacy.ai_diplomacy.clients import (  # noqa: E402
    BaseModelClient,
)

logger = logging.getLogger(__name__)


class AtroposClientMinimal(BaseModelClient):
    """
    Minimal proxy client that forwards LLM requests to Atropos policy server.
    """

    def __init__(
        self,
        model_name: str,
        server_config: Dict,
    ):
        super().__init__(model_name)
        self.server_url = server_config.get("base_url", "http://localhost:8000")
        self.actual_model = server_config.get("model_name", "training-policy")
        self.client = httpx.AsyncClient(timeout=60.0)

        # Track interactions for trajectory collection
        self.interactions: List[Dict] = []
        self.current_power: Optional[str] = None
        self.current_phase: Optional[str] = None

        logger.info(
            f"Initialized AtroposClientMinimal for {model_name} -> {self.actual_model}"
        )

    async def generate_response(
        self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True
    ) -> str:
        """
        Forward prompt to Atropos server and return response.
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
            # Use chat completion API for better compatibility
            messages = [
                {
                    "role": "system",
                    "content": f"You are playing Diplomacy as {self.current_power}.",
                },
                {"role": "user", "content": prompt},
            ]

            response = await self.client.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "model": self.actual_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2000,
                },
            )
            response.raise_for_status()

            result = response.json()
            response_text = result["choices"][0]["message"]["content"]

            # Track interaction
            self.interactions.append(
                {
                    "power": self.current_power,
                    "phase": self.current_phase,
                    "task_type": task_type,
                    "prompt": prompt,
                    "response": response_text,
                }
            )

            return response_text

        except httpx.ConnectError:
            logger.error(f"Failed to connect to Atropos server at {self.server_url}")
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

    async def close(self):
        """Clean up resources."""
        await self.client.aclose()


def register_atropos_models(server_config: Dict):
    """
    Register AtroposClientMinimal with AI_Diplomacy's model loading system.
    """
    from ai_diplomacy import clients

    original_load = clients.load_model_client

    def load_model_client_with_atropos(model_id: str) -> BaseModelClient:
        if model_id.startswith("atropos-"):
            # Create our minimal proxy client
            logger.info(f"Creating AtroposClientMinimal for {model_id}")
            return AtroposClientMinimal(model_id, server_config)
        else:
            # Use original loader for other models
            return original_load(model_id)

    clients.load_model_client = load_model_client_with_atropos
    logger.info("Registered AtroposClientMinimal with AI_Diplomacy")


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
