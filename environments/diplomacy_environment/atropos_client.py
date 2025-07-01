"""
AtroposClient - LLM Client Proxy for AI_Diplomacy Integration

This client implements the AI_Diplomacy BaseModelClient interface and forwards
all LLM requests to an Atropos policy server.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'AI_Diplomacy'))

from ai_diplomacy.clients import BaseModelClient
import httpx
import asyncio
import json
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AtroposClient(BaseModelClient):
    """
    Proxy client that forwards LLM requests to Atropos policy server.
    Implements the AI_Diplomacy BaseModelClient interface.
    """
    
    def __init__(self, model_name: str, server_url: str = "http://localhost:8000"):
        super().__init__(model_name)
        self.server_url = server_url
        self.episode_id: Optional[str] = None
        self.power: Optional[str] = None
        self.client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"Initialized AtroposClient for model {model_name} at {server_url}")
    
    def set_context(self, episode_id: str, power: str):
        """Set the current game context for this client."""
        self.episode_id = episode_id
        self.power = power
        logger.debug(f"Set context: episode={episode_id}, power={power}")
    
    async def generate_response(self, prompt: str, temperature: float = 0.0, 
                              inject_random_seed: bool = True) -> str:
        """
        Forward the prompt to Atropos server and return the response.
        
        This is the main method that AI_Diplomacy calls for all LLM interactions.
        """
        request_data = {
            "prompt": prompt,
            "model": self.model_name,
            "temperature": temperature,
            "episode_id": self.episode_id,
            "power": self.power,
            "metadata": {
                "task_type": self._infer_task_type(prompt),
                "inject_random_seed": inject_random_seed
            }
        }
        
        try:
            logger.debug(f"Sending request for {self.power}: task_type={request_data['metadata']['task_type']}")
            response = await self.client.post(
                f"{self.server_url}/v1/completions",
                json=request_data
            )
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Received response for {self.power}: {len(result.get('text', ''))} chars")
            return result["text"]
            
        except httpx.ConnectError:
            logger.error(f"Failed to connect to Atropos server at {self.server_url}")
            # Return a fallback response for development
            return self._generate_fallback_response(prompt)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _infer_task_type(self, prompt: str) -> str:
        """Infer the type of task from the prompt content."""
        prompt_lower = prompt.lower()
        
        if "orders for this turn" in prompt_lower or "submit orders" in prompt_lower:
            return "orders"
        elif "conversation" in prompt_lower or "message" in prompt_lower or "respond to" in prompt_lower:
            return "negotiation"
        elif "plan" in prompt_lower or "strategy" in prompt_lower or "goals" in prompt_lower:
            return "planning"
        elif "diary" in prompt_lower or "private thoughts" in prompt_lower:
            return "diary"
        else:
            return "general"
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate a simple fallback response for development/testing.
        This allows the system to run even without a connected Atropos server.
        """
        task_type = self._infer_task_type(prompt)
        
        if task_type == "orders":
            # Return empty orders (AI_Diplomacy will use defaults)
            return json.dumps({
                "orders": {},
                "explanations": {"general": "Fallback response - no server connected"}
            })
        elif task_type == "negotiation":
            return json.dumps({
                "messages": [],
                "explanations": {"general": "Fallback response - no server connected"}
            })
        elif task_type == "planning":
            return json.dumps({
                "plans": {"immediate": "Hold all positions"},
                "explanations": {"general": "Fallback response - no server connected"}
            })
        else:
            return "Fallback response - Atropos server not connected"
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()


class AtroposModelRegistry:
    """
    Registry for managing multiple AtroposClient instances.
    Allows different powers to use different models/configurations.
    """
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.clients: Dict[str, AtroposClient] = {}
    
    def get_client(self, model_name: str) -> AtroposClient:
        """Get or create a client for the specified model."""
        if model_name not in self.clients:
            self.clients[model_name] = AtroposClient(model_name, self.server_url)
        return self.clients[model_name]
    
    async def close_all(self):
        """Close all client connections."""
        for client in self.clients.values():
            await client.close()


# Integration with AI_Diplomacy's model loading system
def register_atropos_models():
    """
    Monkey-patch AI_Diplomacy's model loading to recognize Atropos models.
    This should be called before running any games.
    """
    from ai_diplomacy import clients
    
    original_load = clients.load_model_client
    
    def load_model_client_with_atropos(model_id: str) -> BaseModelClient:
        if model_id.startswith("atropos-"):
            # Use environment variable or default server URL
            server_url = os.environ.get("ATROPOS_SERVER_URL", "http://localhost:8000")
            return AtroposClient(model_id, server_url)
        else:
            return original_load(model_id)
    
    clients.load_model_client = load_model_client_with_atropos
    logger.info("Registered Atropos model loader")


if __name__ == "__main__":
    # Simple test of the client
    async def test_client():
        client = AtroposClient("atropos-test", "http://localhost:8000")
        client.set_context("test-episode", "FRANCE")
        
        # Test different prompt types
        test_prompts = [
            "What are your orders for this turn?",
            "Send a message to England about an alliance.",
            "What is your strategic plan for the next few turns?",
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt[:50]}...")
            try:
                response = await client.generate_response(prompt)
                print(f"Response: {response[:100]}...")
            except Exception as e:
                print(f"Error: {e}")
        
        await client.close()
    
    asyncio.run(test_client())