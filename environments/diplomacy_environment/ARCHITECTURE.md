# Diplomacy Environment Architecture - Fresh Design

## Overview

This document outlines a new architecture for integrating AI_Diplomacy with Atropos using the LLM Client Proxy approach.

## Core Design Principles

1. **Minimal Coupling**: Use AI_Diplomacy's existing infrastructure with minimal modifications
2. **Clean Interfaces**: Clear separation between Atropos RL logic and Diplomacy game mechanics
3. **Scalability**: Support for parallel game execution and distributed training
4. **Flexibility**: Easy to switch between different integration modes

## Architecture Components

### 1. AtroposClient (LLM Client Proxy)

```python
# environments/diplomacy_environment/atropos_client.py

from ai_diplomacy.base_model_client import BaseModelClient
import httpx
from typing import Optional, Dict, Any

class AtroposClient(BaseModelClient):
    """
    Proxy client that forwards LLM requests to Atropos policy server.
    Implements the AI_Diplomacy BaseModelClient interface.
    """
    
    def __init__(self, model_name: str, server_config: Dict[str, Any]):
        super().__init__(model_name)
        self.server_url = server_config.get("url", "http://localhost:8000")
        self.episode_id = None
        self.power = None
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def set_context(self, episode_id: str, power: str):
        """Set the current game context for this client."""
        self.episode_id = episode_id
        self.power = power
    
    async def generate_response(self, prompt: str, temperature: float = 0.0, 
                              inject_random_seed: bool = True) -> str:
        """
        Forward the prompt to Atropos server and return the response.
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
        
        response = await self.client.post(
            f"{self.server_url}/v1/completions",
            json=request_data
        )
        response.raise_for_status()
        
        return response.json()["text"]
    
    def _infer_task_type(self, prompt: str) -> str:
        """Infer the type of task from the prompt."""
        if "orders for this turn" in prompt.lower():
            return "orders"
        elif "conversation" in prompt.lower() or "message" in prompt.lower():
            return "negotiation"
        elif "plan" in prompt.lower() or "strategy" in prompt.lower():
            return "planning"
        else:
            return "general"
```

### 2. DiplomacyEnvironment (Main Entry Point)

```python
# environments/diplomacy_environment/diplomacy_env.py

from typing import Dict, List, Optional
import asyncio
from ai_diplomacy.lm_game import play_llm_game
from .atropos_client import AtroposClient
from .config import DiplomacyConfig

class DiplomacyEnvironment:
    """
    Main environment class that orchestrates Diplomacy games for Atropos.
    """
    
    def __init__(self, config: DiplomacyConfig):
        self.config = config
        self.clients = {}  # Map of model_name -> AtroposClient
        
    async def run_episode(self, episode_id: str) -> Dict:
        """
        Run a single Diplomacy game episode.
        """
        # Create clients for each power
        power_clients = {}
        for power in self.config.powers:
            client = self._get_or_create_client(power.model)
            client.set_context(episode_id, power.name)
            power_clients[power.name] = client
        
        # Run the game using AI_Diplomacy's infrastructure
        game_result = await play_llm_game(
            game_id=episode_id,
            model_clients=power_clients,
            max_turns=self.config.max_turns,
            deadline_seconds=self.config.deadline_seconds,
            # ... other game parameters
        )
        
        return self._format_episode_result(game_result)
    
    def _get_or_create_client(self, model_name: str) -> AtroposClient:
        """Get or create an AtroposClient for the given model."""
        if model_name not in self.clients:
            self.clients[model_name] = AtroposClient(
                model_name=model_name,
                server_config=self.config.server_config
            )
        return self.clients[model_name]
    
    def _format_episode_result(self, game_result: Dict) -> Dict:
        """Format the game result for Atropos consumption."""
        # Extract relevant data for training
        return {
            "episode_id": game_result["game_id"],
            "winner": game_result.get("winner"),
            "supply_centers": game_result.get("final_supply_centers"),
            "turns": len(game_result.get("phases", [])),
            "conversations": self._extract_conversations(game_result),
            "orders": self._extract_orders(game_result),
            # Add more fields as needed for training
        }
```

### 3. Integration with Atropos Server

```python
# environments/diplomacy_environment/server_integration.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

class CompletionRequest(BaseModel):
    prompt: str
    model: str
    temperature: float = 0.0
    episode_id: Optional[str] = None
    power: Optional[str] = None
    metadata: Dict[str, Any] = {}

class DiplomacyPolicyHandler:
    """
    Handles policy requests from AtroposClient instances.
    Integrates with Atropos's existing policy serving infrastructure.
    """
    
    def __init__(self, policy_server):
        self.policy_server = policy_server
    
    async def handle_completion(self, request: CompletionRequest) -> Dict[str, str]:
        """
        Process a completion request from a Diplomacy game.
        """
        # Convert Diplomacy prompt to Atropos format
        atropos_request = self._convert_to_atropos_format(request)
        
        # Get response from policy server
        response = await self.policy_server.generate(atropos_request)
        
        # Format response for Diplomacy
        return {"text": response["text"]}
    
    def _convert_to_atropos_format(self, request: CompletionRequest) -> Dict:
        """Convert Diplomacy request to Atropos format."""
        return {
            "prompt": request.prompt,
            "model": request.model,
            "temperature": request.temperature,
            "context": {
                "episode_id": request.episode_id,
                "agent_id": request.power,
                "task_type": request.metadata.get("task_type", "general")
            }
        }
```

### 4. Configuration

```yaml
# environments/diplomacy_environment/config.yaml

environment:
  name: "diplomacy"
  type: "multi_agent"

game_settings:
  max_turns: 20
  deadline_seconds: 300
  powers:
    - name: "AUSTRIA"
      model: "atropos-diplomacy-v1"
    - name: "ENGLAND"
      model: "atropos-diplomacy-v1"
    - name: "FRANCE"
      model: "atropos-diplomacy-v1"
    - name: "GERMANY"
      model: "atropos-diplomacy-v1"
    - name: "ITALY"
      model: "atropos-diplomacy-v1"
    - name: "RUSSIA"
      model: "atropos-diplomacy-v1"
    - name: "TURKEY"
      model: "atropos-diplomacy-v1"

server_config:
  url: "http://localhost:8000"
  timeout: 30

training:
  batch_size: 4  # Number of parallel games
  episodes_per_batch: 100
  save_replays: true
  replay_dir: "./replays"
```

## Data Flow

1. **Episode Start**:
   - Atropos creates a new episode
   - DiplomacyEnvironment initializes AtroposClient instances
   - Clients connect to Atropos policy server

2. **Game Loop**:
   - AI_Diplomacy requests actions via AtroposClient
   - AtroposClient forwards to Atropos server
   - Policy server returns actions
   - AI_Diplomacy executes game logic

3. **Episode End**:
   - Game results collected
   - Training data formatted and sent to Atropos
   - Episode marked complete

## Benefits of This Architecture

1. **Reuses AI_Diplomacy Infrastructure**: All game logic, UI, analysis tools work out of the box
2. **Clean Integration**: AtroposClient provides a simple bridge between systems
3. **Flexible Deployment**: Can run locally or distributed
4. **Easy Testing**: Can test with dummy clients or real policies
5. **Minimal Code**: Most complexity handled by existing systems

## Migration Path

To migrate from the current implementation:

1. Remove existing environment files (keeping this architecture doc)
2. Implement AtroposClient as the bridge
3. Create minimal DiplomacyEnvironment wrapper
4. Add server integration endpoints
5. Test with AI_Diplomacy's existing tools

## Future Extensions

1. **WebSocket Mode**: Add WebSocket client for distributed scenarios
2. **Human Play**: Allow human players via web interface
3. **Tournament Mode**: Run tournaments between different policies
4. **Analysis Integration**: Hook into AI_Diplomacy's betrayal/strategy analysis