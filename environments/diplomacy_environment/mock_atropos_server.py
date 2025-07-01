#!/usr/bin/env python3
"""
Mock Atropos Server for Testing Diplomacy Integration

This demonstrates how the Atropos policy server would handle requests
from the AtroposClient during Diplomacy games.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import random
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock Atropos Policy Server")


class CompletionRequest(BaseModel):
    """Request format from AtroposClient."""
    prompt: str
    model: str
    temperature: float = 0.0
    episode_id: Optional[str] = None
    power: Optional[str] = None
    metadata: Dict[str, Any] = {}


class CompletionResponse(BaseModel):
    """Response format to AtroposClient."""
    text: str
    metadata: Optional[Dict[str, Any]] = None


class MockPolicyHandler:
    """
    Mock policy handler that generates simple but valid Diplomacy responses.
    In a real implementation, this would interface with the actual RL policy.
    """
    
    def __init__(self):
        self.episode_states = {}  # Track state per episode
        
    async def generate_response(self, request: CompletionRequest) -> str:
        """Generate a response based on the task type."""
        task_type = request.metadata.get("task_type", "general")
        
        logger.info(f"Generating {task_type} response for {request.power} in episode {request.episode_id}")
        
        if task_type == "orders":
            return self._generate_orders(request)
        elif task_type == "negotiation":
            return self._generate_negotiation(request)
        elif task_type == "planning":
            return self._generate_plan(request)
        elif task_type == "diary":
            return self._generate_diary(request)
        else:
            return self._generate_general(request)
    
    def _generate_orders(self, request: CompletionRequest) -> str:
        """Generate valid order responses."""
        # In a real implementation, this would:
        # 1. Parse the game state from the prompt
        # 2. Query the RL policy for action probabilities
        # 3. Sample or select best actions
        # 4. Format as valid Diplomacy orders
        
        # For now, return a simple defensive strategy
        power = request.power or "UNKNOWN"
        
        # Mock orders based on power (simplified)
        mock_orders = {
            "FRANCE": {
                "A PAR": "A PAR H",
                "A MAR": "A MAR - SPA",
                "F BRE": "F BRE - MAO"
            },
            "ENGLAND": {
                "F LON": "F LON - NTH",
                "F EDI": "F EDI - NWG", 
                "A LVP": "A LVP - YOR"
            },
            "GERMANY": {
                "A BER": "A BER - KIE",
                "A MUN": "A MUN - RUH",
                "F KIE": "F KIE - DEN"
            }
        }
        
        orders = mock_orders.get(power, {})
        
        response = {
            "orders": orders,
            "explanations": {
                "general": f"Mock policy: Conservative opening for {power}",
                "specific": {unit: "Defensive positioning" for unit in orders}
            }
        }
        
        return json.dumps(response, indent=2)
    
    def _generate_negotiation(self, request: CompletionRequest) -> str:
        """Generate negotiation messages."""
        power = request.power or "UNKNOWN"
        
        # Extract target from prompt (simplified)
        target = None
        if "to ENGLAND" in request.prompt:
            target = "ENGLAND"
        elif "to FRANCE" in request.prompt:
            target = "FRANCE"
        elif "to GERMANY" in request.prompt:
            target = "GERMANY"
        
        messages = []
        if target and random.random() > 0.3:  # 70% chance to send a message
            messages.append({
                "recipient": target,
                "message": f"Greetings from {power}. I propose we work together against our common threats. What are your thoughts on a mutual defense agreement?"
            })
        
        response = {
            "messages": messages,
            "explanations": {
                "general": f"Mock policy: Attempting diplomacy as {power}"
            }
        }
        
        return json.dumps(response, indent=2)
    
    def _generate_plan(self, request: CompletionRequest) -> str:
        """Generate strategic plans."""
        power = request.power or "UNKNOWN"
        
        response = {
            "plans": {
                "immediate": f"Secure home centers and establish defensive positions",
                "short_term": f"Expand cautiously while maintaining diplomatic flexibility",
                "long_term": f"Form stable alliances and work towards 18 supply centers"
            },
            "explanations": {
                "general": f"Mock policy: Balanced strategy for {power}"
            }
        }
        
        return json.dumps(response, indent=2)
    
    def _generate_diary(self, request: CompletionRequest) -> str:
        """Generate diary entries."""
        power = request.power or "UNKNOWN"
        
        return json.dumps({
            "diary_entry": f"Turn {datetime.now().strftime('%Y%m%d')}: As {power}, I must carefully balance expansion with diplomacy. The other powers seem unpredictable.",
            "trust_levels": {
                "ENGLAND": 0.5,
                "FRANCE": 0.5,
                "GERMANY": 0.5,
                "ITALY": 0.5,
                "AUSTRIA": 0.5,
                "RUSSIA": 0.5,
                "TURKEY": 0.5
            }
        })
    
    def _generate_general(self, request: CompletionRequest) -> str:
        """Generate general responses."""
        return f"Mock response from {request.model} for {request.power}: Acknowledged"


# Create global handler
policy_handler = MockPolicyHandler()


@app.post("/v1/completions", response_model=CompletionResponse)
async def generate_completion(request: CompletionRequest):
    """
    Main endpoint that AtroposClient calls.
    
    In a real implementation, this would:
    1. Validate the request
    2. Route to appropriate policy model
    3. Apply any safety/filtering
    4. Return formatted response
    """
    try:
        # Log request details
        logger.info(f"Received request: model={request.model}, episode={request.episode_id}, "
                   f"power={request.power}, task={request.metadata.get('task_type')}")
        
        # Generate response
        text = await policy_handler.generate_response(request)
        
        return CompletionResponse(
            text=text,
            metadata={
                "model_version": "mock-v1",
                "timestamp": datetime.utcnow().isoformat(),
                "episode_id": request.episode_id,
                "power": request.power
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "server": "mock-atropos-policy-server",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint with server info."""
    return {
        "server": "Mock Atropos Policy Server",
        "version": "1.0.0",
        "endpoints": {
            "/v1/completions": "Generate completions for Diplomacy actions",
            "/health": "Server health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the mock server
    logger.info("Starting Mock Atropos Policy Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)