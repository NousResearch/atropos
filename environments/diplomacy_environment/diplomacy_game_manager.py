"""
Game manager for interfacing with AI_Diplomacy engine.

Handles game initialization, state management, and order execution
through either subprocess or API communication.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import websockets

from .diplomacy_types import (
    GameState,
    DiplomacyPhase,
    Territory,
    Unit,
    Order,
    NegotiationMessage,
    DiplomacyEpisodeState,
)

logger = logging.getLogger(__name__)


class DiplomacyGameManager:
    """
    Manages the interface with AI_Diplomacy game engine.
    
    Supports both subprocess and API modes of operation.
    """
    
    def __init__(
        self,
        engine_path: str = "./AI_Diplomacy",
        use_submodule: bool = True,
        api_url: str = "http://localhost:8432",
        websocket_url: str = "ws://localhost:8433",
    ):
        self.engine_path = Path(engine_path)
        self.use_submodule = use_submodule
        self.api_url = api_url
        self.websocket_url = websocket_url
        
        # Process management for subprocess mode
        self.engine_process = None
        self.websocket_connection = None
        
        # Game tracking
        self.active_games: Dict[str, GameState] = {}
    
    async def initialize(self) -> None:
        """Initialize the game manager."""
        if self.use_submodule:
            await self._start_engine_subprocess()
        else:
            await self._verify_api_connection()
    
    async def shutdown(self) -> None:
        """Shutdown the game manager."""
        if self.engine_process:
            self.engine_process.terminate()
            await asyncio.sleep(1)
            if self.engine_process.poll() is None:
                self.engine_process.kill()
        
        if self.websocket_connection:
            await self.websocket_connection.close()
    
    async def _start_engine_subprocess(self) -> None:
        """Start AI_Diplomacy as a subprocess."""
        # Check if engine exists
        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"AI_Diplomacy not found at {self.engine_path}. "
                "Please run: git submodule add https://github.com/EveryInc/AI_Diplomacy.git"
            )
        
        # Start the engine
        cmd = [
            "python", "-m", "ai_diplomacy.server",
            "--port", "8432",
            "--websocket-port", "8433",
            "--headless", "true",
        ]
        
        self.engine_process = subprocess.Popen(
            cmd,
            cwd=self.engine_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for engine to start
        await self._wait_for_engine_ready()
        
        # Connect websocket
        await self._connect_websocket()
    
    async def _verify_api_connection(self) -> None:
        """Verify connection to external AI_Diplomacy API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health") as response:
                    if response.status != 200:
                        raise ConnectionError(f"AI_Diplomacy API not healthy: {response.status}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to AI_Diplomacy API: {e}")
        
        # Connect websocket
        await self._connect_websocket()
    
    async def _wait_for_engine_ready(self, timeout: int = 30) -> None:
        """Wait for engine to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.api_url}/health") as response:
                        if response.status == 200:
                            logger.info("AI_Diplomacy engine is ready")
                            return
            except:
                await asyncio.sleep(1)
        
        raise TimeoutError("AI_Diplomacy engine failed to start")
    
    async def _connect_websocket(self) -> None:
        """Connect to AI_Diplomacy websocket for real-time updates."""
        try:
            self.websocket_connection = await websockets.connect(self.websocket_url)
            logger.info("Connected to AI_Diplomacy websocket")
        except Exception as e:
            logger.warning(f"Failed to connect websocket: {e}")
    
    async def initialize_game(
        self,
        scenario: Dict[str, Any],
        power_assignment: Dict[str, str],
    ) -> GameState:
        """Initialize a new game."""
        # Create game through API
        game_config = {
            "variant": scenario.get("variant", "standard"),
            "starting_position": scenario.get("starting_position"),
            "power_assignment": power_assignment,
            "phase_minutes": 0,  # No time limits for RL training
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/games/create",
                json=game_config,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to create game: {await response.text()}")
                
                game_data = await response.json()
                game_id = game_data["game_id"]
        
        # Get initial game state
        game_state = await self.get_game_state(game_id)
        self.active_games[game_id] = game_state
        
        return game_state
    
    async def get_game_state(self, game_id: str) -> GameState:
        """Get current game state."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/games/{game_id}/state") as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to get game state: {await response.text()}")
                
                data = await response.json()
        
        # Parse into GameState
        game_state = GameState(
            game_id=game_id,
            variant=data["variant"],
            year=data["year"],
            phase=DiplomacyPhase(data["phase"]),
            territories=self._parse_territories(data["territories"]),
            units=self._parse_units(data["units"]),
            supply_centers=data["supply_centers"],
            current_orders={},
            messages_this_phase=[],
        )
        
        return game_state
    
    def _parse_territories(self, territory_data: Dict) -> Dict[str, Territory]:
        """Parse territory data from API."""
        territories = {}
        
        for name, data in territory_data.items():
            territory = Territory(
                name=name,
                abbreviation=data["abbreviation"],
                is_supply_center=data["is_supply_center"],
                is_land=data["type"] in ["land", "coast"],
                is_sea=data["type"] == "sea",
                adjacent=data["adjacent"],
                owner=data.get("owner"),
            )
            territories[name] = territory
        
        return territories
    
    def _parse_units(self, unit_data: List[Dict]) -> Dict[str, Unit]:
        """Parse unit data from API."""
        units = {}
        
        for data in unit_data:
            unit = Unit(
                unit_id=data["unit_id"],
                power=data["power"],
                unit_type=data["type"],
                location=data["location"],
                can_retreat_to=data.get("can_retreat_to", []),
            )
            units[unit.unit_id] = unit
        
        return units
    
    async def submit_messages(
        self,
        game_id: str,
        messages: List[NegotiationMessage],
    ) -> None:
        """Submit negotiation messages."""
        message_data = [
            {
                "from_power": msg.from_power,
                "to_powers": msg.to_powers,
                "content": msg.content,
                "message_type": msg.message_type,
            }
            for msg in messages
        ]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/games/{game_id}/messages",
                json={"messages": message_data},
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to submit messages: {await response.text()}")
    
    async def submit_orders(
        self,
        game_id: str,
        power: str,
        orders: List[Order],
    ) -> None:
        """Submit orders for a power."""
        order_data = [
            {
                "unit_id": order.unit_id,
                "order": order.to_string(),
            }
            for order in orders
        ]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/games/{game_id}/orders/{power}",
                json={"orders": order_data},
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to submit orders: {await response.text()}")
    
    async def execute_turn(self, episode_state: DiplomacyEpisodeState) -> Dict[str, Any]:
        """Execute the current turn and get results."""
        game_id = episode_state.game_state.game_id
        
        # Submit all orders from agents
        for power, agent in episode_state.agents.items():
            if power in episode_state.game_state.current_orders:
                await self.submit_orders(
                    game_id,
                    power,
                    episode_state.game_state.current_orders[power],
                )
        
        # Process turn
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_url}/games/{game_id}/process") as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to process turn: {await response.text()}")
                
                results = await response.json()
        
        # Update game state
        new_state = await self.get_game_state(game_id)
        episode_state.game_state = new_state
        self.active_games[game_id] = new_state
        
        # Parse results
        turn_results = {
            "year": new_state.year,
            "phase": new_state.phase.value,
            "order_results": results.get("order_results", {}),
            "retreats_required": results.get("retreats_required", {}),
            "builds_required": results.get("builds_required", {}),
            "supply_changes": results.get("supply_changes", {}),
            "eliminated_powers": results.get("eliminated_powers", []),
        }
        
        return turn_results
    
    async def get_legal_orders(
        self,
        game_id: str,
        power: str,
    ) -> Dict[str, List[str]]:
        """Get legal orders for a power's units."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.api_url}/games/{game_id}/legal_orders/{power}"
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to get legal orders: {await response.text()}")
                
                return await response.json()
    
    async def validate_orders(
        self,
        game_id: str,
        power: str,
        orders: List[Order],
    ) -> Tuple[bool, List[str]]:
        """Validate a set of orders."""
        order_strings = [order.to_string() for order in orders]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/games/{game_id}/validate_orders",
                json={
                    "power": power,
                    "orders": order_strings,
                },
            ) as response:
                if response.status != 200:
                    return False, ["API error"]
                
                result = await response.json()
                return result["valid"], result.get("errors", [])
    
    async def get_power_stats(
        self,
        game_id: str,
        power: str,
    ) -> Dict[str, Any]:
        """Get statistics for a power."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.api_url}/games/{game_id}/powers/{power}/stats"
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to get power stats: {await response.text()}")
                
                return await response.json()
    
    def calculate_power_strength(self, game_state: GameState, power: str) -> float:
        """Calculate relative strength of a power (0-1)."""
        # Count supply centers
        power_centers = game_state.supply_centers.get(power, 0)
        total_centers = sum(game_state.supply_centers.values())
        
        if total_centers == 0:
            return 0.0
        
        # Basic strength is supply center ratio
        strength = power_centers / total_centers
        
        # Adjust for unit count
        power_units = [u for u in game_state.units.values() if u.power == power]
        total_units = len(game_state.units)
        
        if total_units > 0:
            unit_ratio = len(power_units) / total_units
            strength = 0.7 * strength + 0.3 * unit_ratio
        
        return min(1.0, strength)