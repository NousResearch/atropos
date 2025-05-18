import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional, Literal
import random
import re
import xml.etree.ElementTree as ET

from .game_2048 import Game2048

class Environment2048:
    """
    An environment class for agents to interact with the 2048 game.
    """
    
    def __init__(self, winning_value: int = 2048, max_moves: int = 1000):
        """
        Initialize the 2048 environment.
        
        Args:
            winning_value: The tile value needed to win the game (default: 2048)
            max_moves: Maximum number of moves before terminating the game (default: 1000)
        """
        self.game = Game2048()
        self.winning_value = winning_value
        self.max_moves = max_moves
        self.system_message = (
            "You are an excellent 2048 player. Your goal is to combine tiles "
            f"to reach the {winning_value} tile. After each move, a new 2 or 4 "
            "tile will appear randomly on the board. "
            "Choose the move most likely to lead to success. "
            "Available moves are 'left', 'right', 'up', 'down'. "
            "Return your move as an XML object with a single property 'move', "
            "like so: <move>left</move>"
        )
        
    def reset(self) -> Tuple[Dict[str, Any], str]:
        """
        Reset the environment to a fresh game.
        
        Returns:
            Tuple of (observation, prompt)
        """
        self.game = Game2048()
        observation = self._get_observation()
        prompt = self._get_prompt()
        
        return observation, prompt
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get the current game state as an observation.
        
        Returns:
            Dictionary containing the current game state
        """
        state = self.game.get_state()
        return {
            'id': state['id'],
            'board': state['board'].tolist(),  # Convert numpy array to list for JSON serialization
            'score': state['score'],
            'moves': state['moves'],
            'game_over': state['game_over'],
            'max_tile': state['max_tile'],
            'winning_value': self.winning_value
        }
    
    def _get_prompt(self) -> str:
        """
        Get the prompt to send to the agent.
        
        Returns:
            A string representation of the board
        """
        return self.game.render()
    
    def step(self, action: str) -> Tuple[Dict[str, Any], str, bool, float, Dict[str, Any]]:
        """
        Take a step in the environment by applying the agent's action.
        
        Args:
            action: The agent's action as a string
            
        Returns:
            Tuple of (observation, prompt, done, reward, info)
        """
        # Parse the action
        direction = self._parse_action(action)
        
        if direction is None:
            # Invalid action format
            return (
                self._get_observation(),
                "Invalid action format. Please use <move>direction</move>.",
                True,
                -1.0,
                {'error': 'invalid_format'}
            )
            
        if direction not in ['left', 'right', 'up', 'down']:
            # Invalid direction
            return (
                self._get_observation(),
                f"Invalid direction: {direction}. Valid options are: left, right, up, down.",
                True,
                -1.0,
                {'error': 'invalid_direction'}
            )
        
        # Apply the move
        _, score_added, changed = self.game.move(direction)
        
        # Get new state
        observation = self._get_observation()
        prompt = self._get_prompt()
        
        # Check if game is over
        done = (
            observation['game_over'] or
            observation['max_tile'] >= self.winning_value or
            observation['moves'] >= self.max_moves
        )
        
        # Calculate reward
        reward = self._calculate_reward(observation, score_added, changed, done)
        
        # Additional info
        info = {
            'score_added': score_added,
            'changed': changed,
            'win': observation['max_tile'] >= self.winning_value
        }
        
        return observation, prompt, done, reward, info
    
    def _parse_action(self, action: str) -> Optional[str]:
        """
        Parse the action from the agent's response.
        
        Args:
            action: The agent's action as a string
            
        Returns:
            The parsed direction or None if invalid
        """
        try:
            # Try to parse as XML
            root = ET.fromstring(action)
            if root.tag == 'move':
                return root.text.strip().lower()
            
            # If XML is valid but not in expected format, search for the move tag
            for child in root.iter('move'):
                return child.text.strip().lower()
                
            return None
        except Exception:
            # If XML parsing failed, try regex fallback
            match = re.search(r'<move>(.*?)</move>', action, re.IGNORECASE)
            if match:
                return match.group(1).strip().lower()
            
            # Try to extract from plain text if agent didn't use XML
            for direction in ['left', 'right', 'up', 'down']:
                if re.search(r'\b' + direction + r'\b', action.lower()):
                    return direction
                    
            return None
    
    def _calculate_reward(self, observation: Dict[str, Any], score_added: int, changed: bool, done: bool) -> float:
        """
        Calculate the reward for the current step.
        
        Args:
            observation: The current observation
            score_added: Score added in this step
            changed: Whether the board changed
            done: Whether the episode is done
            
        Returns:
            The calculated reward
        """
        # If the move was invalid (didn't change the board)
        if not changed:
            return -0.1
        
        # If the game is won
        if observation['max_tile'] >= self.winning_value:
            return 10.0
            
        # If the game is over but not won
        if observation['game_over']:
            # Scale based on max tile achieved
            max_tile = observation['max_tile']
            progress = np.log2(max_tile) / np.log2(self.winning_value)
            return progress - 0.5  # Penalty for not winning
        
        # Reward based on score added
        score_reward = score_added / 100.0  # Normalize score
        
        # Small bonus for higher tiles
        max_tile_reward = np.log2(observation['max_tile']) / np.log2(self.winning_value) * 0.1
        
        return score_reward + max_tile_reward
    
    def render(self) -> str:
        """
        Render the current game state.
        
        Returns:
            A string representation of the game state
        """
        state = self._get_observation()
        
        header = f"Game ID: {state['id']} | Score: {state['score']} | Moves: {state['moves']} | Max Tile: {state['max_tile']}\n"
        board = self.game.render()
        
        if state['game_over']:
            footer = "Game Over!"
            if state['max_tile'] >= self.winning_value:
                footer += " You won!"
            else:
                footer += " You lost."
        else:
            footer = ""
            
        return header + board + footer 