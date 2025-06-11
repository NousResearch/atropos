#!/usr/bin/env python3
"""
Quest Generator for TextWorld

Generates objective-driven games with complex multi-step quests.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any

import textworld
from textworld import GameMaker, GameOptions
from textworld.generator import make_game, compile_game

from ..generation_utils import DEFAULT_OUTPUT_FOLDER

logger = logging.getLogger(__name__)


class QuestGenerator:
    """Generator for quest-based TextWorld games."""
    
    # Quest templates for different objectives
    QUEST_TEMPLATES = {
        "fetch": {
            "description": "Find and retrieve a specific object",
            "min_rooms": 3,
            "min_objects": 5,
            "quest_objects": ["key", "book", "coin", "gem", "scroll", "artifact"]
        },
        "delivery": {
            "description": "Deliver an object to a specific location",
            "min_rooms": 4,
            "min_objects": 6,
            "quest_objects": ["letter", "package", "message", "gift", "document"]
        },
        "rescue": {
            "description": "Find and rescue someone or something",
            "min_rooms": 5,
            "min_objects": 8,
            "quest_objects": ["cat", "bird", "child", "friend", "pet"]
        },
        "exploration": {
            "description": "Explore and discover hidden areas",
            "min_rooms": 6,
            "min_objects": 10,
            "quest_objects": ["map", "compass", "torch", "rope", "ladder"]
        },
        "puzzle": {
            "description": "Solve puzzles to progress",
            "min_rooms": 4,
            "min_objects": 7,
            "quest_objects": ["lever", "button", "switch", "mechanism", "panel"]
        }
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the quest generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        
    def generate(self,
                 difficulty: str = "medium",
                 quest_type: Optional[str] = None,
                 output_folder: str = DEFAULT_OUTPUT_FOLDER,
                 filename_prefix: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a quest-based game.
        
        Args:
            difficulty: Game difficulty (easy/medium/hard/expert)
            quest_type: Type of quest (fetch/delivery/rescue/exploration/puzzle)
            output_folder: Where to save the game
            filename_prefix: Prefix for the game file
            
        Returns:
            Tuple of (game_file_path, configuration)
        """
        # Select quest type
        if quest_type is None:
            quest_type = self.rng.choice(list(self.QUEST_TEMPLATES.keys()))
        elif quest_type not in self.QUEST_TEMPLATES:
            logger.warning(f"Unknown quest type: {quest_type}, using 'fetch'")
            quest_type = "fetch"
            
        template = self.QUEST_TEMPLATES[quest_type]
        
        # Determine game parameters based on difficulty
        params = self._get_difficulty_params(difficulty)
        
        # Ensure minimum requirements from template
        params["nb_rooms"] = max(params["nb_rooms"], template["min_rooms"])
        params["nb_objects"] = max(params["nb_objects"], template["min_objects"])
        
        # Create game options
        options = GameOptions()
        options.seeds = self.rng.randint(0, 65535)
        options.nb_rooms = params["nb_rooms"]
        options.nb_objects = params["nb_objects"]
        options.quest_length = params["quest_length"]
        options.quest_breadth = params["quest_breadth"]
        options.grammar.theme = "house"
        options.grammar.include_adj = True
        
        # Set quest generation parameters
        options.chaining.max_depth = params["quest_depth"]
        options.chaining.max_breadth = params["quest_breadth"]
        options.chaining.max_length = params["quest_length"]
        
        # Generate filename
        if filename_prefix is None:
            filename_prefix = f"quest_{quest_type}_{difficulty}"
        
        game_filename = f"{filename_prefix}_seed{options.seeds}.z8"
        options.path = f"{output_folder}/{game_filename}"
        
        try:
            # Generate the game
            game = make_game(options)
            
            # Compile the game
            game_file = compile_game(game, options)
            
            if game_file:
                config = {
                    "type": "quest",
                    "quest_type": quest_type,
                    "difficulty": difficulty,
                    "description": template["description"],
                    "nb_rooms": params["nb_rooms"],
                    "nb_objects": params["nb_objects"],
                    "quest_length": params["quest_length"],
                    "quest_breadth": params["quest_breadth"],
                    "quest_depth": params["quest_depth"],
                    "seed": options.seeds
                }
                
                logger.info(f"Generated quest game: {quest_type} ({difficulty}) - {game_file}")
                return game_file, config
            else:
                logger.error("Failed to compile quest game")
                return None, {}
                
        except Exception as e:
            logger.error(f"Error generating quest game: {e}")
            return None, {}
    
    def generate_with_custom_quest(self,
                                  commands: List[str],
                                  difficulty: str = "medium",
                                  output_folder: str = DEFAULT_OUTPUT_FOLDER,
                                  filename_prefix: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a game with a custom quest defined by commands.
        
        Args:
            commands: List of commands that solve the quest
            difficulty: Game difficulty for world generation
            output_folder: Where to save the game
            filename_prefix: Prefix for the game file
            
        Returns:
            Tuple of (game_file_path, configuration)
        """
        params = self._get_difficulty_params(difficulty)
        
        try:
            # Create game maker
            maker = GameMaker()
            
            # Generate rooms
            for i in range(params["nb_rooms"]):
                room = maker.new_room(f"room_{i}")
                if i == 0:
                    maker.set_player(room)
            
            # Connect rooms (simple linear connection for now)
            rooms = maker.findall("r")
            for i in range(len(rooms) - 1):
                maker.connect(rooms[i].exits["east"], rooms[i+1].exits["west"])
            
            # Add objects
            maker.generate_distractors(params["nb_objects"])
            
            # Set quest from commands
            maker.set_quest_from_commands(commands)
            
            # Build the game
            game = maker.build()
            
            # Create options for compilation
            options = GameOptions()
            options.seeds = self.rng.randint(0, 65535)
            
            if filename_prefix is None:
                filename_prefix = "quest_custom"
            
            game_filename = f"{filename_prefix}_seed{options.seeds}.z8"
            options.path = f"{output_folder}/{game_filename}"
            
            # Compile the game
            game_file = compile_game(game, options)
            
            if game_file:
                config = {
                    "type": "quest",
                    "quest_type": "custom",
                    "difficulty": difficulty,
                    "description": "Custom quest defined by commands",
                    "nb_rooms": params["nb_rooms"],
                    "nb_objects": params["nb_objects"],
                    "commands": commands,
                    "seed": options.seeds
                }
                
                logger.info(f"Generated custom quest game: {game_file}")
                return game_file, config
            else:
                logger.error("Failed to compile custom quest game")
                return None, {}
                
        except Exception as e:
            logger.error(f"Error generating custom quest game: {e}")
            return None, {}
    
    def _get_difficulty_params(self, difficulty: str) -> Dict[str, Any]:
        """Get game parameters based on difficulty level.
        
        Args:
            difficulty: easy/medium/hard/expert
            
        Returns:
            Dictionary of game parameters
        """
        difficulty_settings = {
            "easy": {
                "nb_rooms": 3,
                "nb_objects": 5,
                "quest_length": 3,
                "quest_breadth": 1,
                "quest_depth": 3
            },
            "medium": {
                "nb_rooms": 6,
                "nb_objects": 10,
                "quest_length": 5,
                "quest_breadth": 2,
                "quest_depth": 5
            },
            "hard": {
                "nb_rooms": 10,
                "nb_objects": 15,
                "quest_length": 8,
                "quest_breadth": 3,
                "quest_depth": 8
            },
            "expert": {
                "nb_rooms": 15,
                "nb_objects": 20,
                "quest_length": 12,
                "quest_breadth": 4,
                "quest_depth": 12
            }
        }
        
        return difficulty_settings.get(difficulty, difficulty_settings["medium"])


if __name__ == "__main__":
    """Test the quest generator."""
    import os
    
    # Create output directory
    os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)
    
    # Create generator
    generator = QuestGenerator(seed=42)
    
    # Test different quest types and difficulties
    for quest_type in ["fetch", "delivery", "exploration"]:
        for difficulty in ["easy", "medium", "hard"]:
            print(f"\nGenerating {quest_type} quest ({difficulty})...")
            game_file, config = generator.generate(
                difficulty=difficulty,
                quest_type=quest_type
            )
            
            if game_file:
                print(f"SUCCESS: {game_file}")
                print(f"Config: {config}")
            else:
                print("FAILED")
    
    # Test custom quest
    print("\nGenerating custom quest...")
    commands = ["take key", "unlock door with key", "go east", "take treasure"]
    game_file, config = generator.generate_with_custom_quest(
        commands=commands,
        difficulty="medium"
    )
    
    if game_file:
        print(f"SUCCESS: {game_file}")
        print(f"Config: {config}")
    else:
        print("FAILED")