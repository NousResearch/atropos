#!/usr/bin/env python3
"""
Puzzle Generator for TextWorld

Generates logic puzzle games requiring problem-solving skills.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any

import textworld
from textworld import GameMaker, GameOptions
from textworld.generator import make_game, compile_game

from ..generation_utils import DEFAULT_OUTPUT_FOLDER

logger = logging.getLogger(__name__)


class PuzzleGenerator:
    """Generator for puzzle-based TextWorld games."""
    
    # Puzzle types and their configurations
    PUZZLE_TYPES = {
        "door_sequence": {
            "description": "Unlock doors in the correct sequence",
            "min_rooms": 4,
            "objects": ["red key", "blue key", "green key", "yellow key"],
            "rules": "Each door requires a specific colored key"
        },
        "combination_lock": {
            "description": "Find clues to unlock a combination",
            "min_rooms": 5,
            "objects": ["note", "diary", "calendar", "clock", "painting"],
            "rules": "Examine objects to find the combination"
        },
        "weight_puzzle": {
            "description": "Balance weights to open passages",
            "min_rooms": 3,
            "objects": ["heavy stone", "light stone", "medium stone", "scale"],
            "rules": "Place correct weights on the scale"
        },
        "light_puzzle": {
            "description": "Illuminate rooms in correct pattern",
            "min_rooms": 4,
            "objects": ["torch", "candle", "lamp", "mirror"],
            "rules": "Light sources must be placed strategically"
        },
        "container_puzzle": {
            "description": "Move items between containers",
            "min_rooms": 3,
            "objects": ["small box", "large box", "bag", "chest"],
            "rules": "Items must be organized correctly"
        }
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the puzzle generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        
    def generate(self,
                 difficulty: str = "medium",
                 puzzle_type: Optional[str] = None,
                 output_folder: str = DEFAULT_OUTPUT_FOLDER,
                 filename_prefix: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a puzzle-based game.
        
        Args:
            difficulty: Game difficulty (easy/medium/hard/expert)
            puzzle_type: Type of puzzle
            output_folder: Where to save the game
            filename_prefix: Prefix for the game file
            
        Returns:
            Tuple of (game_file_path, configuration)
        """
        # Select puzzle type
        if puzzle_type is None:
            puzzle_type = self.rng.choice(list(self.PUZZLE_TYPES.keys()))
        elif puzzle_type not in self.PUZZLE_TYPES:
            logger.warning(f"Unknown puzzle type: {puzzle_type}, using 'door_sequence'")
            puzzle_type = "door_sequence"
            
        template = self.PUZZLE_TYPES[puzzle_type]
        
        # Get difficulty parameters
        params = self._get_difficulty_params(difficulty)
        params["nb_rooms"] = max(params["nb_rooms"], template["min_rooms"])
        
        try:
            # Create game maker
            maker = GameMaker()
            
            # Generate the puzzle layout
            if puzzle_type == "door_sequence":
                game_file, config = self._generate_door_sequence_puzzle(
                    maker, params, template, difficulty, output_folder, filename_prefix
                )
            elif puzzle_type == "combination_lock":
                game_file, config = self._generate_combination_puzzle(
                    maker, params, template, difficulty, output_folder, filename_prefix
                )
            elif puzzle_type == "weight_puzzle":
                game_file, config = self._generate_weight_puzzle(
                    maker, params, template, difficulty, output_folder, filename_prefix
                )
            else:
                # Default to basic puzzle generation
                game_file, config = self._generate_basic_puzzle(
                    maker, params, template, difficulty, puzzle_type, output_folder, filename_prefix
                )
                
            return game_file, config
            
        except Exception as e:
            import traceback
            logger.error(f"Error generating puzzle game: {e}")
            traceback.print_exc()
            return None, {}
    
    def _generate_door_sequence_puzzle(self,
                                      maker: GameMaker,
                                      params: Dict[str, Any],
                                      template: Dict[str, Any],
                                      difficulty: str,
                                      output_folder: str,
                                      filename_prefix: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a door sequence puzzle."""
        nb_rooms = params["nb_rooms"]
        
        # Create rooms
        rooms = []
        for i in range(nb_rooms):
            room = maker.new_room(f"room_{i}")
            rooms.append(room)
            if i == 0:
                maker.set_player(room)
        
        # Create linear path with doors
        keys = template["objects"][:nb_rooms-1]
        for i in range(nb_rooms - 1):
            # Connect rooms - maker.connect expects exit objects, not room objects
            path = maker.connect(rooms[i].exits["east"], rooms[i+1].exits["west"])
            door = maker.new_door(path, name=f"{keys[i].split()[0]} door")
            
            # Place key in a room (not the current one)
            key = maker.new(type="k", name=keys[i])
            key_room_idx = self.rng.choice([j for j in range(nb_rooms) if j != i+1])
            rooms[key_room_idx].add(key)
            
            # Make door require key
            maker.add_fact("match", key, door)
            maker.add_fact("locked", door)
        
        # Add goal object in a random room (but not the starting room)
        treasure = maker.new(type="o", name="treasure")
        treasure_room_idx = self.rng.choice(range(1, nb_rooms))  # Not room 0 (start)
        rooms[treasure_room_idx].add(treasure)
        
        # Add some dead-end branches to avoid "last room = treasure" pattern
        if difficulty in ["hard", "expert"] and nb_rooms > 4:
            # Create 1-2 dead end branches
            num_branches = self.rng.randint(1, 2)
            for _ in range(num_branches):
                branch_from = self.rng.choice(range(nb_rooms))
                dead_end = maker.new_room(f"dead_end_{branch_from}")
                # Connect to a random direction that's free
                directions = ["north", "south"]  # Avoid east/west used for main path
                for direction in directions:
                    opposite = {"north": "south", "south": "north"}[direction]
                    if not rooms[branch_from].exits[direction].destination:
                        maker.connect(rooms[branch_from].exits[direction], dead_end.exits[opposite])
                        # Add a distractor object
                        distractor = maker.new(type="o", name=self.rng.choice(["fake treasure", "empty chest", "old scroll", "dusty book"]))
                        dead_end.add(distractor)
                        break
        
        # Create quest - we need to navigate to where the treasure actually is
        quest_commands = []
        
        # First collect all keys (order doesn't matter for collection)
        for key_name in keys:
            quest_commands.append(f"take {key_name}")
            
        # Then navigate through doors to the treasure room
        # This is simplified - in reality the player would need to explore
        current_room = 0
        while current_room < treasure_room_idx:
            if current_room < len(keys):
                quest_commands.extend([
                    f"unlock {keys[current_room].split()[0]} door with {keys[current_room]}",
                    "go east"
                ])
            current_room += 1
            
        quest_commands.append("take treasure")
        
        maker.set_quest_from_commands(quest_commands)
        
        # Add distractors
        if params["nb_objects"] > len(keys):
            maker.generate_distractors(params["nb_objects"] - len(keys))
        
        # Build and compile
        game = maker.build()
        
        options = GameOptions()
        seed_value = self.rng.randint(0, 65535)

        options.seeds = seed_value
        
        if filename_prefix is None:
            filename_prefix = "puzzle_door_sequence"
        
        game_filename = f"{filename_prefix}_{difficulty}_seed{seed_value}.z8"
        options.path = f"{output_folder}/{game_filename}"
        
        game_file = compile_game(game, options)
        
        if game_file:
            config = {
                "type": "puzzle",
                "puzzle_type": "door_sequence",
                "difficulty": difficulty,
                "description": template["description"],
                "nb_rooms": nb_rooms,
                "nb_doors": len(keys),
                "seed": seed_value
            }
            return game_file, config
        
        return None, {}
    
    def _generate_combination_puzzle(self,
                                   maker: GameMaker,
                                   params: Dict[str, Any],
                                   template: Dict[str, Any],
                                   difficulty: str,
                                   output_folder: str,
                                   filename_prefix: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a combination lock puzzle."""
        nb_rooms = params["nb_rooms"]
        
        # Create rooms
        rooms = []
        for i in range(nb_rooms):
            room = maker.new_room(f"room_{i}")
            rooms.append(room)
            if i == 0:
                maker.set_player(room)
        
        # Connect rooms in a hub pattern
        for i in range(1, nb_rooms):
            direction = ["north", "south", "east", "west"][i-1]
            opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}[direction]
            maker.connect(rooms[0].exits[direction], rooms[i].exits[opposite])
        
        # Create combination (based on difficulty)
        combo_length = 3 if difficulty == "easy" else 4 if difficulty == "medium" else 5
        combination = "".join([str(self.rng.randint(0, 9)) for _ in range(combo_length)])
        
        # Place clues
        clue_objects = template["objects"][:combo_length]
        for i, (digit, clue_name) in enumerate(zip(combination, clue_objects)):
            clue = maker.new(type="o", name=clue_name, desc=f"It has the number {digit} written on it.")
            room_idx = (i + 1) % nb_rooms
            rooms[room_idx].add(clue)
        
        # Create locked container with treasure
        safe = maker.new(type="c", name="safe", desc=f"A safe with a {combo_length}-digit combination lock.")
        rooms[0].add(safe)
        maker.add_fact("locked", safe)
        
        treasure = maker.new(type="o", name="treasure")
        safe.add(treasure)
        
        # Create quest (examining all clues then opening safe)
        quest_commands = []
        for clue_name in clue_objects:
            quest_commands.append(f"examine {clue_name}")
        # Note: In a real implementation, we'd need a custom command for entering the combination
        quest_commands.extend([
            "open safe",  # This would require entering the combination
            "take treasure"
        ])
        
        maker.set_walkthrough(quest_commands)
        
        # Build and compile
        game = maker.build()
        
        options = GameOptions()
        seed_value = self.rng.randint(0, 65535)

        options.seeds = seed_value
        
        if filename_prefix is None:
            filename_prefix = "puzzle_combination"
        
        game_filename = f"{filename_prefix}_{difficulty}_seed{seed_value}.z8"
        options.path = f"{output_folder}/{game_filename}"
        
        game_file = compile_game(game, options)
        
        if game_file:
            config = {
                "type": "puzzle",
                "puzzle_type": "combination_lock",
                "difficulty": difficulty,
                "description": template["description"],
                "nb_rooms": nb_rooms,
                "combination_length": combo_length,
                "seed": seed_value
            }
            return game_file, config
        
        return None, {}
    
    def _generate_weight_puzzle(self,
                               maker: GameMaker,
                               params: Dict[str, Any],
                               template: Dict[str, Any],
                               difficulty: str,
                               output_folder: str,
                               filename_prefix: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a weight balance puzzle."""
        nb_rooms = params["nb_rooms"]
        
        # Create rooms
        rooms = []
        for i in range(nb_rooms):
            room = maker.new_room(f"room_{i}")
            rooms.append(room)
            if i == 0:
                maker.set_player(room)
        
        # Connect rooms linearly
        for i in range(nb_rooms - 1):
            maker.connect(rooms[i].exits["east"], rooms[i+1].exits["west"])
        
        # Create scale in middle room
        scale_room_idx = nb_rooms // 2
        scale = maker.new(type="s", name="scale", desc="A balance scale with two plates.")
        rooms[scale_room_idx].add(scale)
        
        # Create weights
        weights = []
        weight_values = {"heavy": 3, "medium": 2, "light": 1}
        for weight_type in ["heavy", "medium", "light"]:
            if difficulty != "easy" or weight_type != "medium":  # Easy mode has fewer weights
                weight = maker.new(type="o", name=f"{weight_type} stone")
                weight.weight = weight_values[weight_type]
                weights.append(weight)
                # Distribute weights across rooms
                room_idx = self.rng.randint(0, nb_rooms - 1)
                rooms[room_idx].add(weight)
        
        # Create door that opens with correct balance
        if nb_rooms > 2:
            path = maker.connect(rooms[scale_room_idx].exits["north"], rooms[-1].exits["south"])
            door = maker.new_door(path, name="heavy door", desc="This door opens when the scale is balanced.")
            maker.add_fact("locked", door)
        
        # Add treasure in final room
        treasure = maker.new(type="o", name="treasure")
        rooms[-1].add(treasure)
        
        # Create quest
        target_weight = 4 if difficulty == "easy" else 5
        # This is simplified - in reality we'd need custom mechanics for the scale
        quest_commands = [
            "take light stone",
            "take heavy stone",
            "put light stone on scale",
            "put heavy stone on scale",
            "go north",
            "take treasure"
        ]
        
        maker.set_walkthrough(quest_commands)
        
        # Build and compile
        game = maker.build()
        
        options = GameOptions()
        seed_value = self.rng.randint(0, 65535)

        options.seeds = seed_value
        
        if filename_prefix is None:
            filename_prefix = "puzzle_weight"
        
        game_filename = f"{filename_prefix}_{difficulty}_seed{seed_value}.z8"
        options.path = f"{output_folder}/{game_filename}"
        
        game_file = compile_game(game, options)
        
        if game_file:
            config = {
                "type": "puzzle",
                "puzzle_type": "weight_puzzle",
                "difficulty": difficulty,
                "description": template["description"],
                "nb_rooms": nb_rooms,
                "nb_weights": len(weights),
                "target_weight": target_weight,
                "seed": seed_value
            }
            return game_file, config
        
        return None, {}
    
    def _generate_basic_puzzle(self,
                              maker: GameMaker,
                              params: Dict[str, Any],
                              template: Dict[str, Any],
                              difficulty: str,
                              puzzle_type: str,
                              output_folder: str,
                              filename_prefix: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a basic puzzle using standard generation."""
        # Use standard game generation with puzzle-themed settings
        options = GameOptions()
        seed_value = self.rng.randint(0, 65535)

        options.seeds = seed_value
        options.nb_rooms = params["nb_rooms"]
        options.nb_objects = params["nb_objects"]
        options.quest_length = params["quest_length"]
        options.quest_breadth = 1  # Linear puzzles
        options.grammar.theme = "house"
        options.grammar.include_adj = True
        
        # Generate the game
        game = make_game(options)
        
        # Compile the game
        if filename_prefix is None:
            filename_prefix = f"puzzle_{puzzle_type}"
        
        game_filename = f"{filename_prefix}_{difficulty}_seed{seed_value}.z8"
        options.path = f"{output_folder}/{game_filename}"
        
        game_file = compile_game(game, options)
        
        if game_file:
            config = {
                "type": "puzzle",
                "puzzle_type": puzzle_type,
                "difficulty": difficulty,
                "description": template["description"],
                "nb_rooms": params["nb_rooms"],
                "nb_objects": params["nb_objects"],
                "quest_length": params["quest_length"],
                "seed": seed_value
            }
            return game_file, config
        
        return None, {}
    
    def _get_difficulty_params(self, difficulty: str) -> Dict[str, Any]:
        """Get game parameters based on difficulty level."""
        difficulty_settings = {
            "easy": {
                "nb_rooms": 3,
                "nb_objects": 4,
                "quest_length": 4,
                "complexity": "simple"
            },
            "medium": {
                "nb_rooms": 5,
                "nb_objects": 8,
                "quest_length": 6,
                "complexity": "moderate"
            },
            "hard": {
                "nb_rooms": 7,
                "nb_objects": 12,
                "quest_length": 10,
                "complexity": "complex"
            },
            "expert": {
                "nb_rooms": 10,
                "nb_objects": 16,
                "quest_length": 15,
                "complexity": "very_complex"
            }
        }
        
        return difficulty_settings.get(difficulty, difficulty_settings["medium"])


if __name__ == "__main__":
    """Test the puzzle generator."""
    import os
    
    # Create output directory
    os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)
    
    # Create generator
    generator = PuzzleGenerator(seed=42)
    
    # Test different puzzle types
    for puzzle_type in ["door_sequence", "combination_lock", "weight_puzzle"]:
        for difficulty in ["easy", "medium"]:
            print(f"\nGenerating {puzzle_type} puzzle ({difficulty})...")
            game_file, config = generator.generate(
                difficulty=difficulty,
                puzzle_type=puzzle_type
            )
            
            if game_file:
                print(f"SUCCESS: {game_file}")
                print(f"Config: {config}")
            else:
                print("FAILED")