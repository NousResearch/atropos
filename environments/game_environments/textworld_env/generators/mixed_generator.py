#!/usr/bin/env python3
"""
Mixed Generator for TextWorld

Generates games that combine elements from different game types.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any

import textworld
from textworld import GameMaker, GameOptions
from textworld.generator import make_game, compile_game

from ..generation_utils import DEFAULT_OUTPUT_FOLDER, compile_game_with_retry
from .quest_generator import QuestGenerator
from .puzzle_generator import PuzzleGenerator
from .navigation_generator import NavigationGenerator

logger = logging.getLogger(__name__)


class MixedGenerator:
    """Generator for games combining multiple game types."""

    # Mixed game templates
    MIXED_TEMPLATES = {
        "quest_puzzle": {
            "description": "A quest that requires solving puzzles",
            "components": ["quest", "puzzle"],
            "min_rooms": 6,
            "features": ["locked doors", "key puzzles", "fetch quests"],
        },
        "navigation_puzzle": {
            "description": "Navigate a maze while solving puzzles",
            "components": ["navigation", "puzzle"],
            "min_rooms": 8,
            "features": ["maze layout", "door puzzles", "landmarks"],
        },
        "exploration_quest": {
            "description": "Explore and complete multiple objectives",
            "components": ["navigation", "quest"],
            "min_rooms": 10,
            "features": ["open world", "multiple quests", "hidden areas"],
        },
        "dungeon_crawler": {
            "description": "Classic dungeon with puzzles and treasures",
            "components": ["navigation", "puzzle", "quest"],
            "min_rooms": 12,
            "features": ["dungeon layout", "traps", "boss room", "treasure"],
        },
        "adventure": {
            "description": "Full adventure combining all elements",
            "components": ["quest", "puzzle", "navigation"],
            "min_rooms": 15,
            "features": ["story", "npcs", "multiple paths", "secrets"],
        },
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize the mixed generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.quest_gen = QuestGenerator(seed)
        self.puzzle_gen = PuzzleGenerator(seed)
        self.nav_gen = NavigationGenerator(seed)

    def generate(
        self,
        difficulty: str = "medium",
        mixed_type: Optional[str] = None,
        output_folder: str = DEFAULT_OUTPUT_FOLDER,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a mixed-type game.

        Args:
            difficulty: Game difficulty (easy/medium/hard/expert)
            mixed_type: Type of mixed game
            output_folder: Where to save the game
            filename_prefix: Prefix for the game file

        Returns:
            Tuple of (game_file_path, configuration)
        """
        # Select mixed type
        if mixed_type is None:
            mixed_type = self.rng.choice(list(self.MIXED_TEMPLATES.keys()))
        elif mixed_type not in self.MIXED_TEMPLATES:
            logger.warning(f"Unknown mixed type: {mixed_type}, using 'quest_puzzle'")
            mixed_type = "quest_puzzle"

        template = self.MIXED_TEMPLATES[mixed_type]

        # Get difficulty parameters
        params = self._get_difficulty_params(difficulty)
        params["nb_rooms"] = max(params["nb_rooms"], template["min_rooms"])

        try:
            # Generate based on mixed type
            if mixed_type == "quest_puzzle":
                return self._generate_quest_puzzle(
                    params, template, difficulty, output_folder, filename_prefix
                )
            elif mixed_type == "navigation_puzzle":
                return self._generate_navigation_puzzle(
                    params, template, difficulty, output_folder, filename_prefix
                )
            elif mixed_type == "exploration_quest":
                return self._generate_exploration_quest(
                    params, template, difficulty, output_folder, filename_prefix
                )
            elif mixed_type == "dungeon_crawler":
                return self._generate_dungeon_crawler(
                    params, template, difficulty, output_folder, filename_prefix
                )
            else:  # adventure
                return self._generate_adventure(
                    params, template, difficulty, output_folder, filename_prefix
                )

        except Exception as e:
            logger.error(f"Error generating mixed game: {e}")
            return None, {}

    def _generate_quest_puzzle(
        self,
        params: Dict[str, Any],
        template: Dict[str, Any],
        difficulty: str,
        output_folder: str,
        filename_prefix: Optional[str],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a game combining quests and puzzles."""
        maker = GameMaker()

        # Create room layout
        rooms = self._create_room_layout(maker, params["nb_rooms"], "linear_branching")

        # Place player
        maker.set_player(rooms[0])

        # Create main quest path with puzzle obstacles
        quest_items = ["ancient scroll", "mystic orb", "golden chalice"]
        puzzle_keys = ["red crystal", "blue crystal", "green crystal"]

        # Place puzzle doors along the path
        for i in range(min(3, len(rooms) - 1)):
            if i + 1 < len(rooms):
                # Create puzzle door
                path = self._find_or_create_path(maker, rooms[i], rooms[i + 1])
                if path:
                    door = maker.new_door(
                        path, name=f"{puzzle_keys[i].split()[0]} door"
                    )
                    maker.add_fact("locked", door)

                    # Place key in a side room (puzzle)
                    key = maker.new(type="k", name=puzzle_keys[i])
                    key_room = self.rng.choice([r for r in rooms if r != rooms[i + 1]])
                    key_room.add(key)
                    maker.add_fact("match", key, door)

                    # Place quest item behind door
                    if i < len(quest_items):
                        item = maker.new(type="o", name=quest_items[i])
                        rooms[i + 1].add(item)

        # Create final goal
        altar = maker.new(
            type="s", name="ancient altar", desc="Place the three sacred items here."
        )
        rooms[-1].add(altar)

        # Build quest commands
        quest_commands = []
        for i, (key, item) in enumerate(zip(puzzle_keys[:3], quest_items[:3])):
            quest_commands.extend(
                [
                    f"take {key}",
                    f"unlock {key.split()[0]} door with {key}",
                    "go east",
                    f"take {item}",
                ]
            )

        quest_commands.extend(
            [f"put {item} on ancient altar" for item in quest_items[:3]]
        )

        maker.set_walkthrough(quest_commands)

        # Add distractors
        maker.generate_distractors(params["nb_objects"])

        # Build and compile
        game = maker.build()

        options = GameOptions()
        seed_value = self.rng.randint(0, 65535)

        options.seeds = seed_value

        if filename_prefix is None:
            filename_prefix = "mixed_quest_puzzle"

        game_filename = f"{filename_prefix}_{difficulty}_seed{seed_value}.z8"
        options.path = f"{output_folder}/{game_filename}"

        game_file = compile_game_with_retry(game, options)

        if game_file:
            config = {
                "type": "mixed",
                "mixed_type": "quest_puzzle",
                "difficulty": difficulty,
                "description": template["description"],
                "nb_rooms": len(rooms),
                "features": template["features"],
                "seed": seed_value,
            }
            return game_file, config

        return None, {}

    def _generate_navigation_puzzle(
        self,
        params: Dict[str, Any],
        template: Dict[str, Any],
        difficulty: str,
        output_folder: str,
        filename_prefix: Optional[str],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a maze with puzzles."""
        maker = GameMaker()

        # Create maze layout
        rooms = self._create_room_layout(maker, params["nb_rooms"], "maze")

        # Place player at entrance
        maker.set_player(rooms[0])

        # Place landmarks for navigation
        landmarks = [
            "stone obelisk",
            "crystal formation",
            "ancient mural",
            "glowing orb",
        ]
        landmark_rooms = self.rng.sample(
            rooms[1:-1], min(len(landmarks), len(rooms) - 2)
        )

        for landmark_name, room in zip(landmarks, landmark_rooms):
            landmark = maker.new(type="o", name=landmark_name)
            room.add(landmark)

        # Create puzzle elements in the maze
        # Color sequence puzzle
        colors = ["red", "blue", "green", "yellow"]
        switches = []

        for i, color in enumerate(colors[: min(4, params["nb_rooms"] // 2)]):
            switch = maker.new(type="o", name=f"{color} switch")
            switch_room = self.rng.choice(rooms[1:-1])
            switch_room.add(switch)
            switches.append((color, switch))

        # Create exit that requires switches
        exit_path = self._find_or_create_path(maker, rooms[-1], rooms[0])
        if exit_path:
            exit_door = maker.new_door(exit_path, name="maze exit")
            maker.add_fact("locked", exit_door)

        # Place treasure
        treasure = maker.new(type="o", name="maze treasure")
        rooms[-1].add(treasure)

        # Create walkthrough
        quest_commands = ["look"]
        # Navigate and activate switches
        for color, switch in switches:
            quest_commands.extend(
                ["go north", f"push {color} switch"]  # Simplified navigation
            )
        quest_commands.extend(
            ["take maze treasure", "go south", "unlock maze exit"]  # Return
        )

        maker.set_walkthrough(quest_commands)

        # Build and compile
        game = maker.build()

        options = GameOptions()
        seed_value = self.rng.randint(0, 65535)

        options.seeds = seed_value

        if filename_prefix is None:
            filename_prefix = "mixed_navigation_puzzle"

        game_filename = f"{filename_prefix}_{difficulty}_seed{seed_value}.z8"
        options.path = f"{output_folder}/{game_filename}"

        game_file = compile_game_with_retry(game, options)

        if game_file:
            config = {
                "type": "mixed",
                "mixed_type": "navigation_puzzle",
                "difficulty": difficulty,
                "description": template["description"],
                "nb_rooms": len(rooms),
                "nb_landmarks": len(landmark_rooms),
                "nb_switches": len(switches),
                "features": template["features"],
                "seed": seed_value,
            }
            return game_file, config

        return None, {}

    def _generate_exploration_quest(
        self,
        params: Dict[str, Any],
        template: Dict[str, Any],
        difficulty: str,
        output_folder: str,
        filename_prefix: Optional[str],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate an open exploration game with multiple quests."""
        # Use standard generation with custom parameters
        options = GameOptions()
        seed_value = self.rng.randint(0, 65535)

        options.seeds = seed_value
        options.nb_rooms = params["nb_rooms"]
        options.nb_objects = params["nb_objects"]
        options.nb_parallel_quests = params.get("nb_parallel_quests", 3)
        options.quest_length = params["quest_length"]
        options.quest_breadth = params["quest_breadth"]
        options.grammar.theme = "house"
        options.grammar.include_adj = True

        # Generate the game
        game = make_game(options)

        # Compile the game
        if filename_prefix is None:
            filename_prefix = "mixed_exploration_quest"

        game_filename = f"{filename_prefix}_{difficulty}_seed{seed_value}.z8"
        options.path = f"{output_folder}/{game_filename}"

        game_file = compile_game_with_retry(game, options)

        if game_file:
            config = {
                "type": "mixed",
                "mixed_type": "exploration_quest",
                "difficulty": difficulty,
                "description": template["description"],
                "nb_rooms": params["nb_rooms"],
                "nb_objects": params["nb_objects"],
                "nb_parallel_quests": options.nb_parallel_quests,
                "quest_length": params["quest_length"],
                "features": template["features"],
                "seed": seed_value,
            }
            return game_file, config

        return None, {}

    def _generate_dungeon_crawler(
        self,
        params: Dict[str, Any],
        template: Dict[str, Any],
        difficulty: str,
        output_folder: str,
        filename_prefix: Optional[str],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a dungeon crawler with all elements."""
        maker = GameMaker()

        # Create dungeon layout with multiple levels
        rooms = self._create_room_layout(maker, params["nb_rooms"], "dungeon")

        # Designate special rooms
        entrance = rooms[0]
        boss_room = rooms[-1]
        treasure_rooms = self.rng.sample(rooms[1:-1], min(3, len(rooms) // 4))

        # Place player
        maker.set_player(entrance)

        # Create dungeon elements
        # 1. Keys and locked doors
        dungeon_keys = ["iron key", "silver key", "golden key"]
        for i, key_name in enumerate(dungeon_keys[: min(3, len(rooms) // 3)]):
            # Create locked door
            if i * 3 + 3 < len(rooms):
                path = self._find_or_create_path(maker, rooms[i * 3], rooms[i * 3 + 3])
                if path:
                    door = maker.new_door(path, name=f"{key_name.split()[0]} door")
                    maker.add_fact("locked", door)

                    # Hide key
                    key = maker.new(type="k", name=key_name)
                    key_room = self.rng.choice(rooms[: i * 3 + 2])
                    key_room.add(key)
                    maker.add_fact("match", key, door)

        # 2. Treasures
        treasures = ["ruby", "emerald", "diamond", "ancient artifact"]
        for treasure_name, room in zip(treasures, treasure_rooms):
            treasure = maker.new(type="o", name=treasure_name)
            room.add(treasure)

        # 3. Boss and final treasure
        boss_key = maker.new(type="k", name="boss key")
        self.rng.choice(rooms[len(rooms) // 2 : -1]).add(boss_key)

        # Boss door
        boss_path = self._find_or_create_path(maker, rooms[-2], boss_room)
        if boss_path:
            boss_door = maker.new_door(boss_path, name="boss door")
            maker.add_fact("locked", boss_door)
            maker.add_fact("match", boss_key, boss_door)

        # Final treasure
        legendary_sword = maker.new(type="o", name="legendary sword")
        boss_room.add(legendary_sword)

        # Create quest
        quest_commands = [
            "look",
            "take iron key",
            "unlock iron door with iron key",
            "go north",
            "take ruby",
            "take boss key",
            "go south",
            "go east",
            "unlock boss door with boss key",
            "go east",
            "take legendary sword",
        ]

        maker.set_walkthrough(quest_commands)

        # Add dungeon atmosphere objects
        atmosphere_objects = ["torch", "skeleton", "cobwebs", "rusty chains"]
        for _ in range(params["nb_objects"] // 2):
            obj_name = self.rng.choice(atmosphere_objects)
            obj = maker.new(type="o", name=obj_name)
            self.rng.choice(rooms).add(obj)

        # Build and compile
        game = maker.build()

        options = GameOptions()
        seed_value = self.rng.randint(0, 65535)

        options.seeds = seed_value

        if filename_prefix is None:
            filename_prefix = "mixed_dungeon_crawler"

        game_filename = f"{filename_prefix}_{difficulty}_seed{seed_value}.z8"
        options.path = f"{output_folder}/{game_filename}"

        game_file = compile_game_with_retry(game, options)

        if game_file:
            config = {
                "type": "mixed",
                "mixed_type": "dungeon_crawler",
                "difficulty": difficulty,
                "description": template["description"],
                "nb_rooms": len(rooms),
                "nb_treasures": len(treasure_rooms) + 1,
                "has_boss_room": True,
                "features": template["features"],
                "seed": seed_value,
            }
            return game_file, config

        return None, {}

    def _generate_adventure(
        self,
        params: Dict[str, Any],
        template: Dict[str, Any],
        difficulty: str,
        output_folder: str,
        filename_prefix: Optional[str],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a full adventure game."""
        # For the most complex type, use the standard generator with maximum features
        options = GameOptions()
        seed_value = self.rng.randint(0, 65535)

        options.seeds = seed_value
        options.nb_rooms = params["nb_rooms"]
        options.nb_objects = params["nb_objects"]
        options.nb_parallel_quests = 4
        options.quest_length = params["quest_length"]
        options.quest_breadth = params["quest_breadth"]
        options.chaining.max_depth = params["quest_length"]
        options.chaining.max_breadth = params["quest_breadth"]
        options.grammar.theme = "house"
        options.grammar.include_adj = True

        # Generate the game
        game = make_game(options)

        # Compile the game
        if filename_prefix is None:
            filename_prefix = "mixed_adventure"

        game_filename = f"{filename_prefix}_{difficulty}_seed{seed_value}.z8"
        options.path = f"{output_folder}/{game_filename}"

        game_file = compile_game_with_retry(game, options)

        if game_file:
            config = {
                "type": "mixed",
                "mixed_type": "adventure",
                "difficulty": difficulty,
                "description": template["description"],
                "nb_rooms": params["nb_rooms"],
                "nb_objects": params["nb_objects"],
                "nb_parallel_quests": 4,
                "quest_complexity": "high",
                "features": template["features"],
                "seed": seed_value,
            }
            return game_file, config

        return None, {}

    def _create_room_layout(
        self, maker: GameMaker, nb_rooms: int, layout_type: str
    ) -> List[Any]:
        """Create a room layout of the specified type."""
        rooms = []

        if layout_type == "linear_branching":
            # Main path with side branches
            for i in range(nb_rooms):
                room = maker.new_room(f"room_{i}")
                rooms.append(room)

                if i > 0:
                    # Connect to previous room
                    maker.connect(rooms[i - 1].exits["east"], room.exits["west"])

                    # Add side branches
                    if i % 3 == 0 and i < nb_rooms - 1:
                        side_room = maker.new_room(f"side_room_{i}")
                        rooms.append(side_room)
                        maker.connect(room.exits["north"], side_room.exits["south"])

        elif layout_type == "maze":
            # Grid-like maze
            grid_size = int(nb_rooms**0.5) + 1
            for x in range(grid_size):
                for y in range(grid_size):
                    if len(rooms) < nb_rooms:
                        room = maker.new_room(f"maze_room_{x}_{y}")
                        rooms.append(room)

                        # Connect to adjacent rooms
                        if x > 0 and self.rng.random() > 0.3:
                            west_room = rooms[(x - 1) * grid_size + y]
                            maker.connect(west_room.exits["east"], room.exits["west"])
                        if y > 0 and self.rng.random() > 0.3:
                            south_room = rooms[x * grid_size + (y - 1)]
                            maker.connect(
                                south_room.exits["north"], room.exits["south"]
                            )

        elif layout_type == "dungeon":
            # Multiple connected sections
            section_size = nb_rooms // 3
            for section in range(3):
                for i in range(section_size):
                    if len(rooms) < nb_rooms:
                        room = maker.new_room(f"dungeon_level{section}_room{i}")
                        rooms.append(room)

                        if i > 0:
                            maker.connect(rooms[-2].exits["east"], room.exits["west"])
                        elif section > 0:
                            # Connect sections
                            prev_section_last = rooms[section * section_size - 1]
                            maker.connect(
                                prev_section_last.exits["north"], room.exits["south"]
                            )
        else:
            # Default linear
            for i in range(nb_rooms):
                room = maker.new_room(f"room_{i}")
                rooms.append(room)
                if i > 0:
                    maker.connect(rooms[i - 1].exits["east"], room.exits["west"])

        return rooms

    def _find_or_create_path(
        self, maker: GameMaker, room1: Any, room2: Any
    ) -> Optional[Any]:
        """Find existing path between rooms or create one."""
        # Try to find existing path
        path = maker.find_path(room1, room2)
        if path:
            return path

        # Try to create new path
        directions = ["north", "south", "east", "west"]
        self.rng.shuffle(directions)

        for direction in directions:
            opposite = {
                "north": "south",
                "south": "north",
                "east": "west",
                "west": "east",
            }[direction]
            try:
                path = maker.connect(room1.exits[direction], room2.exits[opposite])
                return path
            except:
                continue

        return None

    def _get_difficulty_params(self, difficulty: str) -> Dict[str, Any]:
        """Get game parameters based on difficulty level."""
        difficulty_settings = {
            "easy": {
                "nb_rooms": 8,
                "nb_objects": 10,
                "quest_length": 5,
                "quest_breadth": 2,
                "nb_parallel_quests": 2,
            },
            "medium": {
                "nb_rooms": 12,
                "nb_objects": 15,
                "quest_length": 8,
                "quest_breadth": 3,
                "nb_parallel_quests": 3,
            },
            "hard": {
                "nb_rooms": 18,
                "nb_objects": 20,
                "quest_length": 12,
                "quest_breadth": 4,
                "nb_parallel_quests": 4,
            },
            "expert": {
                "nb_rooms": 25,
                "nb_objects": 30,
                "quest_length": 20,
                "quest_breadth": 5,
                "nb_parallel_quests": 5,
            },
        }

        return difficulty_settings.get(difficulty, difficulty_settings["medium"])


if __name__ == "__main__":
    """Test the mixed generator."""
    import os

    # Create output directory
    os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)

    # Create generator
    generator = MixedGenerator(seed=42)

    # Test different mixed types
    for mixed_type in ["quest_puzzle", "navigation_puzzle", "dungeon_crawler"]:
        print(f"\nGenerating {mixed_type} (medium)...")
        game_file, config = generator.generate(
            difficulty="medium", mixed_type=mixed_type
        )

        if game_file:
            print(f"SUCCESS: {game_file}")
            print(f"Config: {config}")
        else:
            print("FAILED")
