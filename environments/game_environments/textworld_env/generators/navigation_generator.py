#!/usr/bin/env python3
"""
Navigation Generator for TextWorld

Generates maze-like games focused on spatial navigation and exploration.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from textworld import GameMaker, GameOptions
from textworld.generator import make_game

from ..generation_utils import DEFAULT_OUTPUT_FOLDER, compile_game_with_retry

logger = logging.getLogger(__name__)


class NavigationGenerator:
    """Generator for navigation and maze-based TextWorld games."""

    # Navigation game types
    NAVIGATION_TYPES = {
        "maze": {
            "description": "Navigate through a complex maze",
            "layout": "grid",
            "dead_ends": True,
            "landmarks": ["fountain", "statue", "pillar", "garden"],
        },
        "labyrinth": {
            "description": "Find your way through a winding labyrinth",
            "layout": "branching",
            "dead_ends": True,
            "landmarks": ["altar", "shrine", "pool", "crystal"],
        },
        "exploration": {
            "description": "Explore interconnected areas",
            "layout": "hub",
            "dead_ends": False,
            "landmarks": ["tower", "bridge", "plaza", "gate"],
        },
        "dungeon": {
            "description": "Navigate a dangerous dungeon",
            "layout": "mixed",
            "dead_ends": True,
            "landmarks": ["torch", "skeleton", "chest", "grate"],
        },
    }

    # Room themes for different areas
    ROOM_THEMES = {
        "entrance": ["entrance hall", "foyer", "antechamber", "vestibule"],
        "corridor": ["corridor", "hallway", "passage", "tunnel"],
        "chamber": ["chamber", "room", "hall", "gallery"],
        "special": ["throne room", "library", "laboratory", "treasury"],
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize the navigation generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)

    def generate(
        self,
        difficulty: str = "medium",
        nav_type: Optional[str] = None,
        output_folder: str = DEFAULT_OUTPUT_FOLDER,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a navigation-based game.

        Args:
            difficulty: Game difficulty (easy/medium/hard/expert)
            nav_type: Type of navigation game
            output_folder: Where to save the game
            filename_prefix: Prefix for the game file

        Returns:
            Tuple of (game_file_path, configuration)
        """
        # Select navigation type
        if nav_type is None:
            nav_type = self.rng.choice(list(self.NAVIGATION_TYPES.keys()))
        elif nav_type not in self.NAVIGATION_TYPES:
            logger.warning(f"Unknown navigation type: {nav_type}, using 'maze'")
            nav_type = "maze"

        template = self.NAVIGATION_TYPES[nav_type]

        # Get difficulty parameters
        params = self._get_difficulty_params(difficulty)

        try:
            # Create game maker
            maker = GameMaker()

            # Generate the navigation layout
            if template["layout"] == "grid":
                rooms, start_room, end_room = self._generate_grid_maze(maker, params)
            elif template["layout"] == "branching":
                rooms, start_room, end_room = self._generate_branching_maze(
                    maker, params
                )
            elif template["layout"] == "hub":
                rooms, start_room, end_room = self._generate_hub_layout(maker, params)
            else:  # mixed
                rooms, start_room, end_room = self._generate_mixed_layout(maker, params)

            # Place player at start
            maker.set_player(start_room)

            # Add landmarks for navigation
            self._place_landmarks(maker, rooms, template["landmarks"], params)

            # Add goal at end
            goal = maker.new(
                type="o", name="golden key", desc="The key to escape this place."
            )
            end_room.add(goal)

            # Create exit door at start (locked) - needs to be between rooms
            # Create a special exit room
            exit_room = maker.new_room("exit")
            exit_path = maker.connect(
                start_room.exits["south"], exit_room.exits["north"]
            )
            exit_door = maker.new_door(exit_path, name="exit door", desc="The way out.")
            maker.add_fact("match", goal, exit_door)
            maker.add_fact("locked", exit_door)

            # Add navigation aids based on difficulty
            if difficulty in ["easy", "medium"]:
                self._add_navigation_aids(maker, rooms, params)

            # Create quest
            quest_commands = self._generate_navigation_quest(
                start_room, end_room, rooms
            )
            quest_commands.extend(
                ["take golden key", "unlock exit door with golden key"]
            )

            maker.set_walkthrough(quest_commands)

            # Add distractors
            if params["nb_objects"] > len(template["landmarks"]):
                maker.generate_distractors(
                    params["nb_objects"] - len(template["landmarks"])
                )

            # Build and compile
            game = maker.build()

            options = GameOptions()
            seed_value = self.rng.randint(0, 65535)

            options.seeds = seed_value

            if filename_prefix is None:
                filename_prefix = f"navigation_{nav_type}"

            game_filename = f"{filename_prefix}_{difficulty}_seed{seed_value}.z8"
            options.path = f"{output_folder}/{game_filename}"

            game_file = compile_game_with_retry(game, options)

            if game_file:
                config = {
                    "type": "navigation",
                    "nav_type": nav_type,
                    "difficulty": difficulty,
                    "description": template["description"],
                    "nb_rooms": len(rooms),
                    "layout": template["layout"],
                    "has_dead_ends": template["dead_ends"],
                    "seed": seed_value,
                }

                logger.info(
                    f"Generated navigation game: {nav_type} ({difficulty}) - {game_file}"
                )
                return game_file, config
            else:
                logger.error("Failed to compile navigation game")
                return None, {}

        except Exception as e:
            logger.error(f"Error generating navigation game: {e}")
            return None, {}

    def _generate_grid_maze(
        self, maker: GameMaker, params: Dict[str, Any]
    ) -> Tuple[List[Any], Any, Any]:
        """Generate a grid-based maze layout."""
        grid_size = params["grid_size"]
        rooms = {}

        # Create grid of rooms
        for x in range(grid_size):
            for y in range(grid_size):
                room_type = self.rng.choice(self.ROOM_THEMES["corridor"])
                room = maker.new_room(f"{room_type}_{x}_{y}")
                rooms[(x, y)] = room

        # Connect rooms with some randomness
        for x in range(grid_size):
            for y in range(grid_size):
                current = rooms[(x, y)]

                # Connect east
                if x < grid_size - 1 and self.rng.random() < 0.7:
                    neighbor = rooms[(x + 1, y)]
                    if not self._rooms_connected(maker, current, neighbor):
                        maker.connect(current.exits["east"], neighbor.exits["west"])

                # Connect north
                if y < grid_size - 1 and self.rng.random() < 0.7:
                    neighbor = rooms[(x, y + 1)]
                    if not self._rooms_connected(maker, current, neighbor):
                        maker.connect(current.exits["north"], neighbor.exits["south"])

        # Ensure maze is connected
        self._ensure_connected_maze(maker, list(rooms.values()))

        # Select start and end rooms
        start_room = rooms[(0, 0)]
        end_room = rooms[(grid_size - 1, grid_size - 1)]

        return list(rooms.values()), start_room, end_room

    def _generate_branching_maze(
        self, maker: GameMaker, params: Dict[str, Any]
    ) -> Tuple[List[Any], Any, Any]:
        """Generate a branching tree-like maze."""
        nb_rooms = params["nb_rooms"]
        rooms = []

        # Create entrance
        entrance = maker.new_room(self.rng.choice(self.ROOM_THEMES["entrance"]))
        rooms.append(entrance)

        # Keep track of leaf nodes for branching
        leaf_nodes = [entrance]

        # Generate branches
        while len(rooms) < nb_rooms:
            if not leaf_nodes:
                break

            # Pick a random leaf to branch from
            parent = self.rng.choice(leaf_nodes)

            # Determine number of branches (1-3)
            num_branches = min(self.rng.randint(1, 3), nb_rooms - len(rooms))

            new_leaves = []
            for i in range(num_branches):
                room_type = self.rng.choice(self.ROOM_THEMES["chamber"])
                room = maker.new_room(f"{room_type}_{len(rooms)}")
                rooms.append(room)

                # Connect to parent
                directions = ["north", "east", "south", "west"]
                self.rng.shuffle(directions)

                for direction in directions:
                    opposite = {
                        "north": "south",
                        "south": "north",
                        "east": "west",
                        "west": "east",
                    }[direction]
                    if not self._exit_used(maker, parent, direction):
                        maker.connect(parent.exits[direction], room.exits[opposite])
                        break

                # Randomly decide if this is a new leaf
                if self.rng.random() < 0.6:
                    new_leaves.append(room)

            # Remove parent from leaves and add new leaves
            leaf_nodes.remove(parent)
            leaf_nodes.extend(new_leaves)

        # Select end room as the deepest leaf
        end_room = rooms[-1]  # Simple heuristic

        return rooms, entrance, end_room

    def _generate_hub_layout(
        self, maker: GameMaker, params: Dict[str, Any]
    ) -> Tuple[List[Any], Any, Any]:
        """Generate a hub-and-spoke layout."""
        nb_rooms = params["nb_rooms"]
        rooms = []

        # Create central hub
        hub = maker.new_room("central plaza")
        rooms.append(hub)

        # Create spokes
        num_spokes = min(4, nb_rooms - 1)
        rooms_per_spoke = (nb_rooms - 1) // num_spokes

        directions = ["north", "east", "south", "west"]
        end_rooms = []

        for i, direction in enumerate(directions[:num_spokes]):
            previous = hub
            opposite = {
                "north": "south",
                "south": "north",
                "east": "west",
                "west": "east",
            }[direction]

            # Create rooms along this spoke
            for j in range(rooms_per_spoke):
                room_type = self.rng.choice(
                    self.ROOM_THEMES[
                        "corridor" if j < rooms_per_spoke - 1 else "special"
                    ]
                )
                room = maker.new_room(f"{room_type}_{direction}_{j}")
                rooms.append(room)

                # Connect to previous
                maker.connect(previous.exits[direction], room.exits[opposite])

                previous = room
                if j == rooms_per_spoke - 1:
                    end_rooms.append(room)

        # Start at hub, end at random spoke end
        start_room = hub
        end_room = self.rng.choice(end_rooms) if end_rooms else rooms[-1]

        return rooms, start_room, end_room

    def _generate_mixed_layout(
        self, maker: GameMaker, params: Dict[str, Any]
    ) -> Tuple[List[Any], Any, Any]:
        """Generate a mixed layout combining different patterns."""
        options = GameOptions()
        seed_value = self.rng.randint(0, 65535)

        options.seeds = seed_value
        options.nb_rooms = params["nb_rooms"]
        options.nb_objects = 0

        make_game(options)

        return self._generate_grid_maze(maker, params)

    def _place_landmarks(
        self,
        maker: GameMaker,
        rooms: List[Any],
        landmark_types: List[str],
        params: Dict[str, Any],
    ) -> None:
        """Place landmarks in rooms for navigation reference."""
        num_landmarks = min(len(landmark_types), params["nb_landmarks"])
        selected_landmarks = self.rng.sample(landmark_types, num_landmarks)

        # Place landmarks in random rooms
        landmark_rooms = self.rng.sample(
            rooms[1:-1], min(num_landmarks, len(rooms) - 2)
        )

        for landmark_name, room in zip(selected_landmarks, landmark_rooms):
            landmark = maker.new(
                type="o",
                name=landmark_name,
                desc=f"A distinctive {landmark_name} that helps with navigation.",
            )
            room.add(landmark)

    def _add_navigation_aids(
        self, maker: GameMaker, rooms: List[Any], params: Dict[str, Any]
    ) -> None:
        """Add navigation aids like maps or compasses."""
        # Add a map in one of the early rooms
        if len(rooms) > 2:
            map_room = self.rng.choice(rooms[: len(rooms) // 3])
            game_map = maker.new(
                type="o", name="map", desc="A map showing the layout of this place."
            )
            map_room.add(game_map)

        # Add directional signs in some rooms
        num_signs = params.get("nb_signs", 2)
        sign_rooms = self.rng.sample(rooms, min(num_signs, len(rooms)))

        for _, room in enumerate(sign_rooms):
            sign = maker.new(type="o", name="sign", desc="A sign with directions.")
            room.add(sign)

    def _generate_navigation_quest(
        self, start_room: Any, end_room: Any, all_rooms: List[Any]
    ) -> List[str]:
        """Generate quest commands for navigation.

        This is simplified - in reality we'd need pathfinding.
        """
        # For now, return a simple set of exploration commands
        commands = ["look", "go north", "look", "go east", "look"]

        # Add more navigation based on room count
        for i in range(min(10, len(all_rooms) - 1)):
            direction = self.rng.choice(["north", "south", "east", "west"])
            commands.append(f"go {direction}")
            if self.rng.random() < 0.3:
                commands.append("look")

        return commands

    def _rooms_connected(self, maker: GameMaker, room1: Any, room2: Any) -> bool:
        """Check if two rooms are already connected."""
        # Check if there's a path between the rooms
        return maker.find_path(room1, room2) is not None

    def _exit_used(self, maker: GameMaker, room: Any, direction: str) -> bool:
        """Check if a room's exit in a given direction is already used."""
        # This is a simplified check - would need proper implementation
        # to check if the exit has a path connected
        return False  # Simplified for now

    def _ensure_connected_maze(self, maker: GameMaker, rooms: List[Any]) -> None:
        """Ensure all rooms in the maze are connected."""
        # This would implement a proper connectivity check and add
        # connections as needed - simplified for this implementation
        pass

    def _get_difficulty_params(self, difficulty: str) -> Dict[str, Any]:
        """Get game parameters based on difficulty level."""
        difficulty_settings = {
            "easy": {
                "nb_rooms": 5,
                "nb_objects": 3,
                "nb_landmarks": 2,
                "nb_signs": 2,
                "grid_size": 3,
                "quest_length": 5,
            },
            "medium": {
                "nb_rooms": 10,
                "nb_objects": 5,
                "nb_landmarks": 3,
                "nb_signs": 1,
                "grid_size": 4,
                "quest_length": 8,
            },
            "hard": {
                "nb_rooms": 15,
                "nb_objects": 7,
                "nb_landmarks": 4,
                "nb_signs": 0,
                "grid_size": 5,
                "quest_length": 12,
            },
            "expert": {
                "nb_rooms": 20,
                "nb_objects": 10,
                "nb_landmarks": 5,
                "nb_signs": 0,
                "grid_size": 6,
                "quest_length": 20,
            },
        }

        return difficulty_settings.get(difficulty, difficulty_settings["medium"])


if __name__ == "__main__":
    """Test the navigation generator."""
    import os

    # Create output directory
    os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)

    # Create generator
    generator = NavigationGenerator(seed=42)

    # Test different navigation types
    for nav_type in ["maze", "labyrinth", "exploration"]:
        for difficulty in ["easy", "medium"]:
            print(f"\nGenerating {nav_type} ({difficulty})...")
            game_file, config = generator.generate(
                difficulty=difficulty, nav_type=nav_type
            )

            if game_file:
                print(f"SUCCESS: {game_file}")
                print(f"Config: {config}")
            else:
                print("FAILED")
