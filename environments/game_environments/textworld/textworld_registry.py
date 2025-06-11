#!/usr/bin/env python3
"""
TextWorld Environment Registry

Provides a registry system for managing both pre-built TextWorld challenges
and dynamically generated games.
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import textworld
import textworld.challenges

from .generation_utils import generate_textworld_game
from .generators import QuestGenerator, PuzzleGenerator, NavigationGenerator, MixedGenerator

logger = logging.getLogger(__name__)


class TextWorldChallengeRegistry:
    """Registry for pre-built TextWorld challenges."""
    
    # Pre-built challenges with their default settings
    CHALLENGES = {
        "tw-simple": {
            "rewards": "balanced",
            "goal": "detailed",
            "test": False
        },
        "tw-cooking": {
            "recipe": 1,
            "take": 1,
            "cook": False,
            "open": False,
            "drop": False,
            "go": 6
        },
        "tw-coin_collector": {
            "level": 1
        },
        "tw-treasure_hunter": {
            "level": 1
        }
    }
    
    # Difficulty mapping for pre-built challenges
    DIFFICULTY_MAPPING = {
        "tw-simple": "easy",
        "tw-cooking": "medium",
        "tw-coin_collector": "easy",
        "tw-treasure_hunter": "medium"
    }
    
    def __init__(self):
        self._challenges = self.CHALLENGES.copy()
        
    def list_challenges(self) -> List[str]:
        """List all available pre-built challenges."""
        return list(self._challenges.keys())
    
    def get_challenge(self, name: str) -> Tuple[str, Dict[str, Any]]:
        """Get challenge name and default settings."""
        if name not in self._challenges:
            raise ValueError(f"Unknown challenge: {name}. Available: {self.list_challenges()}")
        return name, self._challenges[name].copy()
    
    def get_random_challenge(self, difficulty: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Get a random challenge, optionally filtered by difficulty."""
        if difficulty:
            # Filter challenges by difficulty
            filtered = [
                name for name, diff in self.DIFFICULTY_MAPPING.items()
                if diff == difficulty
            ]
            if not filtered:
                logger.warning(f"No challenges found for difficulty {difficulty}, using all")
                filtered = list(self._challenges.keys())
        else:
            filtered = list(self._challenges.keys())
            
        name = random.choice(filtered)
        return self.get_challenge(name)


class TextWorldGenerator:
    """Base class for generating custom TextWorld games."""
    
    DIFFICULTY_SETTINGS = {
        "easy": {
            "n_rooms": 3,
            "n_objects": 5,
            "quest_length": 3,
            "quest_breadth": 1
        },
        "medium": {
            "n_rooms": 6,
            "n_objects": 10,
            "quest_length": 5,
            "quest_breadth": 2
        },
        "hard": {
            "n_rooms": 10,
            "n_objects": 15,
            "quest_length": 8,
            "quest_breadth": 3
        },
        "expert": {
            "n_rooms": 15,
            "n_objects": 20,
            "quest_length": 12,
            "quest_breadth": 4
        }
    }
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.quest_gen = QuestGenerator(seed)
        self.puzzle_gen = PuzzleGenerator(seed)
        self.nav_gen = NavigationGenerator(seed)
        self.mixed_gen = MixedGenerator(seed)
        
    def generate_game(self, 
                     game_type: str = "quest",
                     difficulty: str = "medium",
                     output_folder: str = "environments/game_environments/textworld/tw_generated_games"
                     ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate a game of specified type and difficulty.
        
        Returns:
            Tuple of (game_file_path, game_config)
        """
        # Use specialized generators based on game type
        if game_type == "quest":
            return self.quest_gen.generate(
                difficulty=difficulty,
                output_folder=output_folder
            )
        elif game_type == "puzzle":
            return self.puzzle_gen.generate(
                difficulty=difficulty,
                output_folder=output_folder
            )
        elif game_type == "navigation":
            return self.nav_gen.generate(
                difficulty=difficulty,
                output_folder=output_folder
            )
        elif game_type == "mixed":
            return self.mixed_gen.generate(
                difficulty=difficulty,
                output_folder=output_folder
            )
        else:
            # Fallback to basic generation
            logger.warning(f"Unknown game type: {game_type}, using quest generator")
            return self.quest_gen.generate(
                difficulty=difficulty,
                output_folder=output_folder
            )


class TextWorldEnvironmentRegistry:
    """Main registry combining pre-built and generated games."""
    
    def __init__(self, 
                 generation_ratio: float = 0.7,
                 cache_size: int = 50,
                 seed: Optional[int] = None):
        """
        Args:
            generation_ratio: Ratio of generated games vs pre-built (0.0 to 1.0)
            cache_size: Maximum number of game configurations to cache
            seed: Random seed for reproducibility
        """
        self.challenge_registry = TextWorldChallengeRegistry()
        self.generator = TextWorldGenerator(seed)
        self.generation_ratio = generation_ratio
        self.cache_size = cache_size
        self.rng = random.Random(seed)
        
        # LRU cache for game configurations
        self._game_cache = OrderedDict()
        
        # Track generated files for cleanup
        self._generated_files = set()
        
    def get_environment(self,
                       mode: str = "random",
                       difficulty: Optional[str] = None,
                       game_type: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Get a game file path and configuration.
        
        Args:
            mode: Selection mode - "random", "generated", "challenge"
            difficulty: Difficulty level - "easy", "medium", "hard", "expert", "random"
            game_type: Type of game - "quest", "puzzle", "navigation", "mixed"
            
        Returns:
            Tuple of (game_file_path, configuration)
        """
        # Handle difficulty randomization
        if difficulty == "random":
            difficulty = self.rng.choice(["easy", "medium", "hard", "expert"])
        
        # Determine whether to use generated or pre-built
        if mode == "random":
            use_generated = self.rng.random() < self.generation_ratio
        elif mode == "generated":
            use_generated = True
        elif mode == "challenge":
            use_generated = False
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if use_generated:
            # Generate a new game
            if game_type is None:
                game_type = self.rng.choice(["quest", "puzzle", "navigation", "mixed"])
            
            game_file, config = self.generator.generate_game(
                game_type=game_type,
                difficulty=difficulty or "medium"
            )
            
            if game_file:
                self._generated_files.add(game_file)
                config["source"] = "generated"
                return game_file, config
        
        # Fall back to pre-built challenge
        challenge_name, settings = self.challenge_registry.get_random_challenge(difficulty)
        
        # Generate the challenge
        game_file, _ = generate_textworld_game(
            challenge_name,
            settings,
            filename_prefix=f"challenge_{challenge_name}"
        )
        
        if game_file:
            self._generated_files.add(game_file)
            config = {
                "source": "challenge",
                "challenge_name": challenge_name,
                "settings": settings,
                "difficulty": self.challenge_registry.DIFFICULTY_MAPPING.get(challenge_name, "medium")
            }
            return game_file, config
            
        return None, {}
    
    def cleanup_game_file(self, game_file: str):
        """Clean up a generated game file."""
        if game_file in self._generated_files:
            try:
                if os.path.exists(game_file):
                    os.remove(game_file)
                # Also remove the JSON file if it exists
                json_file = game_file.replace('.z8', '.json')
                if os.path.exists(json_file):
                    os.remove(json_file)
                self._generated_files.remove(game_file)
                logger.debug(f"Cleaned up game file: {game_file}")
            except OSError as e:
                logger.warning(f"Failed to clean up game file {game_file}: {e}")
    
    def cleanup_all(self):
        """Clean up all tracked generated files."""
        files_to_clean = list(self._generated_files)
        for game_file in files_to_clean:
            self.cleanup_game_file(game_file)


# Convenience function for creating a registry
def create_textworld_registry(**kwargs) -> TextWorldEnvironmentRegistry:
    """Create a TextWorld environment registry with given parameters."""
    return TextWorldEnvironmentRegistry(**kwargs)