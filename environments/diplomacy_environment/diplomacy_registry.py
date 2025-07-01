"""
Scenario registry for Diplomacy environment.

Provides varied starting positions and game configurations
to prevent overfitting and encourage robust strategies.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


class DiplomacyScenarioRegistry:
    """
    Registry of Diplomacy game scenarios.
    
    Includes:
    - Standard starting positions
    - Historical scenarios
    - Balanced custom starts
    - Random configurations
    """
    
    def __init__(
        self,
        scenarios_path: Optional[Path] = None,
        distribution: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        self.scenarios_path = scenarios_path or Path(__file__).parent / "scenarios"
        self.distribution = distribution or {
            "standard": 0.4,
            "historical": 0.3,
            "balanced": 0.2,
            "random": 0.1,
        }
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load scenarios
        self.scenarios = {
            "standard": self._get_standard_scenarios(),
            "historical": self._get_historical_scenarios(),
            "balanced": self._get_balanced_scenarios(),
            "random": [],  # Generated on demand
        }
    
    def get_scenario(self, scenario_type: Optional[str] = None) -> Dict[str, Any]:
        """Get a scenario configuration."""
        if scenario_type is None:
            # Sample from distribution
            types = list(self.distribution.keys())
            weights = list(self.distribution.values())
            scenario_type = np.random.choice(types, p=weights)
        
        if scenario_type == "random":
            return self._generate_random_scenario()
        
        scenarios = self.scenarios.get(scenario_type, [])
        if not scenarios:
            # Fallback to standard
            scenarios = self.scenarios["standard"]
        
        return random.choice(scenarios)
    
    def _get_standard_scenarios(self) -> List[Dict[str, Any]]:
        """Get standard Diplomacy starting positions."""
        return [
            {
                "name": "Classic 1901",
                "variant": "standard",
                "starting_position": None,  # Use default
                "description": "Standard Diplomacy starting position",
                "difficulty": "medium",
                "tags": ["classic", "balanced"],
            },
            {
                "name": "Fleet Rome",
                "variant": "standard",
                "starting_position": {
                    "modifications": {
                        "ITALY": {
                            "units": {
                                "remove": ["A ROM"],
                                "add": ["F ROM"],
                            }
                        }
                    }
                },
                "description": "Italy starts with Fleet Rome instead of Army Rome",
                "difficulty": "medium",
                "tags": ["variant", "italy_focused"],
            },
            {
                "name": "Army Trieste",
                "variant": "standard",
                "starting_position": {
                    "modifications": {
                        "AUSTRIA": {
                            "units": {
                                "remove": ["F TRI"],
                                "add": ["A TRI"],
                            }
                        }
                    }
                },
                "description": "Austria starts with Army Trieste instead of Fleet",
                "difficulty": "medium",
                "tags": ["variant", "austria_focused"],
            },
        ]
    
    def _get_historical_scenarios(self) -> List[Dict[str, Any]]:
        """Get historical scenario configurations."""
        return [
            {
                "name": "1902 After Sealion",
                "variant": "standard",
                "starting_position": {
                    "year": 1902,
                    "phase": "spring_orders",
                    "territories": {
                        "LON": {"owner": "GERMANY"},
                        "EDI": {"owner": "GERMANY"},
                        "LVP": {"owner": "ENGLAND"},
                        # ... more territory assignments
                    },
                    "units": {
                        "GERMANY": ["F LON", "F NTH", "A EDI", "A MUN", "A BER"],
                        "ENGLAND": ["F LVP", "F NWG"],
                        "FRANCE": ["F BRE", "A PAR", "A MAR", "F MAO"],
                        # ... more unit positions
                    },
                },
                "description": "After successful German invasion of England",
                "difficulty": "hard",
                "tags": ["historical", "imbalanced", "germany_strong"],
            },
            {
                "name": "1903 Western Triple",
                "variant": "standard",
                "starting_position": {
                    "year": 1903,
                    "phase": "spring_orders",
                    "territories": {
                        # EFG alliance against the rest
                        "MUN": {"owner": "ENGLAND"},
                        "VIE": {"owner": "FRANCE"},
                        "WAR": {"owner": "GERMANY"},
                        # ... more territory assignments
                    },
                    "units": {
                        "ENGLAND": ["F NTH", "F BAL", "A MUN", "F LON", "A EDI"],
                        "FRANCE": ["A VIE", "A TYR", "F MAR", "A PAR", "F BRE"],
                        "GERMANY": ["A WAR", "A SIL", "A BER", "F KIE", "A RUH"],
                        # ... more unit positions
                    },
                },
                "description": "England, France, Germany alliance dominates",
                "difficulty": "expert",
                "tags": ["historical", "alliance", "late_game"],
            },
            {
                "name": "1904 Juggernaut",
                "variant": "standard",
                "starting_position": {
                    "year": 1904,
                    "phase": "spring_orders",
                    "territories": {
                        # Russia-Turkey alliance scenario
                        "SER": {"owner": "TURKEY"},
                        "RUM": {"owner": "RUSSIA"},
                        "BUD": {"owner": "RUSSIA"},
                        "VIE": {"owner": "TURKEY"},
                        # ... more territory assignments
                    },
                    "units": {
                        "RUSSIA": ["A BUD", "F RUM", "A GAL", "A WAR", "F SEV", "F BAL"],
                        "TURKEY": ["A VIE", "A SER", "F BUL", "A CON", "F BLA", "F AEG"],
                        # ... more unit positions
                    },
                },
                "description": "Russia-Turkey alliance threatens Europe",
                "difficulty": "hard",
                "tags": ["historical", "alliance", "juggernaut"],
            },
        ]
    
    def _get_balanced_scenarios(self) -> List[Dict[str, Any]]:
        """Get balanced custom starting positions."""
        return [
            {
                "name": "Central Powers",
                "variant": "standard",
                "starting_position": {
                    "modifications": {
                        # Give central powers slight advantage
                        "GERMANY": {"extra_unit": "A BOH"},
                        "AUSTRIA": {"extra_unit": "A GAL"},
                    }
                },
                "description": "Central powers start with extra armies",
                "difficulty": "medium",
                "tags": ["balanced", "central_focused"],
            },
            {
                "name": "Naval Emphasis",
                "variant": "standard",
                "starting_position": {
                    "modifications": {
                        # Convert some armies to fleets
                        "RUSSIA": {
                            "units": {
                                "remove": ["A WAR"],
                                "add": ["F BAL"],
                            }
                        },
                        "TURKEY": {
                            "units": {
                                "remove": ["A SMY"],
                                "add": ["F SMY"],
                            }
                        },
                    }
                },
                "description": "More fleets for naval-focused gameplay",
                "difficulty": "medium",
                "tags": ["balanced", "naval"],
            },
            {
                "name": "Diplomatic Start",
                "variant": "standard",
                "starting_position": {
                    "neutrals": ["BEL", "HOL", "SER", "BUL", "GRE"],
                    "description": "Some territories start neutral",
                },
                "description": "Neutral territories create diplomatic opportunities",
                "difficulty": "hard",
                "tags": ["balanced", "diplomatic"],
            },
        ]
    
    def _generate_random_scenario(self) -> Dict[str, Any]:
        """Generate a random scenario configuration."""
        # Random modifications to standard start
        modifications = {}
        
        # Randomly swap some unit types
        powers = ["ENGLAND", "FRANCE", "GERMANY", "ITALY", "AUSTRIA", "RUSSIA", "TURKEY"]
        for power in powers:
            if random.random() < 0.3:  # 30% chance to modify
                modifications[power] = self._generate_random_modification(power)
        
        # Random year start (mostly 1901, sometimes later)
        year = 1901
        if random.random() < 0.1:  # 10% chance for later start
            year = random.choice([1902, 1903])
        
        scenario = {
            "name": f"Random_{random.randint(1000, 9999)}",
            "variant": "standard",
            "starting_position": {
                "year": year,
                "modifications": modifications,
            },
            "description": "Randomly generated starting position",
            "difficulty": "random",
            "tags": ["random", "generated"],
        }
        
        return scenario
    
    def _generate_random_modification(self, power: str) -> Dict[str, Any]:
        """Generate random modifications for a power."""
        mod_types = ["swap_unit_type", "different_position", "extra_build"]
        mod_type = random.choice(mod_types)
        
        if mod_type == "swap_unit_type":
            # Swap an army for a fleet or vice versa
            return {
                "units": {
                    "swap_types": True,
                }
            }
        elif mod_type == "different_position":
            # Start unit in different home center
            return {
                "units": {
                    "shuffle_positions": True,
                }
            }
        else:  # extra_build
            # Start with an extra unit
            return {
                "extra_unit": "random",
            }
    
    def add_custom_scenario(self, scenario: Dict[str, Any], category: str = "custom") -> None:
        """Add a custom scenario to the registry."""
        if category not in self.scenarios:
            self.scenarios[category] = []
        
        self.scenarios[category].append(scenario)
    
    def save_scenario(self, scenario: Dict[str, Any], filename: str) -> None:
        """Save a scenario to file."""
        filepath = self.scenarios_path / f"{filename}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(scenario, f, indent=2)
    
    def load_scenario(self, filename: str) -> Dict[str, Any]:
        """Load a scenario from file."""
        filepath = self.scenarios_path / f"{filename}.json"
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_scenario_by_tags(self, tags: List[str]) -> Optional[Dict[str, Any]]:
        """Get a scenario matching specific tags."""
        matching_scenarios = []
        
        for category_scenarios in self.scenarios.values():
            for scenario in category_scenarios:
                scenario_tags = scenario.get("tags", [])
                if all(tag in scenario_tags for tag in tags):
                    matching_scenarios.append(scenario)
        
        if matching_scenarios:
            return random.choice(matching_scenarios)
        
        return None
    
    def get_scenarios_for_curriculum(
        self,
        difficulty_progression: List[str]
    ) -> List[Dict[str, Any]]:
        """Get scenarios following a difficulty progression."""
        curriculum = []
        
        for difficulty in difficulty_progression:
            # Find scenarios matching difficulty
            matching = []
            for category_scenarios in self.scenarios.values():
                for scenario in category_scenarios:
                    if scenario.get("difficulty") == difficulty:
                        matching.append(scenario)
            
            if matching:
                curriculum.append(random.choice(matching))
            else:
                # Fallback to any scenario
                curriculum.append(self.get_scenario())
        
        return curriculum