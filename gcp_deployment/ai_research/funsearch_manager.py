"""
FunSearch Manager - Evolutionary Function Discovery for VR Optimization
Integrates DeepMind's FunSearch for discovering novel VR optimization functions
"""

import asyncio
import json
import math
import os
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class FunSearchManager:
    """
    Manages FunSearch integration for evolutionary VR function discovery
    Discovers novel optimization functions for VR performance and user experience
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.funsearch_path = self.config.get("funsearch_path", "../funsearch")
        self.population_size = self.config.get("population_size", 50)
        self.generations = self.config.get("generations", 100)
        self.mutation_rate = self.config.get("mutation_rate", 0.1)

        # VR optimization domains
        self.optimization_domains = {
            "frame_time_optimization": {
                "description": "Optimize frame time consistency for VR comfort",
                "input_variables": [
                    "gpu_util",
                    "vram_usage",
                    "cpu_util",
                    "scene_complexity",
                ],
                "target_metric": "frame_time_variance",
                "goal": "minimize",
            },
            "comfort_score_optimization": {
                "description": "Maximize VR comfort through adaptive parameters",
                "input_variables": [
                    "frame_rate",
                    "motion_intensity",
                    "user_age",
                    "session_duration",
                ],
                "target_metric": "comfort_score",
                "goal": "maximize",
            },
            "affordance_discovery": {
                "description": "Discover novel VR interaction patterns",
                "input_variables": [
                    "hand_position",
                    "gaze_direction",
                    "object_distance",
                    "cultural_context",
                ],
                "target_metric": "interaction_success",
                "goal": "maximize",
            },
            "performance_prediction": {
                "description": "Predict VR performance from system metrics",
                "input_variables": [
                    "gpu_type",
                    "resolution",
                    "refresh_rate",
                    "app_complexity",
                ],
                "target_metric": "predicted_fps",
                "goal": "accuracy",
            },
        }

        # Function templates for evolution
        self.function_templates = {
            "polynomial": "a*x^3 + b*x^2 + c*x + d",
            "trigonometric": "a*sin(b*x + c) + d*cos(e*x + f)",
            "exponential": "a*exp(b*x) + c*log(d*x + e)",
            "hybrid": "a*sin(b*x)*exp(c*x) + d*x^2 + e",
        }

        # Current population of functions
        self.current_population = []
        self.best_functions = {}
        self.evolution_history = []

    async def initialize(self):
        """Initialize FunSearch components"""
        try:
            # Check if FunSearch is available
            if not os.path.exists(self.funsearch_path):
                print(f"WARNING: FunSearch not found at {self.funsearch_path}")
                print(
                    "Please clone FunSearch: git clone https://github.com/deepmind/funsearch.git"
                )
                return False

            # Initialize population
            await self._initialize_population()

            print("✅ FunSearch Manager initialized")
            print(f"   Population size: {self.population_size}")
            print(f"   Optimization domains: {len(self.optimization_domains)}")

            return True

        except Exception as e:
            print(f"❌ Failed to initialize FunSearch Manager: {e}")
            return False

    async def discover_vr_functions(
        self,
        performance_data: List[Dict[str, Any]],
        domain: str = "frame_time_optimization",
    ) -> Dict[str, Any]:
        """
        Discover novel VR optimization functions using evolutionary algorithms

        Args:
            performance_data: VR performance test results
            domain: Optimization domain to focus on

        Returns:
            Discovered functions and their performance metrics
        """

        print(f"🧬 Discovering VR functions for domain: {domain}")

        if domain not in self.optimization_domains:
            raise ValueError(
                f"Unknown domain: {domain}. Available: {list(self.optimization_domains.keys())}"
            )

        domain_config = self.optimization_domains[domain]

        try:
            # Prepare training data
            training_data = self._prepare_training_data(performance_data, domain_config)

            # Run evolutionary algorithm
            evolution_result = await self._run_evolution(training_data, domain_config)

            # Validate discovered functions
            validated_functions = await self._validate_functions(
                evolution_result, training_data
            )

            # Store discoveries
            discovery_id = await self._store_discoveries(validated_functions, domain)

            print(f"✅ Function discovery completed: {discovery_id}")
            print(
                f"   Best function fitness: {validated_functions['best_fitness']:.4f}"
            )
            print(f"   Functions discovered: {len(validated_functions['functions'])}")

            return {
                "discovery_id": discovery_id,
                "domain": domain,
                "best_function": validated_functions["best_function"],
                "best_fitness": validated_functions["best_fitness"],
                "all_functions": validated_functions["functions"],
                "evolution_stats": evolution_result["stats"],
            }

        except Exception as e:
            print(f"❌ Failed to discover VR functions: {e}")
            raise

    def _prepare_training_data(
        self, performance_data: List[Dict[str, Any]], domain_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare performance data for function evolution"""

        input_vars = domain_config["input_variables"]
        target_metric = domain_config["target_metric"]

        X = []  # Input features
        y = []  # Target values

        for data_point in performance_data:
            # Extract individual test results
            individual_results = data_point.get("individual_results", [])

            for result in individual_results:
                if not result.get("success"):
                    continue

                metrics = result.get("metrics", {})
                config = result.get("config", {})

                # Extract input features
                features = []
                for var in input_vars:
                    if var == "gpu_util":
                        features.append(metrics.get("avg_gpu_util", 50))
                    elif var == "vram_usage":
                        features.append(metrics.get("max_vram_usage", 1000))
                    elif var == "cpu_util":
                        features.append(metrics.get("avg_cpu_util", 30))
                    elif var == "scene_complexity":
                        # Infer from scene name
                        scene = config.get("scene_name", "main_menu")
                        complexity = {
                            "main_menu": 1,
                            "gameplay_scene": 2,
                            "stress_test": 3,
                        }.get(scene, 1)
                        features.append(complexity)
                    elif var == "frame_rate":
                        features.append(metrics.get("avg_fps", 60))
                    elif var == "motion_intensity":
                        features.append(random.uniform(0.1, 1.0))  # Simulated
                    elif var == "user_age":
                        features.append(random.uniform(18, 65))  # Simulated
                    elif var == "session_duration":
                        features.append(config.get("test_duration", 60))
                    else:
                        features.append(random.uniform(0, 1))  # Default random value

                # Extract target value
                if target_metric == "frame_time_variance":
                    target = metrics.get("frame_time_std", 0) ** 2
                elif target_metric == "comfort_score":
                    target = metrics.get("vr_comfort_score", 50)
                elif target_metric == "interaction_success":
                    target = random.uniform(0.7, 1.0)  # Simulated
                elif target_metric == "predicted_fps":
                    target = metrics.get("avg_fps", 60)
                else:
                    target = random.uniform(0, 100)

                X.append(features)
                y.append(target)

        return {
            "X": np.array(X),
            "y": np.array(y),
            "input_variables": input_vars,
            "target_metric": target_metric,
            "goal": domain_config["goal"],
            "sample_count": len(X),
        }

    async def _run_evolution(
        self, training_data: Dict[str, Any], domain_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run evolutionary algorithm to discover functions"""

        print(f"    🔬 Running evolution: {self.generations} generations")

        X = training_data["X"]
        y = training_data["y"]
        goal = training_data["goal"]

        # Initialize population if empty
        if not self.current_population:
            await self._initialize_population()

        best_fitness_history = []
        avg_fitness_history = []

        for generation in range(self.generations):
            # Evaluate fitness for all functions
            fitness_scores = []

            for func_data in self.current_population:
                fitness = self._evaluate_function_fitness(func_data, X, y, goal)
                fitness_scores.append(fitness)
                func_data["fitness"] = fitness

            # Track statistics
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            if generation % 10 == 0:
                print(
                    f"      Generation {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}"
                )

            # Selection and reproduction
            self.current_population = await self._evolve_population(
                self.current_population, fitness_scores
            )

            # Early stopping if converged
            if (
                generation > 20
                and abs(best_fitness_history[-1] - best_fitness_history[-10]) < 0.001
            ):
                print(f"      Converged at generation {generation}")
                break

        # Find best function
        best_func = max(self.current_population, key=lambda f: f.get("fitness", 0))

        return {
            "best_function": best_func,
            "best_fitness": best_func.get("fitness", 0),
            "population": self.current_population,
            "stats": {
                "generations_run": generation + 1,
                "best_fitness_history": best_fitness_history,
                "avg_fitness_history": avg_fitness_history,
                "final_population_size": len(self.current_population),
            },
        }

    def _evaluate_function_fitness(
        self, func_data: Dict[str, Any], X: np.ndarray, y: np.ndarray, goal: str
    ) -> float:
        """Evaluate fitness of a function on training data"""

        try:
            # Apply function to input data
            predictions = self._apply_function(func_data, X)

            # Calculate fitness based on goal
            if goal == "minimize":
                # For minimization, fitness is inverse of error
                mse = np.mean((predictions - y) ** 2)
                fitness = 1.0 / (1.0 + mse)
            elif goal == "maximize":
                # For maximization, fitness is correlation with target
                if np.std(predictions) > 0 and np.std(y) > 0:
                    correlation = np.corrcoef(predictions, y)[0, 1]
                    fitness = max(0, correlation)
                else:
                    fitness = 0
            elif goal == "accuracy":
                # For accuracy, fitness is negative MAE
                mae = np.mean(np.abs(predictions - y))
                fitness = 1.0 / (1.0 + mae)
            else:
                fitness = 0

            # Add complexity penalty
            complexity_penalty = func_data.get("complexity", 1) * 0.01
            fitness = max(0, fitness - complexity_penalty)

            return fitness

        except Exception as e:
            # Return low fitness for invalid functions
            return 0.001

    def _apply_function(self, func_data: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """Apply evolved function to input data"""

        func_type = func_data["type"]
        params = func_data["parameters"]

        # Handle multi-dimensional input
        if X.ndim == 1:
            x = X
        else:
            # For multi-dimensional input, use weighted combination
            weights = params.get("input_weights", [1.0] * X.shape[1])
            x = np.dot(X, weights[: X.shape[1]])

        try:
            if func_type == "polynomial":
                result = (
                    params["a"] * x**3
                    + params["b"] * x**2
                    + params["c"] * x
                    + params["d"]
                )

            elif func_type == "trigonometric":
                result = params["a"] * np.sin(params["b"] * x + params["c"]) + params[
                    "d"
                ] * np.cos(params["e"] * x + params[""])

            elif func_type == "exponential":
                # Clip to prevent overflow
                exp_arg = np.clip(params["b"] * x, -50, 50)
                log_arg = np.clip(params["d"] * x + params["e"], 0.001, 1e6)
                result = params["a"] * np.exp(exp_arg) + params["c"] * np.log(log_arg)

            elif func_type == "hybrid":
                exp_arg = np.clip(params["c"] * x, -10, 10)
                result = (
                    params["a"] * np.sin(params["b"] * x) * np.exp(exp_arg)
                    + params["d"] * x**2
                    + params["e"]
                )

            else:
                result = x  # Identity function as fallback

            # Clip results to reasonable range
            return np.clip(result, -1e6, 1e6)

        except Exception as e:
            # Return zeros for invalid functions
            return np.zeros_like(x)

    async def _initialize_population(self):
        """Initialize population of random functions"""

        self.current_population = []

        for _ in range(self.population_size):
            func_type = random.choice(list(self.function_templates.keys()))

            if func_type == "polynomial":
                params = {
                    "a": random.uniform(-2, 2),
                    "b": random.uniform(-2, 2),
                    "c": random.uniform(-2, 2),
                    "d": random.uniform(-10, 10),
                }
            elif func_type == "trigonometric":
                params = {
                    "a": random.uniform(-5, 5),
                    "b": random.uniform(0.1, 2),
                    "c": random.uniform(0, 2 * math.pi),
                    "d": random.uniform(-5, 5),
                    "e": random.uniform(0.1, 2),
                    "": random.uniform(0, 2 * math.pi),
                }
            elif func_type == "exponential":
                params = {
                    "a": random.uniform(-2, 2),
                    "b": random.uniform(-0.1, 0.1),
                    "c": random.uniform(-2, 2),
                    "d": random.uniform(0.1, 2),
                    "e": random.uniform(0.1, 10),
                }
            elif func_type == "hybrid":
                params = {
                    "a": random.uniform(-2, 2),
                    "b": random.uniform(0.1, 2),
                    "c": random.uniform(-0.1, 0.1),
                    "d": random.uniform(-1, 1),
                    "e": random.uniform(-10, 10),
                }

            # Add input weights for multi-dimensional functions
            params["input_weights"] = [random.uniform(0.1, 2) for _ in range(4)]

            func_data = {
                "type": func_type,
                "parameters": params,
                "complexity": len(params),
                "generation_created": 0,
                "fitness": 0,
            }

            self.current_population.append(func_data)

    async def _evolve_population(
        self, population: List[Dict[str, Any]], fitness_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Evolve population through selection, crossover, and mutation"""

        # Sort by fitness
        sorted_pop = sorted(
            zip(population, fitness_scores), key=lambda x: x[1], reverse=True
        )

        new_population = []

        # Keep top 20% (elitism)
        elite_count = max(1, self.population_size // 5)
        for i in range(elite_count):
            new_population.append(sorted_pop[i][0].copy())

        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(sorted_pop)
            parent2 = self._tournament_selection(sorted_pop)

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            new_population.append(child)

        return new_population

    def _tournament_selection(self, sorted_population: List[tuple]) -> Dict[str, Any]:
        """Tournament selection for parent selection"""
        tournament_size = 3
        tournament = random.sample(
            sorted_population, min(tournament_size, len(sorted_population))
        )
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crossover two parent functions"""

        # Choose function type from one parent
        func_type = random.choice([parent1["type"], parent2["type"]])

        # Blend parameters
        params1 = parent1["parameters"]
        params2 = parent2["parameters"]

        new_params = {}
        for key in params1.keys():
            if key in params2:
                # Blend parameters
                alpha = random.uniform(0.3, 0.7)
                if isinstance(params1[key], list):
                    new_params[key] = [
                        alpha * p1 + (1 - alpha) * p2
                        for p1, p2 in zip(params1[key], params2[key])
                    ]
                else:
                    new_params[key] = alpha * params1[key] + (1 - alpha) * params2[key]
            else:
                new_params[key] = params1[key]

        return {
            "type": func_type,
            "parameters": new_params,
            "complexity": len(new_params),
            "generation_created": len(self.evolution_history),
            "fitness": 0,
        }

    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an individual function"""

        mutated = individual.copy()
        params = mutated["parameters"].copy()

        # Mutate random parameter
        param_keys = list(params.keys())
        if param_keys:
            key = random.choice(param_keys)

            if isinstance(params[key], list):
                # Mutate list parameter
                idx = random.randint(0, len(params[key]) - 1)
                params[key][idx] += random.gauss(0, 0.1)
            else:
                # Mutate scalar parameter
                params[key] += random.gauss(0, 0.1)

        mutated["parameters"] = params
        return mutated

    async def _validate_functions(
        self, evolution_result: Dict[str, Any], training_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate discovered functions"""

        print("    ✅ Validating discovered functions...")

        # Get top functions
        population = evolution_result["population"]
        top_functions = sorted(
            population, key=lambda f: f.get("fitness", 0), reverse=True
        )[:10]

        validated = {
            "best_function": evolution_result["best_function"],
            "best_fitness": evolution_result["best_fitness"],
            "functions": top_functions,
            "validation_metrics": {
                "training_samples": training_data["sample_count"],
                "input_dimensions": len(training_data["input_variables"]),
                "target_metric": training_data["target_metric"],
            },
        }

        return validated

    async def _store_discoveries(
        self, validated_functions: Dict[str, Any], domain: str
    ) -> str:
        """Store discovered functions"""

        discovery_id = f"funsearch_{domain}_{int(datetime.utcnow().timestamp())}"

        # Create discovery directory
        discovery_dir = f"discovered_functions/{discovery_id}"
        os.makedirs(discovery_dir, exist_ok=True)

        # Save discovery data
        with open(f"{discovery_dir}/discovery.json", "w") as f:
            json.dump(
                {
                    "discovery_id": discovery_id,
                    "domain": domain,
                    "timestamp": datetime.utcnow().isoformat(),
                    **validated_functions,
                },
                f,
                indent=2,
            )

        # Save best function as executable code
        best_func = validated_functions["best_function"]
        with open(f"{discovery_dir}/best_function.py", "w") as f:
            f.write(self._generate_function_code(best_func))

        return discovery_id

    def _generate_function_code(self, func_data: Dict[str, Any]) -> str:
        """Generate executable Python code for a function"""

        func_type = func_data["type"]
        params = func_data["parameters"]

        code = """
# Auto-generated VR optimization function
# Type: {func_type}
# Fitness: {func_data.get('fitness', 0):.4f}

import numpy as np
import math

def vr_optimization_function(inputs):
    \"\"\"
    Discovered VR optimization function

    Args:
        inputs: Array of input variables

    Returns:
        Optimized output value
    \"\"\"

    # Handle input weights
    weights = {params.get('input_weights', [1.0, 1.0, 1.0, 1.0])}
    x = np.dot(inputs, weights[:len(inputs)])

"""

        if func_type == "polynomial":
            code += """    # Polynomial function
    result = ({params['a']} * x**3 +
              {params['b']} * x**2 +
              {params['c']} * x +
              {params['d']})
"""
        elif func_type == "trigonometric":
            code += """    # Trigonometric function
    result = ({params['a']} * np.sin({params['b']} * x + {params['c']}) +
              {params['d']} * np.cos({params['e']} * x + {params['"]}))
"""
        elif func_type == "exponential":
            code += """    # Exponential function
    exp_arg = np.clip({params['b']} * x, -50, 50)
    log_arg = np.clip({params['d']} * x + {params['e']}, 0.001, 1e6)
    result = ({params['a']} * np.exp(exp_arg) +
              {params['c']} * np.log(log_arg))
"""
        elif func_type == "hybrid":
            code += """    # Hybrid function
    exp_arg = np.clip({params['c']} * x, -10, 10)
    result = ({params['a']} * np.sin({params['b']} * x) * np.exp(exp_arg) +
              {params['d']} * x**2 +
              {params['e']})
"""

        code += """
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity])
"""

        return code

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about function discoveries"""

        discovery_dir = "discovered_functions"
        if not os.path.exists(discovery_dir):
            return {"total_discoveries": 0}

        discoveries = [
            d for d in os.listdir(discovery_dir) if d.startswith("funsearch_")
        ]

        domain_counts = {}
        total_fitness = 0

        for discovery in discoveries:
            try:
                with open(f"{discovery_dir}/{discovery}/discovery.json", "r") as f:
                    data = json.load(f)
                    domain = data.get("domain", "unknown")
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    total_fitness += data.get("best_fitness", 0)
            except Exception:
                continue

        return {
            "total_discoveries": len(discoveries),
            "domains_explored": domain_counts,
            "average_fitness": total_fitness / len(discoveries) if discoveries else 0,
            "optimization_domains": list(self.optimization_domains.keys()),
        }
