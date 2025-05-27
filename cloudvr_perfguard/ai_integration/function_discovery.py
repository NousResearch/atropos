"""
Function Discovery for VR Optimization
Interfaces with FunSearch for discovering optimization functions
"""

import os
import json
import numpy as np
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import math
import random


class OptimizationDiscovery:
    """
    Discovers optimization functions for VR performance using evolutionary algorithms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.funsearch_path = self.config.get("funsearch_path", "../funsearch")
        self.output_dir = self.config.get("output_dir", "discovered_functions")
        self.population_size = self.config.get("population_size", 30)
        self.generations = self.config.get("generations", 50)
        
        # VR optimization domains
        self.optimization_domains = {
            "frame_time_consistency": {
                "description": "Optimize frame time consistency for VR comfort",
                "target_metric": "frame_time_variance",
                "goal": "minimize"
            },
            "comfort_optimization": {
                "description": "Maximize VR comfort scores",
                "target_metric": "comfort_score",
                "goal": "maximize"
            },
            "performance_efficiency": {
                "description": "Optimize performance per GPU utilization",
                "target_metric": "fps_per_gpu_util",
                "goal": "maximize"
            }
        }
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def check_funsearch_availability(self) -> bool:
        """Check if FunSearch is available"""
        
        funsearch_dir = Path(self.funsearch_path)
        
        if not funsearch_dir.exists():
            print(f"FunSearch not found at {self.funsearch_path}")
            print("Please clone: git clone https://github.com/deepmind/funsearch.git")
            return False
        
        # Check for key files
        required_files = ["funsearch.py", "requirements.txt"]
        for file in required_files:
            if not (funsearch_dir / file).exists():
                print(f"Required file {file} not found in FunSearch directory")
                return False
        
        return True
    
    def discover_optimization_function(
        self,
        training_data: Dict[str, Any],
        domain: str = "frame_time_consistency"
    ) -> Dict[str, Any]:
        """
        Discover optimization function for a specific domain
        
        Args:
            training_data: Formatted training data from PerformanceDataAdapter
            domain: Optimization domain to focus on
            
        Returns:
            Discovery results including best function and metadata
        """
        
        if domain not in self.optimization_domains:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self.optimization_domains.keys())}")
        
        domain_config = self.optimization_domains[domain]
        
        # Prepare optimization data
        optimization_data = self._prepare_optimization_data(training_data, domain_config)
        
        # Check if FunSearch is available
        if self.check_funsearch_availability():
            # Use actual FunSearch
            result = self._discover_with_funsearch(optimization_data, domain_config)
        else:
            # Use fallback evolutionary algorithm
            result = self._discover_with_fallback(optimization_data, domain_config)
        
        # Store the discovered function
        discovery_id = self._store_discovery(result, domain)
        result["discovery_id"] = discovery_id
        
        return result
    
    def _prepare_optimization_data(
        self,
        training_data: Dict[str, Any],
        domain_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for optimization function discovery"""
        
        features = training_data.get("features", np.array([]))
        targets = training_data.get("targets", [])
        target_metric = domain_config["target_metric"]
        
        # Extract target values for the specific domain
        if targets and isinstance(targets[0], dict):
            # Extract specific target metric
            y_values = [t.get(target_metric, 0) for t in targets]
        else:
            # Use targets directly if they're already numeric
            y_values = targets
        
        return {
            "X": features,
            "y": np.array(y_values),
            "feature_names": training_data.get("feature_names", []),
            "target_metric": target_metric,
            "goal": domain_config["goal"],
            "sample_count": len(y_values),
            "feature_ranges": self._calculate_feature_ranges(features)
        }
    
    def _calculate_feature_ranges(self, features: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate feature ranges for normalization"""
        
        if features.size == 0:
            return {}
        
        ranges = {}
        feature_names = ["gpu_util", "vram_usage", "cpu_util", "scene_complexity", "duration", "gpu_type"]
        
        for i in range(min(features.shape[1], len(feature_names))):
            column = features[:, i]
            ranges[feature_names[i]] = {
                "min": float(np.min(column)),
                "max": float(np.max(column)),
                "mean": float(np.mean(column)),
                "std": float(np.std(column))
            }
        
        return ranges
    
    def _discover_with_funsearch(
        self,
        optimization_data: Dict[str, Any],
        domain_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Discover function using actual FunSearch"""
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare input files for FunSearch
                data_file = os.path.join(temp_dir, "optimization_data.json")
                config_file = os.path.join(temp_dir, "config.json")
                
                # Convert numpy arrays to lists for JSON serialization
                serializable_data = {
                    "X": optimization_data["X"].tolist(),
                    "y": optimization_data["y"].tolist(),
                    "feature_names": optimization_data["feature_names"],
                    "target_metric": optimization_data["target_metric"],
                    "goal": optimization_data["goal"]
                }
                
                with open(data_file, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
                
                funsearch_config = {
                    "population_size": self.population_size,
                    "generations": self.generations,
                    "domain": domain_config["description"]
                }
                
                with open(config_file, 'w') as f:
                    json.dump(funsearch_config, f, indent=2)
                
                # Run FunSearch
                cmd = [
                    "python",
                    os.path.join(self.funsearch_path, "funsearch.py"),
                    "--data-file", data_file,
                    "--config-file", config_file,
                    "--output-dir", temp_dir
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    # Parse FunSearch output
                    output_file = os.path.join(temp_dir, "discovered_function.json")
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            funsearch_result = json.load(f)
                        
                        return {
                            "function_code": funsearch_result.get("function_code", ""),
                            "fitness_score": funsearch_result.get("fitness", 0),
                            "discovery_method": "funsearch",
                            "generations_run": funsearch_result.get("generations", 0),
                            "population_size": self.population_size,
                            "discovery_time": datetime.utcnow().isoformat()
                        }
                
                # If FunSearch failed, fall back to simple evolutionary algorithm
                print(f"FunSearch failed: {result.stderr}")
                return self._discover_with_fallback(optimization_data, domain_config)
        
        except Exception as e:
            print(f"Error running FunSearch: {e}")
            return self._discover_with_fallback(optimization_data, domain_config)
    
    def _discover_with_fallback(
        self,
        optimization_data: Dict[str, Any],
        domain_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Discover function using fallback evolutionary algorithm"""
        
        X = optimization_data["X"]
        y = optimization_data["y"]
        goal = optimization_data["goal"]
        
        # Simple evolutionary algorithm for function discovery
        best_function = self._run_simple_evolution(X, y, goal)
        
        return {
            "function_code": self._generate_function_code(best_function),
            "fitness_score": best_function["fitness"],
            "discovery_method": "simple_evolution",
            "generations_run": self.generations,
            "population_size": self.population_size,
            "function_type": best_function["type"],
            "parameters": best_function["parameters"],
            "discovery_time": datetime.utcnow().isoformat()
        }
    
    def _run_simple_evolution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        goal: str
    ) -> Dict[str, Any]:
        """Run simple evolutionary algorithm for function discovery"""
        
        # Initialize population of functions
        population = self._initialize_function_population()
        
        best_fitness = -float('inf') if goal == "maximize" else float('inf')
        best_function = None
        
        for generation in range(self.generations):
            # Evaluate fitness for all functions
            for func in population:
                fitness = self._evaluate_function_fitness(func, X, y, goal)
                func["fitness"] = fitness
                
                # Track best function
                if ((goal == "maximize" and fitness > best_fitness) or
                    (goal == "minimize" and fitness < best_fitness)):
                    best_fitness = fitness
                    best_function = func.copy()
            
            # Evolve population
            population = self._evolve_population(population, goal)
        
        return best_function or population[0]
    
    def _initialize_function_population(self) -> List[Dict[str, Any]]:
        """Initialize population of random functions"""
        
        population = []
        function_types = ["linear", "polynomial", "exponential"]
        
        for _ in range(self.population_size):
            func_type = random.choice(function_types)
            
            if func_type == "linear":
                params = {
                    "weights": [random.uniform(-2, 2) for _ in range(6)],
                    "bias": random.uniform(-10, 10)
                }
            elif func_type == "polynomial":
                params = {
                    "coefficients": [random.uniform(-1, 1) for _ in range(4)],
                    "weights": [random.uniform(-2, 2) for _ in range(6)]
                }
            elif func_type == "exponential":
                params = {
                    "scale": random.uniform(0.1, 2.0),
                    "weights": [random.uniform(-1, 1) for _ in range(6)],
                    "offset": random.uniform(-5, 5)
                }
            
            population.append({
                "type": func_type,
                "parameters": params,
                "fitness": 0
            })
        
        return population
    
    def _evaluate_function_fitness(
        self,
        func: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        goal: str
    ) -> float:
        """Evaluate fitness of a function"""
        
        try:
            predictions = self._apply_function(func, X)
            
            if goal == "minimize":
                # For minimization, fitness is negative MSE
                mse = np.mean((predictions - y) ** 2)
                return -mse
            else:
                # For maximization, fitness is correlation
                if np.std(predictions) > 0 and np.std(y) > 0:
                    correlation = np.corrcoef(predictions, y)[0, 1]
                    return correlation if not np.isnan(correlation) else 0
                else:
                    return 0
        
        except Exception:
            return -1000  # Penalty for invalid functions
    
    def _apply_function(self, func: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """Apply function to input data"""
        
        if X.size == 0:
            return np.array([])
        
        func_type = func["type"]
        params = func["parameters"]
        
        # Compute weighted input
        weights = params.get("weights", [1.0] * X.shape[1])
        x = np.dot(X, weights[:X.shape[1]])
        
        try:
            if func_type == "linear":
                result = x + params.get("bias", 0)
            
            elif func_type == "polynomial":
                coeffs = params.get("coefficients", [0, 0, 0, 1])
                result = (coeffs[0] * x**3 + coeffs[1] * x**2 + 
                         coeffs[2] * x + coeffs[3])
            
            elif func_type == "exponential":
                scale = params.get("scale", 1.0)
                offset = params.get("offset", 0)
                # Clip to prevent overflow
                exp_arg = np.clip(scale * x, -50, 50)
                result = np.exp(exp_arg) + offset
            
            else:
                result = x  # Identity function as fallback
            
            return np.clip(result, -1e6, 1e6)
        
        except Exception:
            return np.zeros_like(x)
    
    def _evolve_population(
        self,
        population: List[Dict[str, Any]],
        goal: str
    ) -> List[Dict[str, Any]]:
        """Evolve population through selection and mutation"""
        
        # Sort by fitness
        reverse_sort = (goal == "maximize")
        population.sort(key=lambda f: f["fitness"], reverse=reverse_sort)
        
        # Keep top 30% (elitism)
        elite_count = max(1, len(population) // 3)
        new_population = population[:elite_count].copy()
        
        # Generate rest through mutation and crossover
        while len(new_population) < self.population_size:
            if random.random() < 0.7:
                # Mutation
                parent = random.choice(population[:elite_count])
                child = self._mutate_function(parent)
            else:
                # Crossover
                parent1 = random.choice(population[:elite_count])
                parent2 = random.choice(population[:elite_count])
                child = self._crossover_functions(parent1, parent2)
            
            new_population.append(child)
        
        return new_population
    
    def _mutate_function(self, func: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a function"""
        
        mutated = {
            "type": func["type"],
            "parameters": {},
            "fitness": 0
        }
        
        # Mutate parameters
        for key, value in func["parameters"].items():
            if isinstance(value, list):
                # Mutate list parameters
                mutated["parameters"][key] = [
                    v + random.gauss(0, 0.1) for v in value
                ]
            else:
                # Mutate scalar parameters
                mutated["parameters"][key] = value + random.gauss(0, 0.1)
        
        return mutated
    
    def _crossover_functions(
        self,
        func1: Dict[str, Any],
        func2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crossover two functions"""
        
        # Choose type from one parent
        child_type = random.choice([func1["type"], func2["type"]])
        
        # Blend parameters
        child_params = {}
        params1 = func1["parameters"]
        params2 = func2["parameters"]
        
        for key in params1.keys():
            if key in params2:
                if isinstance(params1[key], list):
                    # Blend list parameters
                    child_params[key] = [
                        0.5 * p1 + 0.5 * p2
                        for p1, p2 in zip(params1[key], params2[key])
                    ]
                else:
                    # Blend scalar parameters
                    child_params[key] = 0.5 * params1[key] + 0.5 * params2[key]
            else:
                child_params[key] = params1[key]
        
        return {
            "type": child_type,
            "parameters": child_params,
            "fitness": 0
        }
    
    def _generate_function_code(self, func: Dict[str, Any]) -> str:
        """Generate executable Python code for a function"""
        
        func_type = func["type"]
        params = func["parameters"]
        
        code = f"""
# Auto-generated VR optimization function
# Type: {func_type}
# Fitness: {func.get('fitness', 0):.4f}

import numpy as np

def vr_optimization_function(features):
    \"\"\"
    VR optimization function discovered through evolutionary algorithm
    
    Args:
        features: Array of input features [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    
    Returns:
        Optimized output value
    \"\"\"
    
    # Feature weights
    weights = {params.get('weights', [1.0] * 6)}
    x = np.dot(features, weights[:len(features)])
    
"""
        
        if func_type == "linear":
            code += f"""    # Linear function
    result = x + {params.get('bias', 0)}
"""
        elif func_type == "polynomial":
            coeffs = params.get('coefficients', [0, 0, 0, 1])
            code += f"""    # Polynomial function
    result = ({coeffs[0]} * x**3 + {coeffs[1]} * x**2 + 
              {coeffs[2]} * x + {coeffs[3]})
"""
        elif func_type == "exponential":
            scale = params.get('scale', 1.0)
            offset = params.get('offset', 0)
            code += f"""    # Exponential function
    exp_arg = np.clip({scale} * x, -50, 50)
    result = np.exp(exp_arg) + {offset}
"""
        
        code += """    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
"""
        
        return code
    
    def _store_discovery(self, result: Dict[str, Any], domain: str) -> str:
        """Store discovered function"""
        
        discovery_id = f"discovery_{domain}_{int(datetime.utcnow().timestamp())}"
        discovery_dir = Path(self.output_dir) / discovery_id
        discovery_dir.mkdir(parents=True, exist_ok=True)
        
        # Save discovery metadata
        with open(discovery_dir / "discovery.json", "w") as f:
            json.dump({
                "discovery_id": discovery_id,
                "domain": domain,
                "timestamp": datetime.utcnow().isoformat(),
                **result
            }, f, indent=2)
        
        # Save function code
        with open(discovery_dir / "function.py", "w") as f:
            f.write(result.get("function_code", "# No function code generated"))
        
        return discovery_id
    
    def list_discoveries(self) -> List[Dict[str, Any]]:
        """List all discovered functions"""
        
        discoveries = []
        output_path = Path(self.output_dir)
        
        if not output_path.exists():
            return discoveries
        
        for discovery_dir in output_path.iterdir():
            if discovery_dir.is_dir() and discovery_dir.name.startswith("discovery_"):
                metadata_file = discovery_dir / "discovery.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        discoveries.append({
                            "discovery_id": discovery_dir.name,
                            "domain": metadata.get("domain", "unknown"),
                            "fitness_score": metadata.get("fitness_score", 0),
                            "discovery_method": metadata.get("discovery_method", "unknown"),
                            "discovery_time": metadata.get("discovery_time", "unknown")
                        })
                    except Exception as e:
                        print(f"Error reading metadata for {discovery_dir.name}: {e}")
        
        return sorted(discoveries, key=lambda x: x["discovery_time"], reverse=True)
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about function discoveries"""
        
        discoveries = self.list_discoveries()
        
        if not discoveries:
            return {
                "total_discoveries": 0,
                "domains_explored": {},
                "average_fitness": 0,
                "discovery_methods": {}
            }
        
        domains = {}
        methods = {}
        total_fitness = 0
        
        for discovery in discoveries:
            domain = discovery["domain"]
            method = discovery["discovery_method"]
            
            domains[domain] = domains.get(domain, 0) + 1
            methods[method] = methods.get(method, 0) + 1
            total_fitness += discovery["fitness_score"]
        
        return {
            "total_discoveries": len(discoveries),
            "domains_explored": domains,
            "average_fitness": total_fitness / len(discoveries),
            "discovery_methods": methods,
            "latest_discovery": discoveries[0] if discoveries else None
        } 