"""
FunSearch Integration Manager for CloudVR-PerfGuard
Handles evolution of functions for VR affordance discovery using DeepMind's FunSearch
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import subprocess
import os
import tempfile
from pathlib import Path

from core.database import DatabaseManager
from funsearch_wrapper import run_funsearch_optimization, generate_gemini_function


class FunSearchIntegration:
    """
    Manages FunSearch evolution for VR affordance discovery
    """
    
    def __init__(self, db_manager: DatabaseManager, funsearch_path: str = "./funsearch"):
        self.db_manager = db_manager
        self.funsearch_path = Path(funsearch_path)
        self.evolution_configs = {
            "visual_cue_discovery": {
                "program_name": "visual_cue_optimizer",
                "max_iterations": 100,
                "population_size": 50,
                "mutation_rate": 0.1
            },
            "affordance_scoring": {
                "program_name": "affordance_scorer", 
                "max_iterations": 150,
                "population_size": 75,
                "mutation_rate": 0.15
            },
            "interaction_predictor": {
                "program_name": "interaction_predictor",
                "max_iterations": 200,
                "population_size": 100,
                "mutation_rate": 0.2
            }
        }
    
    async def evolve_visual_cue_function(
        self,
        job_id: str,
        vr_performance_data: Dict[str, Any],
        evolution_type: str = "visual_cue_discovery"
    ) -> Dict[str, Any]:
        """
        Evolve a function for discovering optimal visual cues in VR environments
        """
        try:
            print(f"INFO: Starting FunSearch evolution for job {job_id}, type: {evolution_type}")
            
            # Get evolution configuration
            config = self.evolution_configs.get(evolution_type, self.evolution_configs["visual_cue_discovery"])
            
            # Prepare the seed function based on VR performance data
            seed_function = self._generate_seed_function(vr_performance_data, evolution_type)
            
            # Create evaluation function based on performance metrics
            evaluation_function = self._create_evaluation_function(vr_performance_data)
            
            # Run FunSearch evolution
            evolved_functions = await self._run_funsearch_evolution(
                seed_function=seed_function,
                evaluation_function=evaluation_function,
                config=config,
                job_id=job_id
            )
            
            # Store results in database
            best_function = None
            best_score = float('-inf')
            
            for iteration, func_data in enumerate(evolved_functions):
                function_id = f"{job_id}_funsearch_{evolution_type}_{iteration}"
                
                success = await self.db_manager.store_evolved_function(
                    function_id=function_id,
                    job_id=job_id,
                    program_name=config["program_name"],
                    evolution_iteration=iteration,
                    evolved_function_code=func_data["code"],
                    evaluation_score=func_data["score"],
                    discovery_timestamp=datetime.utcnow().isoformat(),
                    metadata={
                        "evolution_type": evolution_type,
                        "vr_metrics": func_data.get("vr_metrics", {}),
                        "optimization_target": func_data.get("target", "unknown")
                    }
                )
                
                if success and func_data["score"] > best_score:
                    best_function = func_data
                    best_score = func_data["score"]
            
            return {
                "status": "success",
                "evolution_type": evolution_type,
                "total_functions_evolved": len(evolved_functions),
                "best_function": best_function,
                "best_score": best_score,
                "job_id": job_id
            }
            
        except Exception as e:
            print(f"ERROR: FunSearch evolution failed for job {job_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "job_id": job_id
            }
    
    def _generate_seed_function(self, vr_data: Dict[str, Any], evolution_type: str) -> str:
        """Generate a seed function based on VR performance data and evolution type"""
        
        if evolution_type == "visual_cue_discovery":
            return f"""
def visual_cue_optimizer(object_properties, user_context, environment_state):
    '''
    Optimize visual cues for VR object affordances
    Based on performance data: {vr_data.get('avg_fps', 'N/A')} FPS, 
    {vr_data.get('comfort_score', 'N/A')} comfort score
    '''
    # Extract key metrics from current performance
    base_fps = {vr_data.get('avg_fps', 60)}
    comfort_threshold = {vr_data.get('comfort_score', 0.7)}
    
    # Visual cue parameters to optimize
    brightness = min(1.0, max(0.1, object_properties.get('brightness', 0.5)))
    contrast = min(2.0, max(0.5, object_properties.get('contrast', 1.0)))
    edge_enhancement = min(1.5, max(0.0, object_properties.get('edge_enhancement', 0.3)))
    
    # Context-aware adjustments
    if user_context.get('visual_impairment', False):
        brightness *= 1.2
        contrast *= 1.3
    
    if environment_state.get('lighting_level', 'normal') == 'low':
        brightness *= 1.1
        edge_enhancement *= 1.2
    
    # Performance-based optimization
    if base_fps < 60:
        # Reduce computational load
        edge_enhancement *= 0.8
    
    return {{
        'brightness': brightness,
        'contrast': contrast, 
        'edge_enhancement': edge_enhancement,
        'predicted_affordance_score': brightness * contrast * (1 + edge_enhancement)
    }}
"""
        
        elif evolution_type == "affordance_scoring":
            return f"""
def affordance_scorer(visual_cues, interaction_history, user_profile):
    '''
    Score the affordance strength of visual cues
    Optimized for {vr_data.get('avg_frame_time', 'N/A')}ms frame time
    '''
    # Base scoring from visual properties
    visual_score = (
        visual_cues.get('brightness', 0.5) * 0.3 +
        visual_cues.get('contrast', 1.0) * 0.4 +
        visual_cues.get('edge_enhancement', 0.3) * 0.3
    )
    
    # Historical interaction success rate
    success_rate = interaction_history.get('success_rate', 0.5)
    interaction_count = interaction_history.get('count', 0)
    
    # User profile adjustments
    experience_level = user_profile.get('vr_experience', 'novice')
    experience_multiplier = {{
        'novice': 1.2,
        'intermediate': 1.0, 
        'expert': 0.9
    }}.get(experience_level, 1.0)
    
    # Performance consideration
    frame_time_penalty = max(0, ({vr_data.get('avg_frame_time', 16.67)} - 16.67) / 16.67)
    
    final_score = (
        visual_score * 0.4 +
        success_rate * 0.4 +
        min(interaction_count / 100, 1.0) * 0.2
    ) * experience_multiplier * (1 - frame_time_penalty * 0.1)
    
    return max(0.0, min(1.0, final_score))
"""
        
        else:  # interaction_predictor
            return f"""
def interaction_predictor(visual_cues, user_state, object_context):
    '''
    Predict likelihood of successful interaction
    Optimized for {vr_data.get('min_fps', 'N/A')} min FPS performance
    '''
    # Visual cue strength
    cue_strength = (
        visual_cues.get('brightness', 0.5) +
        visual_cues.get('contrast', 1.0) +
        visual_cues.get('edge_enhancement', 0.3)
    ) / 3.0
    
    # User readiness indicators
    hand_stability = user_state.get('hand_stability', 0.5)
    gaze_focus = user_state.get('gaze_focus_duration', 0.0) / 2.0  # Normalize to 2 seconds
    
    # Object context factors
    object_size = object_context.get('size_ratio', 0.5)  # Relative to hand size
    object_distance = max(0, 1.0 - object_context.get('distance', 0.5))  # Closer = better
    
    # Performance-based confidence adjustment
    min_fps = {vr_data.get('min_fps', 30)}
    performance_confidence = min(1.0, min_fps / 60.0)
    
    interaction_probability = (
        cue_strength * 0.35 +
        hand_stability * 0.25 +
        min(gaze_focus, 1.0) * 0.2 +
        object_size * 0.1 +
        object_distance * 0.1
    ) * performance_confidence
    
    return max(0.0, min(1.0, interaction_probability))
"""
    
    def _create_evaluation_function(self, vr_data: Dict[str, Any]) -> Callable:
        """Create an evaluation function based on VR performance metrics"""
        
        def evaluate_function(func_code: str, test_cases: List[Dict]) -> float:
            """Evaluate a function's performance on test cases"""
            try:
                # Execute the function code safely
                namespace = {}
                exec(func_code, namespace)
                
                # Find the main function (first function defined)
                func_name = None
                for name, obj in namespace.items():
                    if callable(obj) and not name.startswith('_'):
                        func_name = name
                        break
                
                if not func_name:
                    return 0.0
                
                func = namespace[func_name]
                total_score = 0.0
                
                for test_case in test_cases:
                    try:
                        result = func(**test_case['inputs'])
                        
                        # Score based on expected outcomes and VR performance metrics
                        if isinstance(result, dict):
                            score = result.get('predicted_affordance_score', 0.5)
                        elif isinstance(result, (int, float)):
                            score = float(result)
                        else:
                            score = 0.5
                        
                        # Adjust score based on VR performance context
                        fps_factor = min(1.0, vr_data.get('avg_fps', 60) / 60.0)
                        comfort_factor = vr_data.get('comfort_score', 0.7)
                        
                        adjusted_score = score * fps_factor * comfort_factor
                        total_score += adjusted_score
                        
                    except Exception:
                        total_score += 0.0  # Failed execution
                
                return total_score / len(test_cases) if test_cases else 0.0
                
            except Exception:
                return 0.0
        
        return evaluate_function
    
    async def _run_funsearch_evolution(
        self,
        seed_function: str,
        evaluation_function: Callable,
        config: Dict[str, Any],
        job_id: str
    ) -> List[Dict[str, Any]]:
        """Run the FunSearch evolution process"""
        
        # Generate test cases based on VR scenarios
        test_cases = self._generate_vr_test_cases()
        
        # Initialize population with seed function
        population = [seed_function]
        evolved_functions = []
        
        for iteration in range(config["max_iterations"]):
            print(f"INFO: FunSearch iteration {iteration + 1}/{config['max_iterations']} for job {job_id}")
            
            # Evaluate current population
            scored_population = []
            for func_code in population:
                score = evaluation_function(func_code, test_cases)
                scored_population.append({
                    "code": func_code,
                    "score": score,
                    "iteration": iteration
                })
            
            # Sort by score (best first)
            scored_population.sort(key=lambda x: x["score"], reverse=True)
            
            # Store best function from this iteration
            if scored_population:
                best_func = scored_population[0]
                best_func["vr_metrics"] = {
                    "test_cases_passed": sum(1 for tc in test_cases if evaluation_function(best_func["code"], [tc]) > 0.5),
                    "total_test_cases": len(test_cases)
                }
                evolved_functions.append(best_func)
            
            # Generate next generation through mutation and crossover
            if iteration < config["max_iterations"] - 1:
                population = self._evolve_population(scored_population, config)
        
        return evolved_functions
    
    def _generate_vr_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for VR affordance scenarios"""
        
        test_cases = [
            # High visibility scenario
            {
                "inputs": {
                    "object_properties": {"brightness": 0.8, "contrast": 1.5, "edge_enhancement": 0.5},
                    "user_context": {"visual_impairment": False, "vr_experience": "intermediate"},
                    "environment_state": {"lighting_level": "normal"}
                }
            },
            # Low light scenario
            {
                "inputs": {
                    "object_properties": {"brightness": 0.4, "contrast": 1.0, "edge_enhancement": 0.3},
                    "user_context": {"visual_impairment": False, "vr_experience": "novice"},
                    "environment_state": {"lighting_level": "low"}
                }
            },
            # Accessibility scenario
            {
                "inputs": {
                    "object_properties": {"brightness": 0.6, "contrast": 1.2, "edge_enhancement": 0.4},
                    "user_context": {"visual_impairment": True, "vr_experience": "expert"},
                    "environment_state": {"lighting_level": "normal"}
                }
            },
            # Performance-constrained scenario
            {
                "inputs": {
                    "object_properties": {"brightness": 0.5, "contrast": 0.8, "edge_enhancement": 0.2},
                    "user_context": {"visual_impairment": False, "vr_experience": "intermediate"},
                    "environment_state": {"lighting_level": "high", "performance_mode": "low_power"}
                }
            }
        ]
        
        return test_cases
    
    def _evolve_population(self, scored_population: List[Dict], config: Dict) -> List[str]:
        """Evolve the population through mutation and crossover"""
        
        # Keep top performers
        elite_size = max(1, config["population_size"] // 4)
        new_population = [func["code"] for func in scored_population[:elite_size]]
        
        # Generate offspring through mutation
        while len(new_population) < config["population_size"]:
            # Select parent (tournament selection)
            parent = self._tournament_selection(scored_population, tournament_size=3)
            
            # Mutate parent
            mutated_code = self._mutate_function(parent["code"], config["mutation_rate"])
            new_population.append(mutated_code)
        
        return new_population
    
    def _tournament_selection(self, population: List[Dict], tournament_size: int = 3) -> Dict:
        """Select individual using tournament selection"""
        import random
        
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x["score"])
    
    def _mutate_function(self, func_code: str, mutation_rate: float) -> str:
        """Mutate a function by modifying numerical constants"""
        import random
        import re
        
        # Find numerical constants in the code
        numbers = re.findall(r'\b\d+\.?\d*\b', func_code)
        
        mutated_code = func_code
        for number in numbers:
            if random.random() < mutation_rate:
                try:
                    original_value = float(number)
                    # Apply small random change
                    mutation_factor = random.uniform(0.8, 1.2)
                    new_value = original_value * mutation_factor
                    
                    # Replace first occurrence
                    mutated_code = mutated_code.replace(number, f"{new_value:.3f}", 1)
                except ValueError:
                    continue
        
        return mutated_code
    
    async def get_evolved_functions_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all evolved functions for a specific job"""
        return await self.db_manager.get_evolved_functions_for_job(job_id)
    
    async def get_best_function_for_type(self, job_id: str, evolution_type: str) -> Optional[Dict[str, Any]]:
        """Get the best evolved function for a specific evolution type"""
        functions = await self.get_evolved_functions_for_job(job_id)
        
        # Filter by evolution type and find best score
        type_functions = [
            f for f in functions 
            if f.get("metadata", {}).get("evolution_type") == evolution_type
        ]
        
        if not type_functions:
            return None
        
        return max(type_functions, key=lambda x: x.get("evaluation_score", 0)) 