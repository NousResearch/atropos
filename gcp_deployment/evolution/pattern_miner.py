import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import fields # To get field names from dataclass

# Assuming VisualCue is in .synthetic_users and parameter definitions in .visual_cue_evolver
from .synthetic_users import VisualCue
from .visual_cue_evolver import PARAM_ORDER as EVOLVER_PARAM_ORDER, PARAM_RANGES as EVOLVER_PARAM_RANGES

class AffordancePatternMiner:
    def __init__(self, evolution_runs_data: List[Dict[str, Any]]):
        """
        Initializes the pattern miner with data from multiple evolution runs.
        Args:
            evolution_runs_data: A list of dictionaries, where each dict represents
                                 one evolution run. Expected keys in each dict:
                                 - 'best_cues_pareto_front': List of dicts representing best cues (from serialized VisualCue).
                                 - 'config': Dict of run configuration (optional, for context).
                                 - 'logbook': List of dicts from DEAP stats (optional, for fitness progression).
                                 - 'segment_name': Name of the segment for this run.
        """
        self.runs_data = evolution_runs_data
        self.param_names: List[str] = EVOLVER_PARAM_ORDER
        self.param_ranges: Dict[str, Tuple[float, float]] = EVOLVER_PARAM_RANGES

        self.all_best_cues_from_pareto: List[VisualCue] = []
        self.all_best_cues_params_from_pareto: List[List[float]] = []

        for run_data in self.runs_data:
            if run_data.get('status') == 'completed':
                pareto_cues_data = run_data.get('best_cues_pareto_front', [])
                for cue_dict in pareto_cues_data:
                    # Reconstruct VisualCue object from dict
                    # Ensure all necessary fields are present in cue_dict
                    try:
                        # Filter cue_dict for fields actually in VisualCue
                        vc_fields = {f.name for f in fields(VisualCue)}
                        filtered_cue_dict = {k: v for k, v in cue_dict.items() if k in vc_fields}
                        cue_obj = VisualCue(**filtered_cue_dict)
                        self.all_best_cues_from_pareto.append(cue_obj)
                        
                        # Extract parameters in the defined order
                        params_list = [getattr(cue_obj, name, None) for name in self.param_names]
                        if None not in params_list:
                             self.all_best_cues_params_from_pareto.append(params_list)
                        else:
                            print(f"Warning: Missing parameters for cue {cue_dict} in run for segment {run_data.get('segment_name')}")   
                    except TypeError as e:
                        print(f"Warning: Could not reconstruct VisualCue from dict {cue_dict} for segment {run_data.get('segment_name')}. Error: {e}")

    def analyze_convergence_of_best_cues(self) -> Dict[str, Dict[str, float]]:
        """Find what parameters consistently evolve to similar values among best cues from Pareto fronts."""
        if not self.all_best_cues_params_from_pareto:
            print("No best cue parameter data available from Pareto fronts for convergence analysis.")
            return {}

        df_params = pd.DataFrame(self.all_best_cues_params_from_pareto, columns=self.param_names)
        param_stats = {}

        for param_name in self.param_names:
            if param_name not in df_params.columns:
                print(f"Warning: Parameter {param_name} not found in DataFrame columns.")
                continue
            values = df_params[param_name].dropna().values 
            if len(values) < 2:
                # Ensure values are native Python types for JSON serialization
                mean_val = float(np.mean(values)) if len(values) > 0 else None
                min_val = float(np.min(values)) if len(values) > 0 else None
                max_val = float(np.max(values)) if len(values) > 0 else None
                param_stats[param_name] = {
                    'mean': mean_val, 
                    'std': 0.0, 
                    'min_observed': min_val, 
                    'max_observed': max_val, 
                    'convergence_strength': 1.0 if len(values) > 0 else 0.0
                }
                continue

            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            min_obs_val = float(np.min(values))
            max_obs_val = float(np.max(values))
            
            param_config_range = self.param_ranges.get(param_name)
            convergence_strength_val = 0.0
            if param_config_range:
                range_min, range_max = param_config_range
                defined_range_width = range_max - range_min
                if defined_range_width == 0: 
                    convergence_strength_val = 1.0 if std_val == 0 else 0.0
                else:
                    convergence_strength_val = 1.0 - (std_val / defined_range_width)
            else: 
                observed_range_width = max_obs_val - min_obs_val
                convergence_strength_val = 1.0 - (std_val / observed_range_width) if observed_range_width > 0 else (1.0 if std_val == 0 else 0.0)
            
            param_stats[param_name] = {
                'mean': mean_val,
                'std': std_val,
                'min_observed': min_obs_val,
                'max_observed': max_obs_val,
                'convergence_strength': float(max(0.0, min(1.0, convergence_strength_val))) # Ensure native float and clamped
            }
        
        strong_patterns = {
            param: stats for param, stats in param_stats.items()
            if stats.get('convergence_strength', 0) > 0.7 
        }
        return strong_patterns

    def analyze_fitness_progression(self, objective_index=0) -> Dict:
        """Analyzes how a specific objective (e.g., max touch_rate) progressed over generations across runs."""
        fitness_progress_all_runs = []
        for run_data in self.runs_data:
            if run_data.get('status') != 'completed': continue
            logbook = run_data.get('logbook') # This needs to be saved from the deap run
            if logbook and isinstance(logbook, list) and len(logbook) > 0:
                # Assuming stats are registered with keys like 'max_touch_rate' or general 'max' for single obj
                # For multi-objective, we might look at the max of the chosen objective_index
                try:
                    # For multi-objective stats as registered in evolve_visual_cues_multi_objective:
                    if objective_index == 0 and 'max_touch_rate' in logbook[0]:
                        obj_progression = [entry['max_touch_rate'] for entry in logbook if 'max_touch_rate' in entry]
                    elif objective_index == 0 and 'max' in logbook[0] and isinstance(logbook[0]['max'], (float, int)):
                        # Fallback for simple stats if only one 'max' is logged (e.g. from single-obj test runs)
                        obj_progression = [entry['max'] for entry in logbook if 'max' in entry]
                    elif 'max' in logbook[0] and isinstance(logbook[0]['max'], (list, tuple)) and len(logbook[0]['max']) > objective_index:
                        obj_progression = [entry['max'][objective_index] for entry in logbook if 'max' in entry and len(entry['max']) > objective_index]
                    else:
                        # Try to find if fitness values are directly logged
                        obj_progression = [entry['fit'][objective_index] for entry in logbook if 'fit' in entry and len(entry['fit']) > objective_index]
                    
                    if obj_progression:
                        fitness_progress_all_runs.append(obj_progression)
                except (KeyError, IndexError, TypeError) as e:
                    print(f"Could not extract fitness progression for objective {objective_index} from logbook in segment {run_data.get('segment_name')}. Error: {e}") 
        
        if not fitness_progress_all_runs:
            return {"error": f"No logbook data with fitness for objective index {objective_index} found."}

        min_gens = min(len(run_fitness) for run_fitness in fitness_progress_all_runs) if fitness_progress_all_runs else 0
        if min_gens == 0:
            return {"error": "Logbook data is empty or inconsistent."}
            
        avg_fitness_progression = []
        for i in range(min_gens):
            avg_fitness_progression.append(np.mean([run_fitness[i] for run_fitness in fitness_progress_all_runs]))
            
        return {
            f"avg_objective_{objective_index}_progression": avg_fitness_progression,
            "num_runs_analyzed": len(fitness_progress_all_runs),
            "generations_analyzed": min_gens
        }

# Example usage (if run directly, requires mock or real data)
if __name__ == '__main__':
    # Mockup of what `evolution_runs_data` might look like, loaded from JSON for example
    # This data would come from the output of `run_segmented_evolutions.py` (segmented_evolution_results.json)
    mock_evolution_runs_data = [
        {
            "segment_name": "young_gamers",
            "status": "completed",
            "best_cues_pareto_front": [
                VisualCue(glow=0.8, pulse_hz=3.5, edge=0.3, color_hue=120, particle_density=0.7, animation_type=4).__dict__,
                VisualCue(glow=0.7, pulse_hz=3.8, edge=0.4, color_hue=100, particle_density=0.6, animation_type=1).__dict__
            ],
            "logbook": [{'gen': i, 'max_touch_rate': 0.6 + i*0.015, 'avg_touch_rate': 0.5+i*0.01} for i in range(15)] # Mock logbook
        },
        {
            "segment_name": "seniors_low_va",
            "status": "completed",
            "best_cues_pareto_front": [
                VisualCue(glow=0.4, pulse_hz=1.2, edge=0.8, color_hue=240, particle_density=0.1, animation_type=2).__dict__
            ],
            "logbook": [{'gen': i, 'max_touch_rate': 0.4 + i*0.02, 'avg_touch_rate': 0.3+i*0.015} for i in range(15)]
        },
        {
            "segment_name": "no_users_test",
            "status": "skipped_no_users",
            "best_cues_pareto_front": []
        }
    ]

    print("Testing AffordancePatternMiner...")
    miner = AffordancePatternMiner(mock_evolution_runs_data)
    
    print("\n--- Convergence Analysis of Best Cues from Pareto Fronts ---")
    convergence_patterns = miner.analyze_convergence_of_best_cues()
    if convergence_patterns:
        for param, stats in convergence_patterns.items():
            print(f"  Parameter: {param}")
            for stat_name, stat_value in stats.items():
                print(f"    {stat_name}: {stat_value:.3f}")
    else:
        print("No strong convergence patterns found or no data.")

    print("\n--- Fitness Progression Analysis (Objective 0: Touch Rate) ---")
    fitness_prog = miner.analyze_fitness_progression(objective_index=0)
    if "error" in fitness_prog:
        print(f"Error: {fitness_prog['error']}")
    else:
        print(f"  Avg Max Touch Rate per Gen (first 10 gens): {fitness_prog.get(f'avg_objective_0_progression', [])[:10]}")
        print(f"  Num Runs Analyzed: {fitness_prog.get('num_runs_analyzed')}")
        print(f"  Generations Analyzed: {fitness_prog.get('generations_analyzed')}") 