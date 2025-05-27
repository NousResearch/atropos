import numpy as np
from typing import List, Dict, Any

# Assuming VisualCue and other necessary data structures might be imported for type hinting or context
# from .synthetic_users import VisualCue, SyntheticUser 
# from .pattern_miner import AffordancePatternMiner # If it provides structured data
# from .analysis_tools import discover_interaction_effects # If it provides structured data

class InsightGenerator:
    def __init__(self, 
                 convergence_patterns: Dict[str, Dict[str, float]] = None, 
                 interaction_effects: List[Dict[str, Any]] = None, 
                 segmented_best_cues: Dict[str, List[Dict]] = None, # e.g. {'young_gamers': [cue_dict1, cue_dict2]}
                 all_runs_summary_data: List[Dict] = None # For broader stats like overall best cues
                 ):
        """
        Initializes the InsightGenerator with various analyses results.
        Args:
            convergence_patterns: Output from AffordancePatternMiner.analyze_convergence_of_best_cues.
            interaction_effects: Output from discover_interaction_effects.
            segmented_best_cues: A dictionary where keys are segment names and values are lists of best cue dicts for that segment.
            all_runs_summary_data: List of dicts, each summarizing a run (could include best_cue, fitness, config).
        """
        self.convergence_patterns = convergence_patterns if convergence_patterns else {}
        self.interaction_effects = interaction_effects if interaction_effects else []
        self.segmented_best_cues = segmented_best_cues if segmented_best_cues else {}
        self.all_runs_summary_data = all_runs_summary_data if all_runs_summary_data else []

    def generate_key_discoveries(self) -> List[str]:
        insights = []

        # --- 1. Insights from Convergence Patterns --- 
        if self.convergence_patterns:
            insights.append("**Key Convergence Patterns:**")
            for param, stats in self.convergence_patterns.items():
                if stats.get('convergence_strength',0) > 0.7:
                    insights.append(
                        f"  - Strong convergence for '{param}': Mean={stats['mean']:.2f} (StdDev: {stats['std']:.2f}, Range: [{stats['min_observed']:.2f}-{stats['max_observed']:.2f}]). "
                        f"Strength: {stats['convergence_strength']:.2f}. This suggests a universally preferred value or narrow optimal range."
                    )
                    # Specific example from plan for pulse_hz
                    if param == 'pulse_hz' and 2.0 <= stats['mean'] <= 3.0 and stats['convergence_strength'] > 0.75:
                        insights.append(
                            f"    * Universal Resonance Frequency: Optimal pulse_hz converges to {stats['mean']:.1f}Hz, potentially aligning with human cognitive/physiological rhythms (e.g., resting breathing rate)."
                        )
        else:
            insights.append("No strong convergence patterns identified across all runs.")

        # --- 2. Insights from Interaction Effects --- 
        if self.interaction_effects:
            insights.append("\n**Significant Interaction Effects (Synergies/Antagonisms):**")
            for effect in self.interaction_effects:
                param_pair = effect['parameter_pair']
                corr = effect['correlation_with_target']
                effect_type = effect['effect_type']
                insights.append(
                    f"  - Parameters '{param_pair[0]}' and '{param_pair[1]}' show a {effect_type} effect. (Correlation: {corr:.2f})"
                )
                if effect_type == 'synergistic' and 'glow' in param_pair and 'edge' in param_pair and corr > 0.3:
                     insights.append("    * Edge+Glow Synergy: The combination of edge highlights and glow is more effective than expected from individual effects.")
        else:
            insights.append("No significant 2-way interaction effects found with current thresholds.")

        # --- 3. Insights from Segmented Evolution --- (Example based on plan)
        if self.segmented_best_cues:
            insights.append("\n**Demographic-Specific Cue Preferences:**")
            young_cues = self.segmented_best_cues.get('young_gamers', [])
            senior_cues = self.segmented_best_cues.get('seniors_low_va', []) # Match key from run_segmented_evolutions
            # Add more segments as defined in your segmented runs

            if young_cues and senior_cues:
                # Extract average parameter values for key differentiating params
                avg_young_glow = np.mean([cue.get('glow', 0.5) for cue in young_cues]) if young_cues else 0
                avg_senior_glow = np.mean([cue.get('glow', 0.5) for cue in senior_cues]) if senior_cues else 0
                avg_young_edge = np.mean([cue.get('edge', 0.5) for cue in young_cues]) if young_cues else 0
                avg_senior_edge = np.mean([cue.get('edge', 0.5) for cue in senior_cues]) if senior_cues else 0
                avg_young_particles = np.mean([cue.get('particle_density', 0) for cue in young_cues]) if young_cues else 0
                avg_senior_particles = np.mean([cue.get('particle_density', 0) for cue in senior_cues]) if senior_cues else 0

                if avg_senior_glow < avg_young_glow * 0.7 and avg_senior_edge > avg_young_edge * 1.1:
                    insights.append(
                        f"  - Age-Glow/Edge Shift: Younger users tend to prefer higher glow (avg: {avg_young_glow:.2f}) while seniors favor stronger edge highlights (avg_edge for seniors: {avg_senior_edge:.2f} vs {avg_young_edge:.2f} for young)."
                    )
                if avg_young_particles > 0.3 and avg_senior_particles < 0.15:
                    insights.append(
                        f"  - Age-Complexity (Particles): Younger users show preference for cues with more particle effects (avg_density: {avg_young_particles:.2f}) compared to seniors (avg_density: {avg_senior_particles:.2f})."
                    )
            
            east_asia_cues = self.segmented_best_cues.get('east_asia_novice', [])
            if east_asia_cues:
                avg_ea_hue = np.mean([cue.get('color_hue', 180) for cue in east_asia_cues])
                # Example: if red (0-30 or 330-360) is significantly less common for best EA cues
                num_red_cues = sum(1 for cue in east_asia_cues if cue.get('color_hue', 180) <= 30 or cue.get('color_hue', 180) >= 330)
                if len(east_asia_cues) > 0 and (num_red_cues / len(east_asia_cues)) < 0.1: # If less than 10% are red
                    insights.append(
                        f"  - Cultural Color Preference (East Asia): Initial findings suggest hues in the red spectrum might be less optimal for East Asian novices compared to other hues (avg hue: {avg_ea_hue:.0f})."
                    )
        else:
            insights.append("No specific segmented cue data provided for demographic insights.")
        
        # --- 4. Overall Best Performing Cues (from all_runs_summary_data) ---
        # This requires fitness values to be associated with cues in all_runs_summary_data
        # Assuming all_runs_summary_data contains dicts with 'best_cues_pareto_front' (list of cue dicts)
        # and each cue dict within has its fitness values, e.g. {'fitness_scores': (touch_rate, access, complex)}
        # For simplicity, let's find the cue with the highest touch_rate from all Pareto fronts.
        overall_top_cue = None
        max_touch_rate = -1

        if self.all_runs_summary_data:
            insights.append("\n**Highlights from Overall Runs:**")
            for run_summary in self.all_runs_summary_data:
                if run_summary.get('status') == 'completed':
                    for cue_data in run_summary.get('best_cues_pareto_front',[]):
                        # Assuming fitness values are stored with the cue_data or can be re-derived
                        # For multi-objective, the 'fitness_values' would be a tuple
                        # Let's assume the first objective was touch_rate
                        fitness_values = cue_data.get('fitness_values') # This needs to be added during result saving
                        if fitness_values and isinstance(fitness_values, (list, tuple)) and len(fitness_values) > 0:
                            current_touch_rate = fitness_values[0]
                            if current_touch_rate > max_touch_rate:
                                max_touch_rate = current_touch_rate
                                overall_top_cue = cue_data
        
        if overall_top_cue:
            insights.append(f"  - Top Touch-Rate Cue Found: {overall_top_cue} (Touch Rate: {max_touch_rate:.3f})")

        # --- 5. Example "Breakthrough" Insights (from plan, highly speculative) ---
        # These would require more specific analyses or different types of data.
        # insights.append("\n**Potential Breakthrough Concepts (Speculative):**")
        # if some_condition_for_haptic_priming:
        #     insights.append("  - Haptic Priming Potential: Certain visual cue combinations (e.g., slow large pulse, soft blue glow) show patterns that might correlate with anticipation of specific haptic feedback.")
        # if some_condition_for_7_param_grammar:
        #     insights.append("  - Towards a Minimal Grammar: Analysis suggests a core set of ~5-7 parameters (e.g., pulse_hz, glow, edge, dominant_hue, animation_type) can define a large portion of effective affordances.")

        return insights

# Example Usage:
if __name__ == '__main__':
    # Mock data - in reality, this would come from PatternMiner and other analyses
    mock_convergence = {
        'pulse_hz': {'mean': 2.35, 'std': 0.2, 'min_observed': 2.0, 'max_observed': 2.8, 'convergence_strength': 0.85},
        'glow': {'mean': 0.38, 'std': 0.1, 'min_observed': 0.2, 'max_observed': 0.6, 'convergence_strength': 0.72}
    }
    mock_interactions = [
        {'parameter_pair': ('glow', 'edge'), 'correlation_with_target': 0.45, 'effect_type': 'synergistic'}
    ]
    mock_segmented_cues = {
        'young_gamers': [
            {'glow': 0.7, 'pulse_hz': 3.8, 'edge': 0.4, 'particle_density': 0.6, 'fitness_values': (0.75,0.6,0.5)},
            {'glow': 0.8, 'pulse_hz': 3.5, 'edge': 0.3, 'particle_density': 0.7, 'fitness_values': (0.78,0.65,0.6)}
            ],
        'seniors_low_va': [
            {'glow': 0.3, 'pulse_hz': 1.2, 'edge': 0.85, 'particle_density': 0.1, 'fitness_values': (0.6,0.68,0.2)}
            ],
        'east_asia_novice': [
            {'glow': 0.5, 'pulse_hz': 2.0, 'edge': 0.6, 'color_hue': 150.0, 'fitness_values': (0.55,0.5,0.3)}
        ]
    }
    mock_all_runs_summary = [
        {'status': 'completed', 'segment_name': 'run1', 'best_cues_pareto_front': [mock_segmented_cues['young_gamers'][1]], 'config': {}, 'logbook': []}
    ]

    print("Testing InsightGenerator...")
    insight_gen = InsightGenerator(
        convergence_patterns=mock_convergence, 
        interaction_effects=mock_interactions, 
        segmented_best_cues=mock_segmented_cues,
        all_runs_summary_data=mock_all_runs_summary
    )
    
    key_discoveries = insight_gen.generate_key_discoveries()
    print("\n--- Generated Key Discoveries ---")
    for discovery in key_discoveries:
        print(discovery) 