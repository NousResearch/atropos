import pandas as pd
import numpy as np
from typing import List, Dict, Any
from itertools import combinations

# Assuming PARAM_ORDER is available from visual_cue_evolver for default param names
# If this script is run standalone, you might need to define it or pass it.
from .visual_cue_evolver import PARAM_ORDER as DEFAULT_PARAM_ORDER

def discover_interaction_effects(
    experiment_results_df: pd.DataFrame, 
    parameter_columns: List[str] = None,
    target_metric_column: str = 'fitness' # or 'touch_rate' or specific objective like 'obj0_touch_rate'
    ) -> List[Dict[str, Any]]:
    """
    Finds synergistic or antagonistic parameter combinations from a DataFrame of experiment results.

    Args:
        experiment_results_df: DataFrame where each row is a run's best cue (or an individual cue),
                               columns include parameter names (e.g., 'glow', 'pulse_hz')
                               and a target metric column (e.g., 'fitness', 'touch_rate').
        parameter_columns: Optional list of parameter column names to consider for interactions.
                           If None, defaults to DEFAULT_PARAM_ORDER.
        target_metric_column: The name of the column containing the fitness/performance metric.

    Returns:
        A list of dictionaries, each describing a significant interaction effect.
    """
    if target_metric_column not in experiment_results_df.columns:
        # Try to infer common fitness/objective column names
        possible_targets = ['fitness', 'touch_rate', 'obj0_touch_rate', 'objective_0']
        found_target = False
        for pt in possible_targets:
            if pt in experiment_results_df.columns:
                target_metric_column = pt
                found_target = True
                print(f"Inferred target metric column: {target_metric_column}")
                break
        if not found_target:
            raise ValueError(f"DataFrame must contain a valid target metric column. Tried: {possible_targets}")

    if parameter_columns is None:
        parameter_columns = [p for p in DEFAULT_PARAM_ORDER if p in experiment_results_df.columns]
    
    # Filter out non-numeric or problematic parameters for simple multiplication-based interaction terms
    valid_params_for_interaction = []
    for p_col in parameter_columns:
        if p_col in experiment_results_df.columns and pd.api.types.is_numeric_dtype(experiment_results_df[p_col]):
            if experiment_results_df[p_col].nunique() > 1: # Ensure there's variance
                 valid_params_for_interaction.append(p_col)
            else:
                print(f"Skipping parameter '{p_col}' for interaction analysis due to no variance.")
        else:
            print(f"Skipping parameter '{p_col}' for interaction analysis as it's non-numeric or missing.")

    interactions = []
    df_copy = experiment_results_df.copy() # Work on a copy

    for param1, param2 in combinations(valid_params_for_interaction, 2):
        interaction_col_name = f'{param1}_x_{param2}'
        df_copy[interaction_col_name] = df_copy[param1] * df_copy[param2]
        
        # Check for sufficient variance in the interaction term and target metric
        if df_copy[interaction_col_name].std() < 1e-6 or df_copy[target_metric_column].std() < 1e-6:
            corr = 0.0 # Avoid correlation with constant series or near-constant
        else:
            try:
                corr = df_copy[interaction_col_name].corr(df_copy[target_metric_column])
                if pd.isna(corr): # Handle cases where correlation might be NaN (e.g. if one series is constant after all)
                    corr = 0.0
            except Exception as e:
                print(f"Could not calculate correlation for {interaction_col_name} and {target_metric_column}: {e}")
                corr = 0.0
        
        # Significance threshold for correlation
        # Plan suggested 0.3, can be adjusted
        if abs(corr) > 0.2: # Slightly lower threshold to catch more potential interactions
            interactions.append({
                'parameter_pair': (param1, param2),
                'interaction_term': interaction_col_name,
                'correlation_with_target': round(corr, 4),
                'effect_type': 'synergistic' if corr > 0 else ('antagonistic' if corr < 0 else 'neutral')
            })
            
    return sorted(interactions, key=lambda x: abs(x['correlation_with_target']), reverse=True)

# Example Usage (if run directly or in a notebook)
if __name__ == '__main__':
    print("Testing discover_interaction_effects...")
    # Create mock data for testing
    num_samples = 100
    mock_data = {
        'glow': np.random.rand(num_samples),
        'pulse_hz': np.random.uniform(0.5, 5.0, num_samples),
        'edge': np.random.rand(num_samples),
        'color_saturation': np.random.rand(num_samples),
        'fitness': np.random.rand(num_samples) # Initial random fitness
    }
    df_mock_results = pd.DataFrame(mock_data)

    # Simulate some interaction effects in the mock data
    # Example 1: glow and edge are synergistic
    df_mock_results['fitness'] += df_mock_results['glow'] * df_mock_results['edge'] * 0.5 
    # Example 2: high pulse_hz and high glow are antagonistic (reduce fitness)
    df_mock_results['fitness'] -= df_mock_results['pulse_hz']/5 * df_mock_results['glow'] * 0.3
    # Normalize fitness to be roughly 0-1 after adding effects
    df_mock_results['fitness'] = (df_mock_results['fitness'] - df_mock_results['fitness'].min()) / \
                                 (df_mock_results['fitness'].max() - df_mock_results['fitness'].min() + 1e-6)


    # Define which parameter columns to check for interactions
    # These should match columns in your DataFrame from evolution results
    param_cols = ['glow', 'pulse_hz', 'edge', 'color_saturation'] 

    interaction_insights = discover_interaction_effects(
        df_mock_results, 
        parameter_columns=param_cols, 
        target_metric_column='fitness'
    )
    
    print("\nInteraction Effects Found:")
    if interaction_insights:
        for insight in interaction_insights:
            print(f"  - Pair: {insight['parameter_pair']}, Correlation: {insight['correlation_with_target']}, Type: {insight['effect_type']}")
    else:
        print("  No significant interaction effects found with current thresholds.") 