import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd # Added for potential heatmap data structuring
from typing import List, Dict, Any

# If VisualCue objects are passed directly from DEAP HallOfFame/ParetoFront
# from .synthetic_users import VisualCue 

def plot_pareto_front_3d(
    pareto_front_individuals: List[Any], 
    objectives:List[str]=['Objective 1 (e.g., Touch Rate)', 'Objective 2 (e.g., Accessibility)', 'Objective 3 (e.g., -Complexity)'],
    title: str = 'Pareto Front of VR Affordance Cues',
    save_path: str = None
    ) -> plt.Figure:
    """
    Creates a 3D scatter plot of the Pareto front.
    Args:
        pareto_front_individuals: A list of DEAP individuals from the ParetoFront.
                                  Each individual is expected to have a .fitness.values attribute (tuple).
        objectives: List of names for the objectives (for axis labels).
        title: Title of the plot.
        save_path: Optional path to save the figure.
    Returns:
        matplotlib.figure.Figure object
    """
    if not pareto_front_individuals:
        print("Pareto front is empty, cannot plot.")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.text2D(0.5, 0.5, "Pareto front data is empty.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title)
        if save_path:
            plt.savefig(save_path)
            print(f"Empty Pareto plot saved to {save_path}")
        return fig

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract objective values - ensure fitness values are present and correct
    try:
        x = [ind.fitness.values[0] for ind in pareto_front_individuals]
        y = [ind.fitness.values[1] for ind in pareto_front_individuals]
        # Objective 3 (complexity) has a negative weight, so its raw value might be positive.
        # If it was stored as a positive value and weight was negative, we might need to invert for plotting if desired.
        # Or plot as is and label axis as "-Complexity" or "Cost (lower is better)"
        z = [ind.fitness.values[2] for ind in pareto_front_individuals]
    except (AttributeError, IndexError) as e:
        print(f"Error accessing fitness values from individuals: {e}. Ensure individuals have .fitness.values as a tuple of 3 floats.")
        ax.text2D(0.5, 0.5, "Error accessing fitness data.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig

    scatter = ax.scatter(x, y, z, c='r', s=50, depthshade=True, label='Pareto Optimal Solutions')
    
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.set_zlabel(objectives[2]) # If complexity is minimized, this axis represents that value directly.
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Pareto front plot saved to {save_path}")
    return fig

def plot_parameter_convergence(
    convergence_data: Dict[str, Dict[str, float]], 
    title: str = 'Parameter Convergence Strength',
    save_path: str = None
    ) -> plt.Figure:
    """
    Creates a horizontal bar plot showing parameter convergence strength.
    Args:
        convergence_data: Dictionary from AffordancePatternMiner.analyze_convergence_of_best_cues.
                          Expected format: {'param_name': {'convergence_strength': value, ...}}
        title: Title of the plot.
        save_path: Optional path to save the figure.
    Returns:
        matplotlib.figure.Figure object
    """
    if not convergence_data:
        print("Convergence data is empty, cannot plot.")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.text(0.5, 0.5, "Convergence data is empty.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title)
        if save_path:
            plt.savefig(save_path)
            print(f"Empty convergence plot saved to {save_path}")
        return fig

    params = list(convergence_data.keys())
    convergence_strengths = [p_data.get('convergence_strength', 0) for p_data in convergence_data.values()]
    mean_values = [p_data.get('mean', 0) for p_data in convergence_data.values()]

    df_plot = pd.DataFrame({
        'Parameter': params,
        'Strength': convergence_strengths,
        'MeanValue': mean_values
    }).sort_values(by='Strength', ascending=True)

    fig, ax = plt.subplots(figsize=(12, len(params) * 0.5 + 1))
    bars = ax.barh(df_plot['Parameter'], df_plot['Strength'], color=sns.color_palette("viridis", len(params)))
    ax.set_xlabel('Convergence Strength (0 to 1, higher is stronger convergence)')
    ax.set_ylabel('Cue Parameter')
    ax.set_title(title)
    ax.set_xlim(0, 1.05) # Strength is normalized 0-1

    # Add text for mean value on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_text = f'{df_plot["MeanValue"].iloc[i]:.2f}'
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2., label_text, 
                ha='left', va='center', color='black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Parameter convergence plot saved to {save_path}")
    return fig

if __name__ == '__main__':
    print("Testing visualization functions...")

    # --- Test plot_pareto_front_3d --- 
    # Mock DEAP individuals with multi-objective fitness
    class MockFitnessMulti:
        def __init__(self, values):
            self.values = values

    class MockIndividualMulti:
        def __init__(self, fitness_values):
            self.fitness = MockFitnessMulti(fitness_values)

    mock_pareto_front = [
        MockIndividualMulti((0.8, 0.7, 0.2)),
        MockIndividualMulti((0.75, 0.8, 0.3)),
        MockIndividualMulti((0.85, 0.65, 0.15)),
        MockIndividualMulti((0.6, 0.85, 0.4)),
        MockIndividualMulti((0.9, 0.5, 0.1)),
    ]
    fig_pareto = plot_pareto_front_3d(mock_pareto_front, save_path="mock_pareto_front.png")
    # plt.show() # Uncomment to display if running locally and not in a headless environment
    plt.close(fig_pareto) # Close the figure to free memory

    # --- Test plot_parameter_convergence --- 
    mock_convergence_data = {
        'glow': {'mean': 0.65, 'std': 0.1, 'min_observed': 0.4, 'max_observed': 0.9, 'convergence_strength': 0.8},
        'pulse_hz': {'mean': 2.45, 'std': 0.3, 'min_observed': 1.8, 'max_observed': 3.1, 'convergence_strength': 0.9},
        'edge': {'mean': 0.30, 'std': 0.2, 'min_observed': 0.05, 'max_observed': 0.7, 'convergence_strength': 0.6},
        'color_hue': {'mean': 180.5, 'std': 60.0, 'min_observed': 90.0, 'max_observed': 270.0, 'convergence_strength': 0.5},
         'particle_density': {'mean': 0.22, 'std': 0.15, 'min_observed': 0.0, 'max_observed': 0.5, 'convergence_strength': 0.75}
    }
    fig_convergence = plot_parameter_convergence(mock_convergence_data, save_path="mock_parameter_convergence.png")
    # plt.show() # Uncomment to display
    plt.close(fig_convergence)

    print("\nMock plots generated and saved as mock_pareto_front.png and mock_parameter_convergence.png (if matplotlib is correctly configured).")
    print("Visualization test finished.") 