import random
from deap import base, creator, tools, algorithms
from typing import List, Tuple, Dict
import numpy as np

# Assuming VisualCue, SyntheticUser, create_synthetic_user_population, 
# and calculate_overall_fitness are in a sibling file synthetic_users.py
# For imports from sibling modules in a package, use relative imports:
from .synthetic_users import VisualCue, SyntheticUser, create_synthetic_user_population, calculate_overall_fitness

# Define parameter ranges
PARAM_RANGES = {
    "glow": (0.0, 1.0),
    "pulse_hz": (0.5, 5.0),
    "edge": (0.0, 1.0),
}

# Order of parameters in the individual list
PARAM_ORDER = ["glow", "pulse_hz", "edge"]

# Create fitness and individual types
# These should be created only once
if not hasattr(creator, "FitnessMax"): # Check if already created
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"): # Check if already created
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator: generates a float in the range for each parameter
for i, param_name in enumerate(PARAM_ORDER):
    min_val, max_val = PARAM_RANGES[param_name]
    toolbox.register(f"attr_{param_name}", random.uniform, min_val, max_val)

# Structure initializers
# Individual: list of attributes generated according to PARAM_ORDER
individual_attrs = [getattr(toolbox, f"attr_{param_name}") for name in PARAM_ORDER]
toolbox.register("individual", tools.initCycle, creator.Individual, tuple(individual_attrs), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def individual_to_visual_cue(individual: List[float]) -> VisualCue:
    """Converts a DEAP individual (list of floats) to a VisualCue object."""
    kwargs = {param_name: individual[i] for i, param_name in enumerate(PARAM_ORDER)}
    return VisualCue(**kwargs)

# Global cache for synthetic user population
USER_POPULATION_CACHE: List[SyntheticUser] = []

def initialize_user_population_cache(num_users: int):
    global USER_POPULATION_CACHE
    # Only recreate if significantly different or not set
    if not USER_POPULATION_CACHE or len(USER_POPULATION_CACHE) != num_users:
        print(f"Initializing user population cache with {num_users} users.")
        USER_POPULATION_CACHE = create_synthetic_user_population(num_users)

def deap_evaluate(individual: List[float]) -> Tuple[float]:
    """
    Evaluation function for DEAP.
    Takes an individual (list of floats) and returns its fitness as a tuple.
    """
    if not USER_POPULATION_CACHE:
        # Fallback: initialize with a default number if not called explicitly
        print("Warning: USER_POPULATION_CACHE not initialized. Initializing with 100 users.")
        initialize_user_population_cache(100)
    
    cue = individual_to_visual_cue(individual)
    fitness_score = calculate_overall_fitness(cue, USER_POPULATION_CACHE)
    return (fitness_score,)

toolbox.register("evaluate", deap_evaluate)

# Genetic operators
toolbox.register("mate", tools.cxBlend, alpha=0.5) # Blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2) # Gaussian mutation

# Decorator to ensure mutated/crossed-over values stay within defined bounds
def check_bounds_decorator(param_order_list, param_ranges_dict):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for individual_index, indiv in enumerate(offspring):
                # Ensure we don't go beyond the actual length of the individual
                max_genes = min(len(indiv), len(param_order_list))
                for gene_index in range(max_genes):
                    if gene_index < len(param_order_list):
                        param_name = param_order_list[gene_index]
                        if param_name in param_ranges_dict:
                            min_val, max_val = param_ranges_dict[param_name]
                            offspring[individual_index][gene_index] = max(min_val, min(indiv[gene_index], max_val))
            return offspring
        return wrapper
    return decorator

# Apply the decorator to mate and mutate operations
# Pass PARAM_ORDER and PARAM_RANGES to the decorator factory
toolbox.decorate("mate", check_bounds_decorator(PARAM_ORDER, PARAM_RANGES))
toolbox.decorate("mutate", check_bounds_decorator(PARAM_ORDER, PARAM_RANGES))

toolbox.register("select", tools.selTournament, tournsize=3)

# (The evolve_visual_cues function will be added later or in a more complete version)

# --- Update Parameter Definitions for Expanded VisualCue ---
NEW_PARAM_RANGES = {
    "glow": (0.0, 1.0),
    "pulse_hz": (0.5, 5.0), 
    "edge": (0.0, 1.0),
    "color_hue": (0.0, 360.0),
    "color_saturation": (0.5, 1.0),
    "color_value": (0.7, 1.0),
    "particle_density": (0.0, 1.0), 
    "particle_speed": (0.1, 2.0),   
    "animation_type": (0, 4), # Integer for discrete types: 0=static, 1=pulse, 2=breathe, 3=wave, 4=spiral
    "size_change_amplitude": (0.0, 0.3),
    "blur_amount": (0.0, 1.0),
}

NEW_PARAM_ORDER = [
    "glow", "pulse_hz", "edge", "color_hue", "color_saturation", "color_value",
    "particle_density", "particle_speed", "animation_type", 
    "size_change_amplitude", "blur_amount"
]

# Update global PARAM_RANGES and PARAM_ORDER for other functions in this file if they use them
PARAM_RANGES = NEW_PARAM_RANGES
PARAM_ORDER = NEW_PARAM_ORDER

# Re-register attributes and individual for the new parameters
# Clear existing attributes from toolbox to avoid issues if script is re-run
existing_attrs = [key for key in toolbox.__dict__ if key.startswith("attr_")]
for attr_name in existing_attrs:
    del toolbox.__dict__[attr_name]

for i, param_name in enumerate(NEW_PARAM_ORDER):
    min_val, max_val = NEW_PARAM_RANGES[param_name]
    if param_name == "animation_type": # Integer parameter
        toolbox.register(f"attr_{param_name}", random.randint, min_val, max_val)
    else: # Float parameter
        toolbox.register(f"attr_{param_name}", random.uniform, min_val, max_val)

new_individual_attrs = [getattr(toolbox, f"attr_{param_name}") for name in NEW_PARAM_ORDER]

# Unregister old individual and population if they exist from previous setup
if "individual" in toolbox.__dict__:
    toolbox.unregister("individual")
if "population" in toolbox.__dict__:
    toolbox.unregister("population")

creator.create("FitnessMaxSingle", base.Fitness, weights=(1.0,)) # Ensure FitnessMax is defined if not already
creator.create("IndividualSingle", list, fitness=creator.FitnessMaxSingle) # Ensure Individual is defined

toolbox.register("individual", tools.initCycle, creator.IndividualSingle, tuple(new_individual_attrs), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# The individual_to_visual_cue function should now correctly map the expanded list of parameters
# The deap_evaluate function will use this updated mapping via individual_to_visual_cue
# The check_bounds_decorator will also use the updated global PARAM_RANGES and PARAM_ORDER

# Ensure evaluate, mate, mutate, select are registered with these new Individual types
# (Their core logic doesn't change, but they operate on these new individuals)
# If deap_evaluate was registered, it will use the new individual_to_visual_cue

# (Keep the rest of the file: individual_to_visual_cue, USER_POPULATION_CACHE, 
# initialize_user_population_cache, deap_evaluate, genetic operators, check_bounds_decorator etc.
# The `individual_to_visual_cue` function should implicitly work if PARAM_ORDER is updated globally.)

# Add the evolve_visual_cues function if it's not already present or update it
def evolve_visual_cues(
    num_generations: int = 20, 
    population_size: int = 50, 
    num_users: int = 100, # This will be used by initialize_user_population_cache
    cxpb: float = 0.7, 
    mutpb: float = 0.2
    ) -> VisualCue:
    """
    Runs the evolutionary algorithm to find an optimal VisualCue.
    """
    initialize_user_population_cache(num_users)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # algorithms.eaSimple is suitable for single objective
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=num_generations,
                                       stats=stats, halloffame=hof, verbose=True)
    
    print("Evolution Logbook (Single Objective):")
    for record in logbook:
        print(record)

    if hof and hof[0].fitness.valid:
        best_individual_params = list(hof[0])
        best_cue = individual_to_visual_cue(best_individual_params)
        print("\nBest Individual Found (Single Objective):")
        print(f"  Parameters: {best_individual_params}")
        print(f"  VisualCue: {best_cue}")
        print(f"  Fitness: {hof[0].fitness.values[0]}")
        return best_cue
    else:
        print("No valid best individual found in Hall of Fame. Selecting best from final population.")
        # Fallback: return the best from the final population if HOF is empty or invalid
        valid_pop = [ind for ind in pop if ind.fitness.valid]
        if not valid_pop:
            print("Error: Final population has no individuals with valid fitness.")
            return individual_to_visual_cue([random.uniform(PARAM_RANGES[p][0], PARAM_RANGES[p][1]) if p != 'animation_type' else random.randint(PARAM_RANGES[p][0], PARAM_RANGES[p][1]) for p in PARAM_ORDER]) # Return a random cue

        best_pop_ind = tools.selBest(valid_pop, 1)[0]
        best_cue = individual_to_visual_cue(list(best_pop_ind))
        print("\nBest Individual from Final Population (Single Objective):")
        print(f"  Parameters: {list(best_pop_ind)}")
        print(f"  VisualCue: {best_cue}")
        print(f"  Fitness: {best_pop_ind.fitness.values[0]}")
        return best_cue

if __name__ == '__main__':
    print("Starting evolutionary visual cue discovery (with expanded params)...")
    best_discovered_cue = evolve_visual_cues(num_generations=10, population_size=20, num_users=30)
    print(f"\nProcess finished. Best cue: {best_discovered_cue}")

# --- Update Fitness and Individual for Multi-objective ---
# Clear previous single-objective fitness and individual if they exist to avoid DEAP errors on re-definition
if hasattr(creator, "FitnessMaxSingle"): del creator.FitnessMaxSingle
if hasattr(creator, "IndividualSingle"): del creator.IndividualSingle

if hasattr(creator, "FitnessMax"): # From original Day 0 setup
    del creator.FitnessMax 

# If Individual exists and we are about to define IndividualMulti, it's safe to delete the old one.
if hasattr(creator, "Individual"): 
    del creator.Individual

# Weights: (touch_rate, accessibility_score, complexity_score)
# Maximize touch_rate, Maximize accessibility_score, Minimize complexity (hence -1.0 for complexity weight)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 0.5, -0.2)) 
creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)

# --- Re-register toolbox components with new IndividualMulti type ---
toolbox.unregister("individual") # Unregister previous single-objective individual if any
toolbox.unregister("population") # Unregister previous population if any

# Re-register attributes if they were cleared or to be certain
existing_attrs = [key for key in toolbox.__dict__ if key.startswith("attr_")]
if not existing_attrs or len(existing_attrs) != len(PARAM_ORDER):
    for i, param_name in enumerate(PARAM_ORDER):
        min_val, max_val = PARAM_RANGES[param_name]
        if param_name == "animation_type":
            toolbox.register(f"attr_{param_name}", random.randint, min_val, max_val)
        else:
            toolbox.register(f"attr_{param_name}", random.uniform, min_val, max_val)

individual_attrs_multi = [getattr(toolbox, f"attr_{name}") for name in PARAM_ORDER]
toolbox.register("individual", tools.initCycle, creator.IndividualMulti, tuple(individual_attrs_multi), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- Multi-Objective Evaluation Function ---
def evaluate_multi_objective(individual: List[float]) -> Tuple[float, float, float]:
    global CURRENT_EVALUATION_USER_LIST
    if not CURRENT_EVALUATION_USER_LIST:
        # This should not happen if set_current_evaluation_user_list is called before evolution
        print("Error: CURRENT_EVALUATION_USER_LIST is not set for evaluation.")
        return (0.0, 0.0, 1.0) # Return poor fitness
        
    cue = individual_to_visual_cue(individual) 
    
    touch_rate = calculate_overall_fitness(cue, CURRENT_EVALUATION_USER_LIST)
    
    accessibility_users = [u for u in CURRENT_EVALUATION_USER_LIST if hasattr(u, 'visual_acuity_factor') and u.visual_acuity_factor < 0.8]
    if not accessibility_users:
        accessibility_score = 0.0 
    else:
        accessibility_score = calculate_overall_fitness(cue, accessibility_users)
    
    complexity = cue.complexity_score 
    
    return (touch_rate, accessibility_score, complexity)

if "evaluate" in toolbox.__dict__: toolbox.unregister("evaluate")
toolbox.register("evaluate", evaluate_multi_objective)

# --- Update Selection Algorithm for Multi-Objective ---
if "select" in toolbox.__dict__: toolbox.unregister("select")
toolbox.register("select", tools.selNSGA2)

# Mate and Mutate operators remain the same, but check_bounds_decorator should still apply
# Ensure check_bounds_decorator is defined as in the previous step (using PARAM_ORDER and PARAM_RANGES)
toolbox.decorate("mate", check_bounds_decorator(PARAM_ORDER, PARAM_RANGES))
toolbox.decorate("mutate", check_bounds_decorator(PARAM_ORDER, PARAM_RANGES))

# --- Update evolve_visual_cues_multi_objective to accept users and set the global --- 
def evolve_visual_cues_multi_objective(
    users_for_this_run: List[SyntheticUser], # New parameter
    num_generations: int = 30, 
    population_size: int = 100, 
    # num_users parameter is now implicitly len(users_for_this_run)
    cxpb: float = 0.7, 
    mutpb: float = 0.2
    ) -> Tuple[List[VisualCue], List[Dict]]: # Return cues and logbook
    
    # Set the global user list for the evaluation function for this specific run
    set_current_evaluation_user_list(users_for_this_run)
    if not users_for_this_run:
        print("Warning: evolve_visual_cues_multi_objective called with empty user list. Aborting run.")
        return [], []

    pop = toolbox.population(n=population_size)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg_touch_rate", lambda x: np.mean([fit[0] for fit in x]))
    stats.register("avg_accessibility", lambda x: np.mean([fit[1] for fit in x]))
    stats.register("avg_complexity", lambda x: np.mean([fit[2] for fit in x]))
    stats.register("max_touch_rate", lambda x: np.max([fit[0] for fit in x]))
    
    # Capture logbook explicitly
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    hof.update(pop) # Update HOF with initial population
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream) # Print initial generation stats

    # Begin the generational process
    for gen in range(1, num_generations + 1):
        offspring = toolbox.select(pop, len(pop)) # NSGA-II selection
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        hof.update(offspring) # Update HOF with new offspring
        pop = toolbox.select(pop + offspring, population_size) # Environmental selection from combined population
        
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream) # Print stats for current generation

    print(f"\nPareto Front (Non-Dominated Solutions): {len(hof)} individuals")
    discovered_cues = []
    for i, ind_params in enumerate(hof): 
        cue = individual_to_visual_cue(list(ind_params))
        discovered_cues.append(cue)
        # print(f"  Solution {i}: {cue}") # Already printed by logbook stream effectively
        # print(f"    Fitness (TouchRate, Accessibility, Complexity): {ind_params.fitness.values}")
    
    return discovered_cues, logbook.chapters["fitness"].select("gen", "max_touch_rate", "avg_touch_rate", "avg_accessibility", "avg_complexity") # Return a serializable log

# Update the main execution block for testing multi-objective
if __name__ == '__main__':
    import numpy as np 
    print("Starting multi-objective evolutionary visual cue discovery...")
    
    # Create a sample user population for the direct test run
    test_users = create_realistic_population(50) # from synthetic_users
    if not test_users:
        print("Failed to create test users. Exiting.")
    else:
        pareto_front_cues, logs = evolve_visual_cues_multi_objective(
            users_for_this_run=test_users,
            num_generations=15, 
            population_size=30
        )
        print(f"\nProcess finished. Found {len(pareto_front_cues)} non-dominated solutions.")
        if pareto_front_cues:
            print("One of the best solutions (example):", pareto_front_cues[0])
            print("Fitness values:", pareto_front_cues[0].fitness.values if hasattr(pareto_front_cues[0], 'fitness') else "N/A")
        print("\nLogs from run:")
        for log_entry in logs:
            print(log_entry)

# Update individual_to_visual_cue and deap_evaluate to not rely on a global USER_POPULATION_CACHE
# individual_to_visual_cue remains the same as it's independent of user cache

# The deap_evaluate function (and its multi-objective version) needs access to the specific user list for that run.
# This is best handled by passing it to the evolve function, which then sets it for its internal evaluation calls.
# For DEAP, one common way is to have the evaluate function be a closure or use a global that is set per run.
# We will use a global that the main evolution function for a given run will set. This is simpler than extensive refactoring
# of DEAP's evaluation call signature if not directly supported.

CURRENT_EVALUATION_USER_LIST: List[SyntheticUser] = []

def set_current_evaluation_user_list(users: List[SyntheticUser]):
    global CURRENT_EVALUATION_USER_LIST
    CURRENT_EVALUATION_USER_LIST = users
