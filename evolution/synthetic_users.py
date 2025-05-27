import random
from dataclasses import dataclass, field
from typing import List, Dict
import asyncio
import numpy as np # For sampling from distributions

@dataclass
class VisualCue:
    glow: float = field(default_factory=lambda: random.uniform(0.0, 1.0))
    pulse_hz: float = field(default_factory=lambda: random.uniform(0.5, 5.0))
    edge: float = field(default_factory=lambda: random.uniform(0.0, 1.0))
    
    color_hue: float = field(default_factory=lambda: random.uniform(0.0, 360.0))
    color_saturation: float = field(default_factory=lambda: random.uniform(0.5, 1.0))
    color_value: float = field(default_factory=lambda: random.uniform(0.7, 1.0))
    particle_density: float = field(default_factory=lambda: random.uniform(0.0, 1.0))
    particle_speed: float = field(default_factory=lambda: random.uniform(0.1, 2.0))
    animation_type: int = field(default_factory=lambda: random.randint(0, 4)) 
    size_change_amplitude: float = field(default_factory=lambda: random.uniform(0.0, 0.3))
    blur_amount: float = field(default_factory=lambda: random.uniform(0.0, 1.0))
    
    @property
    def complexity_score(self) -> float:
        score = 0.0
        score += self.glow * 0.1
        score += (self.pulse_hz / 5.0) * 0.1 if self.animation_type == 1 else 0
        score += self.edge * 0.1
        score += self.particle_density * 0.3
        score += self.particle_speed * 0.1
        score += (1 if self.animation_type > 0 else 0) * 0.1
        score += self.size_change_amplitude * 0.2 if self.animation_type == 2 else 0
        score += self.blur_amount * 0.15
        return min(1.0, score)

@dataclass
class SyntheticUser:
    user_id: int
    age: int
    gaming_hours_per_week: float
    vr_experience_level: str  # 'none', 'novice', 'intermediate', 'expert'
    dominant_hand: str  # 'left', 'right'
    visual_acuity_factor: float  # 0.5 (poorer) to 1.5 (better), 1.0 is average
    reaction_time_multiplier: float  # 0.7 (faster) to 1.5 (slower), 1.0 is average
    cultural_region: str  # 'north_america', 'europe', 'east_asia', 'south_asia', 'latin_america', 'africa', 'oceania', 'middle_east'

def create_synthetic_user_population(num_users: int) -> List[SyntheticUser]:
    population = []
    for i in range(num_users):
        age = random.randint(15, 70)
        population.append(SyntheticUser(user_id=i, age=age))
    return population

def evaluate_cue_fitness_for_user(cue: VisualCue, user: SyntheticUser) -> float:
    """Enhanced fitness based on research-backed principles (heuristic model)."""
    
    # Base response varies by age (simulating peak performance/preference around 25)
    # This is a simple quadratic model: peak at 25, declines symmetrically.
    # (age - 25)^2 can be large, so scale it down. Max age diff is (70-25)=45 or (13-25)=-12. 45^2 = 2025.
    # Let's use a gentler linear or capped decline.
    age_offset = abs(user.age - 25)
    age_factor = max(0.3, 1.0 - age_offset * 0.015) # Peak at 1.0, declines by 0.015 per year from 25. Min 0.3.
    
    # VR experience affects cue preference complexity
    complexity_preference_score = 0.0
    if user.vr_experience_level == 'expert':
        # Experts might appreciate well-utilized complexity or be more discerning
        complexity_preference_score = cue.complexity_score * 0.2 # Max +0.2 if complex
    elif user.vr_experience_level == 'novice' or user.vr_experience_level == 'none':
        complexity_preference_score = (1.0 - cue.complexity_score) * 0.2 # Max +0.2 if simple
    else: # intermediate
        complexity_preference_score = (1.0 - abs(cue.complexity_score - 0.5)) * 0.1 # Prefers mid-complexity

    # Cultural color preferences (example based on plan)
    # Assuming color_hue is 0-360
    color_affinity_score = 0.0
    # User plan mentioned 'west' but cultural_region has 'europe', 'north_america'
    # Let's map 'west' to these for the example.
    if user.cultural_region in ['east_asia'] and 330 <= cue.color_hue <= 360 or 0 <= cue.color_hue <= 20: # Reds for East Asia
        color_affinity_score = -0.15  # Penalty for red in East Asia (example)
    elif user.cultural_region in ['europe', 'north_america'] and 200 <= cue.color_hue <= 260: # Blues for West
        color_affinity_score = 0.1  # Bonus for blue in West (example)
    
    # Frequency resonance (2-3Hz breathing rate)
    freq_resonance_score = 0.0
    if 2.0 <= cue.pulse_hz <= 3.0: # Animation type 1 (pulse) or 2 (breathe) should use pulse_hz
        if cue.animation_type in [1, 2]: # Only apply if it's a pulsing/breathing animation
            freq_resonance_score = 0.2
    elif cue.animation_type in [1,2] and (cue.pulse_hz < 1.0 or cue.pulse_hz > 4.0): # Penalize extremes if pulsing/breathing
        freq_resonance_score = -0.1

    # Visual acuity affects edge vs glow preference
    visibility_score = 0.0
    # visual_acuity_factor: 0.5 (poorer) to 1.5 (better), 1.0 is average
    if user.visual_acuity_factor < 0.8: # Poorer acuity
        visibility_score += cue.edge * 0.25 # Stronger reliance on edges
        visibility_score += (1.0 - cue.blur_amount) * 0.15 # Less blur is better
        visibility_score -= cue.particle_density * 0.1 # Particles might be distracting
    else: # Better acuity
        visibility_score += cue.glow * 0.15
        visibility_score += cue.particle_density * 0.05 # Can appreciate subtle particles
        if cue.blur_amount < 0.3: # Prefers sharpness too
            visibility_score += 0.05
            
    # Combine factors with weights. Base fitness can be 0.1 to 0.3 to ensure some base interaction chance.
    # Weights should sum roughly to (1.0 - base_random_range) if fitness is 0-1.
    # Let max possible positive score be around 0.7 from features, plus 0.2 base, plus noise.
    # Current weights: Age (0.2), Complexity (0.2), Color (0.1), Freq (0.2), Visibility (0.25) = 0.95. Good.

    fitness = (
        0.1 + # Base interaction probability
        age_factor * 0.20 + 
        complexity_preference_score + # Max 0.2
        color_affinity_score +      # Max +/- 0.15
        freq_resonance_score +      # Max +0.2
        visibility_score          # Max ~0.25 - 0.4 based on acuity factors
    )
    
    # Add some noise for realism
    fitness += random.gauss(0, 0.03) # Reduced noise std dev
    
    return max(0.0, min(1.0, fitness))

def calculate_overall_fitness(cue: VisualCue, users: List[SyntheticUser]) -> float:
    if not users:
        return 0.0
    total_fitness = sum(evaluate_cue_fitness_for_user(cue, user) for user in users)
    return total_fitness / len(users) if users else 0.0

# --- Additions for Padres Integration ---

# Example of what your client might look like (replace with actual)
class MockPadresClient:
    async def run_affordance_test(self, config: Dict) -> Dict:
        print(f"Mock Padres: Simulating test with config: {config}")
        # Simulate some logic based on config
        touch_prob = 0.0
        if config.get('glow_intensity', 0) > 0.6: # More responsive to higher glow
            touch_prob += 0.4
        if config.get('pulse_frequency', 0) > 2.0 and config.get('pulse_frequency',0) < 3.5: # Sweet spot for pulse
            touch_prob += 0.3 * config.get('user_profile', {}).get('reaction_baseline', 0.5)
        if config.get('edge_width', 0) > 1.5: # Edge width in pixels
            touch_prob += 0.3
        
        # Interaction effect: high glow + high edge is good for older users
        if config.get('user_profile', {}).get('age', 30) > 50:
            if config.get('glow_intensity', 0) > 0.5 and config.get('edge_width', 0) > 2.5:
                touch_prob += 0.2
        
        # Simulate some noise/randomness
        touch_prob += random.uniform(-0.05, 0.05)
        return {'touch_probability': max(0.0, min(1.0, touch_prob))}

async def evaluate_cue_fitness_for_user_with_padres(cue: VisualCue, user: SyntheticUser, padres_client_instance) -> float:
    """Actually test in your VR environment via Padres API."""
    # Convert cue to Padres parameters - ADAPT THIS TO YOUR PADRES API
    padres_config = {
        'glow_intensity': cue.glow,
        'pulse_frequency': cue.pulse_hz,
        'edge_width': cue.edge * 5.0,  # Example: Scale 0-1 'edge' to 0-5 pixels
        # Add any other parameters your Padres API expects based on the VisualCue
        'user_profile': { # Example user profile structure for Padres
            'age': user.age,
            # Example: reaction_baseline, could be influenced by age, vr_experience etc.
            'reaction_baseline': 1.0 - (user.age / 150.0) 
        }
    }
    
    try:
        # Assuming padres_client_instance has an async method run_affordance_test
        result = await padres_client_instance.run_affordance_test(padres_config)
        return result.get('touch_probability', 0.0)
    except Exception as e:
        print(f"Error calling Padres API for user {user.user_id}, cue {cue}: {e}")
        return 0.0 # Return a low fitness score on error

def calculate_overall_fitness_with_padres(cue: VisualCue, users: List[SyntheticUser], padres_client_instance) -> float:
    """
    Calculates overall fitness by calling Padres API for each user.
    This is a synchronous wrapper for DEAP's synchronous evaluation function.
    """
    if not users:
        return 0.0
    
    # If your padres_client is async, you need to run the async calls.
    # One way to do this if the main DEAP loop is synchronous:
    async def _gather_fitness_results():
        tasks = [evaluate_cue_fitness_for_user_with_padres(cue, user, padres_client_instance) for user in users]
        individual_fitness_scores = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and sum valid scores
        valid_scores = [score for score in individual_fitness_scores if isinstance(score, (float, int))]
        if not valid_scores:
             print(f"Warning: All Padres API calls failed for cue {cue}")
             return 0.0
        return sum(valid_scores) / len(valid_scores) # Average of successful evaluations

    try:
        # Check if an event loop is already running.
        # FastAPI/Uvicorn runs its own loop. If this is called from there, we can't use asyncio.run() directly.
        # However, DEAP's evaluate is usually called in a separate process/thread pool by some distributed EA setups,
        # or synchronously in simpler setups.
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # This case is tricky. If DEAP is running in the same thread as FastAPI's event loop,
            # creating new tasks like this might work, but it's complex.
            # A better approach for FastAPI is to have the evolution run in a separate process
            # or use a thread pool executor for the blocking asyncio.run call.
            # For now, let's assume this function might be called where we can run a new loop.
            # This part might need adjustment based on how DEAP is invoked.
            # A simpler but potentially blocking approach if not in an async context:
            return asyncio.run(_gather_fitness_results())
        else:
            return asyncio.run(_gather_fitness_results())
    except RuntimeError as e: # Handles "cannot run event loop while another loop is running"
        # This is a common issue if DEAP is called directly from an async FastAPI endpoint.
        # The ideal solution is to run DEAP in a separate thread or process.
        # As a fallback for now, if we are already in a running loop (e.g. testing in Jupyter notebook with await)
        # This will likely NOT work correctly if called from a synchronous DEAP function inside an async framework.
        # You'd need to use something like `nest_asyncio` or run DEAP in a thread.
        print(f"RuntimeError with asyncio: {e}. Consider how DEAP calls this from an async context.")
        # For a quick test outside an async server, this might be sufficient:
        return asyncio.run(_gather_fitness_results())

# The deap_evaluate function in visual_cue_evolver.py will need to be modified
# to call calculate_overall_fitness_with_padres instead of calculate_overall_fitness,
# and it will need access to your padres_client_instance.
# Example (to be done in visual_cue_evolver.py):
#
# from .synthetic_users import calculate_overall_fitness_with_padres
# # ... in visual_cue_evolver.py ...
# # padres_client_instance_for_evolver = YourPadresClient() # Initialize here or pass in
#
# def deap_evaluate_padres(individual: List[float]) -> Tuple[float]:
#     if not USER_POPULATION_CACHE:
#         initialize_user_population_cache(100) # Default
#     cue = individual_to_visual_cue(individual)
#     # Ensure padres_client_instance_for_evolver is accessible
#     fitness_score = calculate_overall_fitness_with_padres(cue, USER_POPULATION_CACHE, padres_client_instance_for_evolver)
#     return (fitness_score,)
#
# # Then: toolbox.register("evaluate", deap_evaluate_padres)
#

# Helper for distributions
def sample_from_distribution(dist_data: dict):
    choices = list(dist_data.keys())
    probabilities = list(dist_data.values())
    return random.choices(choices, weights=probabilities, k=1)[0]

def sample_age_from_distribution(age_dist_list: list):
    age_ranges = [item[0:2] for item in age_dist_list]
    probabilities = [item[2] for item in age_dist_list]
    chosen_range_idx = random.choices(range(len(age_ranges)), weights=probabilities, k=1)[0]
    chosen_range = age_ranges[chosen_range_idx]
    return random.randint(chosen_range[0], chosen_range[1])

def get_age_group_for_map(age: int, age_map: dict):
    for age_range_tuple in age_map.keys():
        if age_range_tuple[0] <= age <= age_range_tuple[1]:
            return age_range_tuple
    # Fallback to the first defined age group if no exact match (should ideally cover all ages)
    return list(age_map.keys())[0]

def create_realistic_population(num_users: int) -> List[SyntheticUser]:
    """Create population based on (example) real gaming demographics"""
    population = []
    
    age_distribution_list = [
        (13, 17, 0.15), (18, 24, 0.30), (25, 34, 0.25),
        (35, 44, 0.15), (45, 54, 0.10), (55, 70, 0.05)
    ]
    
    vr_experience_by_age_group = {
        (13, 17): {'none': 0.3, 'novice': 0.5, 'intermediate': 0.15, 'expert': 0.05},
        (18, 24): {'none': 0.2, 'novice': 0.4, 'intermediate': 0.3, 'expert': 0.1},
        (25, 34): {'none': 0.15, 'novice': 0.35, 'intermediate': 0.35, 'expert': 0.15},
        (35, 44): {'none': 0.2, 'novice': 0.3, 'intermediate': 0.35, 'expert': 0.15},
        (45, 54): {'none': 0.3, 'novice': 0.4, 'intermediate': 0.2, 'expert': 0.1},
        (55, 70): {'none': 0.5, 'novice': 0.3, 'intermediate': 0.15, 'expert': 0.05},
    }
    cultural_regions_dist = { 
        'north_america': 0.25, 'europe': 0.25, 'east_asia': 0.20, 
        'south_asia': 0.10, 'latin_america': 0.07, 'africa': 0.05, 
        'oceania': 0.03, 'middle_east': 0.05
    }

    for i in range(num_users):
        age = sample_age_from_distribution(age_distribution_list)
        age_group_key = get_age_group_for_map(age, vr_experience_by_age_group)
        vr_exp = sample_from_distribution(vr_experience_by_age_group[age_group_key])
        
        gaming_hours = random.uniform(1, 10) # Base
        if vr_exp == 'novice': gaming_hours += random.uniform(0,5)
        elif vr_exp == 'intermediate': gaming_hours += random.uniform(5,15)
        elif vr_exp == 'expert': gaming_hours += random.uniform(10,30)
        if age < 18: gaming_hours *= 1.5
        elif age > 50: gaming_hours *= 0.6
        gaming_hours = max(0, min(50, gaming_hours))

        visual_acuity = max(0.5, min(1.5, random.gauss(1.0, 0.15)))
        
        base_reaction_multiplier = 1.0
        if age < 20: base_reaction_multiplier -= 0.15
        if age > 60: base_reaction_multiplier += 0.25
        if gaming_hours > 25: base_reaction_multiplier -= 0.1
        if vr_exp == 'expert': base_reaction_multiplier -= 0.1
        reaction_multiplier = max(0.6, min(1.8, random.gauss(base_reaction_multiplier, 0.1)))

        user = SyntheticUser(
            user_id=i,
            age=age,
            gaming_hours_per_week=round(gaming_hours,1),
            vr_experience_level=vr_exp,
            dominant_hand=random.choices(['right', 'left'], weights=[0.88, 0.12])[0],
            visual_acuity_factor=round(visual_acuity,2),
            reaction_time_multiplier=round(reaction_multiplier,2),
            cultural_region=sample_from_distribution(cultural_regions_dist)
        )
        population.append(user)
    return population

# For standalone testing of this file:
if __name__ == '__main__':
    print("Testing synthetic_users.py with enhanced fitness function...")
    realistic_pop = create_realistic_population(20) # Test with 20 users
    print("\nGenerated 20 realistic synthetic users (sample):")
    for i, u in enumerate(realistic_pop):
        if i < 5: print(u) # Print first 5
    
    print("\n--- Testing Fitness of Sample Cues ---")
    sample_cues = [
        VisualCue(glow=0.8, pulse_hz=2.5, edge=0.2, color_hue=220, animation_type=1, complexity_score=0.5), # Expected good for West, good freq
        VisualCue(glow=0.2, pulse_hz=4.5, edge=0.8, color_hue=10, animation_type=2, complexity_score=0.2),   # Expected good for East Asia (no red), good for low acuity
        VisualCue(glow=0.5, pulse_hz=1.0, edge=0.5, color_hue=150, particle_density=0.9, animation_type=4, complexity_score=0.8) # Complex, potentially good for experts
    ]

    for i, test_cue in enumerate(sample_cues):
        print(f"\nCue {i+1}: {test_cue}")
        print(f"  Complexity Score: {test_cue.complexity_score:.2f}")
        individual_fitness_scores = [evaluate_cue_fitness_for_user(test_cue, user) for user in realistic_pop]
        avg_fitness = np.mean(individual_fitness_scores)
        print(f"  Average fitness with placeholder model: {avg_fitness:.4f}")
        print(f"  Fitness scores distribution (sample): Min={min(individual_fitness_scores):.2f}, Max={max(individual_fitness_scores):.2f}, Std={np.std(individual_fitness_scores):.2f}")

    # Example of how overall fitness would be calculated (used by DEAP)
    # test_overall_fitness = calculate_overall_fitness(sample_cues[0], realistic_pop)
    # print(f"\nOverall fitness for Cue 1 (for DEAP): {test_overall_fitness:.4f}")
