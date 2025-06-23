import asyncio
import json
from typing import Callable, Dict, List

# Assuming these modules are in the 'evolution' package relative to this script's location
from evolution.synthetic_users import (  # For potential direct use or if evolver needs it
    SyntheticUser,
    VisualCue,
    calculate_overall_fitness,
    create_realistic_population,
)
from evolution.visual_cue_evolver import (
    USER_POPULATION_CACHE,
    evolve_visual_cues_multi_objective,
    initialize_user_population_cache,
)


async def run_single_segment_evolution_async(
    segment_name: str,
    segment_users: List[SyntheticUser],
    num_generations: int,
    population_size: int,
) -> Dict:
    """Runs evolution for a single segment and returns results."""
    print(
        f"--- Starting evolution for segment: {segment_name} ({len(segment_users)} users) ---"
    )

    # Note: evolve_visual_cues_multi_objective is synchronous.
    # asyncio.to_thread runs it in a separate thread to avoid blocking the event loop.
    try:
        logbook_data = []  # To store serializable logbook
        if hasattr(asyncio, "to_thread"):  # Python 3.9+
            pareto_front_cues, log_output = await asyncio.to_thread(
                evolve_visual_cues_multi_objective,  # This now takes users as first arg
                users_for_this_run=segment_users,
                num_generations=num_generations,
                population_size=population_size,
                # cxpb, mutpb use defaults from the evolver function signature
            )
            logbook_data = (
                log_output  # log_output is already serializable list of dicts
            )
        else:
            print(
                f"Warning: asyncio.to_thread not available. Running segment {segment_name} synchronously."
            )
            pareto_front_cues, log_output = evolve_visual_cues_multi_objective(
                users_for_this_run=segment_users,
                num_generations=num_generations,
                population_size=population_size,
            )
            logbook_data = log_output

        serialized_cues = []
        if pareto_front_cues:
            for cue in pareto_front_cues:
                serialized_cues.append(
                    cue.__dict__ if hasattr(cue, "__dict__") else str(cue)
                )

        return {
            "segment_name": segment_name,
            "num_users_in_segment": len(segment_users),
            "best_cues_pareto_front": serialized_cues,
            "logbook": logbook_data,  # Include the logbook
            "status": "completed",
        }
    except Exception as e:
        print(f"Error during evolution for segment {segment_name}: {e}")
        import traceback

        traceback.print_exc()
        return {
            "segment_name": segment_name,
            "status": "failed",
            "error": str(e),
            "logbook": [],  # Empty logbook on error
        }


async def main_segmented_runs():
    print("Creating full realistic user population...")
    # Create a base large population once
    full_user_population = create_realistic_population(
        num_users=200
    )  # Reduced for quicker test
    print(f"Full population size: {len(full_user_population)}")

    # Define user segments (filters)
    segments: Dict[str, Callable[[SyntheticUser], bool]] = {
        "young_gamers": lambda u: u.age < 25 and u.gaming_hours_per_week > 15,
        "seniors_low_va": lambda u: u.age > 55 and u.visual_acuity_factor < 0.9,
        "vr_experts_europe": lambda u: u.vr_experience_level == "expert"
        and u.cultural_region == "europe",
        "accessibility_focus_reaction": lambda u: u.reaction_time_multiplier > 1.3,
        "east_asia_novice": lambda u: u.cultural_region == "east_asia"
        and u.vr_experience_level == "novice",
    }

    all_segment_results = []
    tasks = []

    # Common evolution parameters
    num_generations_segment = 10  # Reduced for test
    population_size_segment = 20  # Reduced for test

    for segment_name, filter_func in segments.items():
        segment_users = [u for u in full_user_population if filter_func(u)]

        if not segment_users:
            print(
                "Skipping segment "{segment_name}' due to no users matching criteria."
            )
            all_segment_results.append(
                {
                    "segment_name": segment_name,
                    "status": "skipped_no_users",
                    "best_cues_pareto_front": [],
                }
            )
            continue

        # Create an async task for each segment
        # Note: If evolve_visual_cues_multi_objective directly uses the global USER_POPULATION_CACHE,
        # running these truly in parallel with asyncio.gather will lead to race conditions on that global.
        # The `await asyncio.to_thread` helps run the synchronous DEAP code in a separate thread,
        # mitigating direct asyncio event loop blocking, but the global USER_POPULATION_CACHE remains an issue
        # for true parallelism. For a robust solution, USER_POPULATION_CACHE should be passed to the function.
        # The current setup will effectively run them sequentially if `asyncio.to_thread` is not used,
        # or if it is used, the global USER_POPULATION_CACHE will be overwritten by each thread before its DEAP run.
        # For a true parallel test, `evolve_visual_cues_multi_objective` needs refactoring to not use globals.
        # Given the plan's progression, we simulate the orchestration here.
        # For now, let's run them sequentially to avoid global state issues with the current `USER_POPULATION_CACHE` approach.
        print(f"Preparing to run segment: {segment_name}")
        result = await run_single_segment_evolution_async(
            segment_name,
            segment_users,
            num_generations_segment,
            population_size_segment,
        )
        all_segment_results.append(result)

    print("\n--- All Segmented Evolutions Attempted ---")
    for res in all_segment_results:
        print(f"Segment: {res['segment_name']}, Status: {res['status']}")
        if res["status"] == "completed":
            print(f"  Found {len(res['best_cues_pareto_front'])} non-dominated cues.")
            # print(f"  Example best cue(s): {res['best_cues_pareto_front'][:2]}") # Print first few

    # Save results
    with open("segmented_evolution_results.json", "w") as f:
        json.dump(all_segment_results, f, indent=2)
    print("\nSegmented evolution results saved to segmented_evolution_results.json")


if __name__ == "__main__":
    asyncio.run(main_segmented_runs())
