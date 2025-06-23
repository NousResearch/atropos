# test_single_generation.py (was test_evolution_concept.py in plan)
import random
from dataclasses import dataclass, field
from typing import List


# Assuming VisualCue is defined in synthetic_users.py or locally for test
# For this standalone test, let's redefine it simply.
@dataclass
class VisualCueTest:  # Renamed to avoid conflict if importing from synthetic_users
    glow: float = field(default_factory=lambda: random.random())  # 0-1
    pulse_hz: float = field(default_factory=lambda: random.uniform(0.5, 5.0))  # 0.5-5.0
    edge: float = field(default_factory=lambda: random.random())  # 0-1
    fitness: float = 0.0  # To store fitness


def random_cue_test():
    return VisualCueTest()


def fitness_test(cue: VisualCueTest, users: List[dict]) -> float:
    touches = 0
    if not users:
        return 0.0
    for user in users:
        # Simple model: young users like fast pulses
        if user["age"] < 25:
            if cue.pulse_hz > 3.0:
                touches += 1
        # Older users like edge highlights
        elif user["age"] > 50:
            if cue.edge > 0.7:
                touches += 1
    return touches / len(users) if users else 0.0


def run_test():
    print("Running conceptual test of evolution (10 generations)...")
    # Create population
    population: List[VisualCueTest] = [random_cue_test() for _ in range(20)]
    users = [{"age": random.randint(15, 70)} for _ in range(100)]

    # Evolve for 10 generations
    for gen in range(10):
        # Evaluate fitness
        for cue_obj in population:  # Renamed cue to cue_obj to avoid conflict
            cue_obj.fitness = fitness_test(cue_obj, users)

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        if not population:
            print(f"Gen {gen}: Population became empty.")
            break

        print(f"Gen {gen}: Best fitness = {population[0].fitness:.3f}")
        print(
            f"  Best cue: glow={population[0].glow:.2f}, "
            f"pulse={population[0].pulse_hz:.2f}Hz, edge={population[0].edge:.2f}"
        )

        # Simplified evolution: Keep top 50%, replace rest with mutations of top
        if len(population) > 1:
            top_half_count = len(population) // 2
            if top_half_count == 0 and len(population) > 0:
                top_half_count = 1

            parents = population[:top_half_count]
            offspring_count = len(population) - top_half_count
            new_population = parents[:]

            for _ in range(offspring_count):
                if not parents:
                    break
                parent = random.choice(parents)
                mutated_cue = VisualCueTest(
                    glow=max(0, min(1, parent.glow + random.uniform(-0.1, 0.1))),
                    pulse_hz=max(
                        0.5, min(5.0, parent.pulse_hz + random.uniform(-0.5, 0.5))
                    ),
                    edge=max(0, min(1, parent.edge + random.uniform(-0.1, 0.1))),
                )
                new_population.append(mutated_cue)
            population = new_population[: len(population)]
        elif not population:
            print(f"Gen {gen}: Population empty, cannot evolve.")
            break

    print("\nEvolution concept test finished.")
    if population:
        print(f"Final best cue: {population[0]}")
    else:
        print("Final population was empty.")


if __name__ == "__main__":
    run_test()
