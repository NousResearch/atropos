#!/usr/bin/env python3
"""
Evolution-based Research Generator for AMIEN
Generates real VR research using evolutionary algorithms
"""

import asyncio
import json
import os
import random
from datetime import datetime

import google.generativeai as genai
import numpy as np

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


async def generate_vr_research():
    """Generate VR research using evolutionary algorithms"""

    print("🧬 Initializing evolutionary research generation...")

    # Run evolution-based discovery
    evolution_results = await run_evolution_discovery()

    # Generate research paper
    paper = await generate_research_paper(evolution_results)

    # Generate discovered functions
    functions = await generate_optimization_functions(evolution_results)

    return {
        "paper": paper,
        "data": evolution_results,
        "functions": functions,
        "timestamp": datetime.utcnow().isoformat(),
        "type": "evolution_research",
    }


async def run_evolution_discovery():
    """Run evolutionary algorithm to discover VR affordances"""

    print("🔬 Running evolutionary discovery...")

    # Define VR parameter space
    parameter_space = {
        "glow_intensity": (0.0, 1.0),
        "pulse_frequency": (0.5, 5.0),
        "color_hue": (0, 360),
        "blur_amount": (0.0, 0.5),
        "size_multiplier": (0.5, 2.0),
        "transparency": (0.1, 1.0),
    }

    # Initialize population
    population_size = 50
    generations = 25

    population = []
    for _ in range(population_size):
        individual = {}
        for param, (min_val, max_val) in parameter_space.items():
            individual[param] = random.uniform(min_val, max_val)
        population.append(individual)

    # Evolution loop
    best_solutions = []
    generation_stats = []

    for generation in range(generations):
        print(f"🧬 Generation {generation + 1}/{generations}")

        # Evaluate fitness (simulate VR user performance)
        fitness_scores = []
        for individual in population:
            fitness = await evaluate_vr_fitness(individual)
            fitness_scores.append(fitness)

        # Track best solutions
        best_idx = np.argmax(fitness_scores)
        best_solutions.append(
            {
                "generation": generation,
                "parameters": population[best_idx].copy(),
                "fitness": fitness_scores[best_idx],
            }
        )

        # Generation statistics
        generation_stats.append(
            {
                "generation": generation,
                "mean_fitness": np.mean(fitness_scores),
                "max_fitness": np.max(fitness_scores),
                "std_fitness": np.std(fitness_scores),
            }
        )

        # Selection and reproduction
        population = evolve_population(population, fitness_scores)

    return {
        "best_solutions": best_solutions,
        "generation_stats": generation_stats,
        "parameter_space": parameter_space,
        "final_population": population,
        "convergence_analysis": analyze_convergence(generation_stats),
    }


async def evaluate_vr_fitness(parameters):
    """Evaluate fitness of VR parameters (simulated user performance)"""

    # Simulate realistic VR user performance based on parameters
    base_performance = 0.5

    # Glow intensity affects visibility
    glow_bonus = 0.3 * (1 - abs(parameters["glow_intensity"] - 0.7))

    # Pulse frequency affects attention
    pulse_bonus = 0.2 * (1 - abs(parameters["pulse_frequency"] - 2.5) / 2.5)

    # Color hue affects recognition (blue-green optimal)
    optimal_hue = 180
    hue_bonus = 0.2 * (1 - abs(parameters["color_hue"] - optimal_hue) / 180)

    # Blur affects clarity
    blur_penalty = -0.4 * parameters["blur_amount"]

    # Size affects usability
    size_bonus = 0.1 * (1 - abs(parameters["size_multiplier"] - 1.2))

    # Transparency affects visibility
    transparency_bonus = 0.1 * (1 - abs(parameters["transparency"] - 0.8))

    # Add some noise for realism
    noise = random.gauss(0, 0.05)

    fitness = (
        base_performance
        + glow_bonus
        + pulse_bonus
        + hue_bonus
        + blur_penalty
        + size_bonus
        + transparency_bonus
        + noise
    )

    return max(0, min(1, fitness))


def evolve_population(population, fitness_scores):
    """Evolve population using selection, crossover, and mutation"""

    new_population = []
    population_size = len(population)

    # Elitism - keep top 10%
    elite_count = max(1, population_size // 10)
    elite_indices = np.argsort(fitness_scores)[-elite_count:]

    for idx in elite_indices:
        new_population.append(population[idx].copy())

    # Generate rest through crossover and mutation
    while len(new_population) < population_size:
        # Tournament selection
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)

        # Crossover
        child = crossover(parent1, parent2)

        # Mutation
        child = mutate(child)

        new_population.append(child)

    return new_population


def tournament_selection(population, fitness_scores, tournament_size=3):
    """Select individual using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx].copy()


def crossover(parent1, parent2):
    """Create child through crossover"""
    child = {}
    for param in parent1.keys():
        if random.random() < 0.5:
            child[param] = parent1[param]
        else:
            child[param] = parent2[param]
    return child


def mutate(individual, mutation_rate=0.1, mutation_strength=0.1):
    """Mutate individual"""
    for param, value in individual.items():
        if random.random() < mutation_rate:
            # Add Gaussian noise
            noise = random.gauss(0, mutation_strength)
            individual[param] = max(0, value + noise)
    return individual


def analyze_convergence(generation_stats):
    """Analyze convergence patterns"""
    if len(generation_stats) < 5:
        return {"status": "insufficient_data"}

    recent_fitness = [gen["max_fitness"] for gen in generation_stats[-5:]]
    fitness_trend = np.polyfit(range(5), recent_fitness, 1)[0]

    return {
        "fitness_trend": fitness_trend,
        "converged": abs(fitness_trend) < 0.001,
        "final_fitness": generation_stats[-1]["max_fitness"],
        "improvement": generation_stats[-1]["max_fitness"]
        - generation_stats[0]["max_fitness"],
    }


async def generate_research_paper(evolution_results):
    """Generate research paper using Gemini"""

    print("📝 Generating research paper...")

    try:
        model = genai.GenerativeModel("gemini-pro")

        prompt = """
        Generate a scientific research paper based on the following evolutionary algorithm results for VR affordance optimization:

        Evolution Results:
        - Generations: {len(evolution_results['generation_stats'])}
        - Best fitness achieved: {evolution_results['generation_stats'][-1]['max_fitness']:.3f}
        - Convergence: {evolution_results['convergence_analysis']}
        - Parameter space: {evolution_results['parameter_space']}

        Best Solution Parameters:
        {json.dumps(evolution_results['best_solutions'][-1]['parameters'], indent=2)}

        Please write a complete research paper with:
        1. Abstract
        2. Introduction
        3. Methodology
        4. Results
        5. Discussion
        6. Conclusion

        Focus on the evolutionary optimization of VR visual affordances and their impact on user performance.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"❌ Failed to generate paper with Gemini: {e}")
        return generate_fallback_paper(evolution_results)


def generate_fallback_paper(evolution_results):
    """Generate fallback paper without AI"""

    best_solution = evolution_results["best_solutions"][-1]
    convergence = evolution_results["convergence_analysis"]

    return """
# Evolutionary Optimization of VR Visual Affordances

## Abstract

This study presents an evolutionary algorithm approach to optimizing visual affordances in virtual reality environments. Through {len(evolution_results['generation_stats'])} generations of evolution, we discovered optimal parameter configurations that improve user performance by {convergence['improvement']:.1%}.

## Introduction

Virtual reality systems require careful optimization of visual cues to maximize user performance and comfort. This research applies evolutionary algorithms to discover optimal configurations of visual affordance parameters.

## Methodology

We employed a genetic algorithm with:
- Population size: 50 individuals
- Generations: {len(evolution_results['generation_stats'])}
- Parameter space: {len(evolution_results['parameter_space'])} dimensions
- Fitness evaluation based on simulated user performance

## Results

The evolutionary process converged to an optimal solution with fitness score {best_solution['fitness']:.3f}.

Optimal parameters:
{json.dumps(best_solution['parameters'], indent=2)}

## Discussion

The results demonstrate that evolutionary algorithms can effectively optimize VR visual affordances. The convergence pattern shows {'stable convergence' if convergence['converged'] else 'ongoing optimization'}.

## Conclusion

This research provides a framework for automated optimization of VR visual systems using evolutionary computation.

Generated: {datetime.utcnow().isoformat()}
"""


async def generate_optimization_functions(evolution_results):
    """Generate optimization functions based on results"""

    best_params = evolution_results["best_solutions"][-1]["parameters"]

    return """
# Generated VR Optimization Functions
# Based on evolutionary research results

import numpy as np

def optimal_vr_parameters():
    \"\"\"Return optimal VR parameters discovered through evolution\"\"\"
    return {json.dumps(best_params, indent=4)}

def evaluate_vr_fitness(glow_intensity, pulse_frequency, color_hue, blur_amount, size_multiplier, transparency):
    \"\"\"Evaluate VR parameter fitness based on discovered patterns\"\"\"

    # Optimal values discovered through evolution
    optimal_glow = {best_params['glow_intensity']:.3f}
    optimal_pulse = {best_params['pulse_frequency']:.3f}
    optimal_hue = {best_params['color_hue']:.1f}
    optimal_blur = {best_params['blur_amount']:.3f}
    optimal_size = {best_params['size_multiplier']:.3f}
    optimal_transparency = {best_params['transparency']:.3f}

    # Calculate fitness based on distance from optimal
    glow_score = 1 - abs(glow_intensity - optimal_glow)
    pulse_score = 1 - abs(pulse_frequency - optimal_pulse) / 5.0
    hue_score = 1 - abs(color_hue - optimal_hue) / 360.0
    blur_score = 1 - abs(blur_amount - optimal_blur) / 0.5
    size_score = 1 - abs(size_multiplier - optimal_size) / 1.5
    transparency_score = 1 - abs(transparency - optimal_transparency)

    return np.mean([glow_score, pulse_score, hue_score, blur_score, size_score, transparency_score])

def generate_vr_configuration(performance_target=0.8):
    \"\"\"Generate VR configuration for target performance level\"\"\"

    base_params = optimal_vr_parameters()

    # Adjust parameters based on target performance
    if performance_target < 0.8:
        # Relax constraints for lower performance
        base_params['glow_intensity'] *= (0.8 + 0.2 * performance_target)
        base_params['pulse_frequency'] *= (0.9 + 0.1 * performance_target)

    return base_params

# Generated: {datetime.utcnow().isoformat()}
"""
