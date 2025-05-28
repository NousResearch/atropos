#!/usr/bin/env python3
"""
Massive Scale VR Experiment Runner for AMIEN
Runs large-scale VR experiments with thousands of simulated users
"""

import asyncio
import json
import random
import numpy as np
from datetime import datetime
import concurrent.futures
from typing import List, Dict

async def run_vr_experiments(sample_size: int):
    """Run massive scale VR experiments"""
    
    print(f"🧪 Starting massive scale VR experiments with {sample_size} users...")
    
    # Generate diverse user demographics
    user_demographics = generate_user_demographics(sample_size)
    
    # Run experiments in parallel batches
    batch_size = min(1000, sample_size // 10)  # Process in batches
    results = []
    
    for i in range(0, sample_size, batch_size):
        batch_users = user_demographics[i:i + batch_size]
        print(f"🔬 Processing batch {i//batch_size + 1}/{(sample_size + batch_size - 1)//batch_size}")
        
        batch_results = await run_experiment_batch(batch_users)
        results.extend(batch_results)
    
    # Analyze results
    analysis = analyze_experiment_results(results)
    
    return {
        "experiment_results": results,
        "analysis": analysis,
        "sample_size": sample_size,
        "timestamp": datetime.utcnow().isoformat(),
        "type": "massive_scale_experiments"
    }

def generate_user_demographics(sample_size: int) -> List[Dict]:
    """Generate diverse user demographics for experiments"""
    
    demographics = []
    
    for user_id in range(sample_size):
        # Age distribution (realistic VR user base)
        age_weights = [0.05, 0.25, 0.35, 0.25, 0.10]  # 18-25, 26-35, 36-45, 46-55, 56+
        age_ranges = [(18, 25), (26, 35), (36, 45), (46, 55), (56, 70)]
        age_range = random.choices(age_ranges, weights=age_weights)[0]
        age = random.randint(*age_range)
        
        # VR experience level
        experience_levels = ["novice", "intermediate", "expert"]
        experience_weights = [0.4, 0.45, 0.15]
        experience = random.choices(experience_levels, weights=experience_weights)[0]
        
        # Geographic regions
        regions = ["North America", "Europe", "East Asia", "South America", "Other"]
        region_weights = [0.35, 0.30, 0.20, 0.10, 0.05]
        region = random.choices(regions, weights=region_weights)[0]
        
        # Visual acuity (affects VR performance)
        visual_acuity = random.normalvariate(1.0, 0.2)  # 20/20 = 1.0
        visual_acuity = max(0.3, min(2.0, visual_acuity))  # Clamp to realistic range
        
        # Gaming background
        gaming_hours_per_week = max(0, random.normalvariate(8, 6))
        
        # Motion sensitivity
        motion_sensitivity = random.uniform(0.1, 1.0)  # 0.1 = low, 1.0 = high
        
        demographics.append({
            "user_id": user_id,
            "age": age,
            "experience": experience,
            "region": region,
            "visual_acuity": visual_acuity,
            "gaming_hours_per_week": gaming_hours_per_week,
            "motion_sensitivity": motion_sensitivity
        })
    
    return demographics

async def run_experiment_batch(users: List[Dict]) -> List[Dict]:
    """Run VR experiments for a batch of users"""
    
    # Use thread pool for CPU-intensive simulation
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        loop = asyncio.get_event_loop()
        
        # Create tasks for each user
        tasks = []
        for user in users:
            task = loop.run_in_executor(executor, simulate_vr_experiment, user)
            tasks.append(task)
        
        # Wait for all experiments to complete
        results = await asyncio.gather(*tasks)
    
    return results

def simulate_vr_experiment(user: Dict) -> Dict:
    """Simulate VR experiment for a single user"""
    
    # Generate VR task scenarios
    scenarios = [
        "object_selection",
        "spatial_navigation", 
        "hand_tracking",
        "menu_interaction",
        "3d_manipulation"
    ]
    
    scenario_results = []
    
    for scenario in scenarios:
        # Simulate performance based on user characteristics
        base_performance = simulate_scenario_performance(user, scenario)
        
        # Add measurement noise
        noise = random.normalvariate(0, 0.05)
        measured_performance = max(0, min(1, base_performance + noise))
        
        # Simulate timing data
        completion_time = simulate_completion_time(user, scenario, measured_performance)
        
        # Simulate error rates
        error_rate = simulate_error_rate(user, scenario, measured_performance)
        
        scenario_results.append({
            "scenario": scenario,
            "performance_score": measured_performance,
            "completion_time_seconds": completion_time,
            "error_rate": error_rate,
            "comfort_rating": simulate_comfort_rating(user, scenario)
        })
    
    # Calculate overall metrics
    overall_performance = np.mean([s["performance_score"] for s in scenario_results])
    overall_completion_time = np.mean([s["completion_time_seconds"] for s in scenario_results])
    overall_error_rate = np.mean([s["error_rate"] for s in scenario_results])
    overall_comfort = np.mean([s["comfort_rating"] for s in scenario_results])
    
    return {
        "user_id": user["user_id"],
        "user_demographics": user,
        "scenario_results": scenario_results,
        "overall_metrics": {
            "performance_score": overall_performance,
            "completion_time_seconds": overall_completion_time,
            "error_rate": overall_error_rate,
            "comfort_rating": overall_comfort
        },
        "experiment_timestamp": datetime.utcnow().isoformat()
    }

def simulate_scenario_performance(user: Dict, scenario: str) -> float:
    """Simulate user performance for a specific VR scenario"""
    
    base_performance = 0.6  # Base performance level
    
    # Age effects (younger users typically perform better)
    age_factor = max(0.5, 1.2 - (user["age"] - 25) * 0.01)
    
    # Experience effects
    experience_multipliers = {"novice": 0.8, "intermediate": 1.0, "expert": 1.3}
    experience_factor = experience_multipliers[user["experience"]]
    
    # Visual acuity effects
    visual_factor = min(1.2, user["visual_acuity"])
    
    # Gaming background effects
    gaming_factor = 1.0 + min(0.3, user["gaming_hours_per_week"] * 0.02)
    
    # Scenario-specific modifiers
    scenario_modifiers = {
        "object_selection": 1.0,
        "spatial_navigation": 0.9,  # Slightly harder
        "hand_tracking": 0.8,       # More challenging
        "menu_interaction": 1.1,    # Easier
        "3d_manipulation": 0.7      # Most challenging
    }
    
    scenario_factor = scenario_modifiers.get(scenario, 1.0)
    
    # Motion sensitivity effects (affects comfort and performance)
    motion_factor = 1.1 - user["motion_sensitivity"] * 0.2
    
    # Calculate final performance
    performance = (base_performance * age_factor * experience_factor * 
                  visual_factor * gaming_factor * scenario_factor * motion_factor)
    
    return max(0.1, min(1.0, performance))

def simulate_completion_time(user: Dict, scenario: str, performance: float) -> float:
    """Simulate task completion time"""
    
    # Base completion times (seconds)
    base_times = {
        "object_selection": 15.0,
        "spatial_navigation": 45.0,
        "hand_tracking": 30.0,
        "menu_interaction": 20.0,
        "3d_manipulation": 60.0
    }
    
    base_time = base_times.get(scenario, 30.0)
    
    # Performance inversely affects completion time
    time_factor = 2.0 - performance  # Higher performance = faster completion
    
    # Add individual variation
    variation = random.normalvariate(1.0, 0.2)
    
    completion_time = base_time * time_factor * variation
    
    return max(5.0, completion_time)  # Minimum 5 seconds

def simulate_error_rate(user: Dict, scenario: str, performance: float) -> float:
    """Simulate error rate for the task"""
    
    # Base error rates
    base_error_rates = {
        "object_selection": 0.15,
        "spatial_navigation": 0.20,
        "hand_tracking": 0.25,
        "menu_interaction": 0.10,
        "3d_manipulation": 0.30
    }
    
    base_error = base_error_rates.get(scenario, 0.20)
    
    # Performance inversely affects error rate
    error_factor = 1.5 - performance
    
    # Experience affects error rate
    experience_modifiers = {"novice": 1.3, "intermediate": 1.0, "expert": 0.7}
    experience_factor = experience_modifiers[user["experience"]]
    
    error_rate = base_error * error_factor * experience_factor
    
    return max(0.0, min(1.0, error_rate))

def simulate_comfort_rating(user: Dict, scenario: str) -> float:
    """Simulate comfort rating (1-10 scale)"""
    
    base_comfort = 7.0
    
    # Motion sensitivity affects comfort
    motion_penalty = user["motion_sensitivity"] * 2.0
    
    # Age affects comfort (older users may have more issues)
    age_penalty = max(0, (user["age"] - 40) * 0.05)
    
    # Experience improves comfort
    experience_bonuses = {"novice": 0, "intermediate": 0.5, "expert": 1.0}
    experience_bonus = experience_bonuses[user["experience"]]
    
    # Scenario-specific comfort
    scenario_modifiers = {
        "object_selection": 0.5,
        "spatial_navigation": -0.5,  # Can cause motion sickness
        "hand_tracking": 0.0,
        "menu_interaction": 1.0,     # Most comfortable
        "3d_manipulation": -0.3      # Slightly uncomfortable
    }
    
    scenario_modifier = scenario_modifiers.get(scenario, 0.0)
    
    comfort = (base_comfort - motion_penalty - age_penalty + 
              experience_bonus + scenario_modifier)
    
    # Add some random variation
    comfort += random.normalvariate(0, 0.5)
    
    return max(1.0, min(10.0, comfort))

def analyze_experiment_results(results: List[Dict]) -> Dict:
    """Analyze the massive scale experiment results"""
    
    print("📊 Analyzing experiment results...")
    
    # Extract overall metrics
    performance_scores = [r["overall_metrics"]["performance_score"] for r in results]
    completion_times = [r["overall_metrics"]["completion_time_seconds"] for r in results]
    error_rates = [r["overall_metrics"]["error_rate"] for r in results]
    comfort_ratings = [r["overall_metrics"]["comfort_rating"] for r in results]
    
    # Demographic analysis
    demographic_analysis = analyze_demographics(results)
    
    # Scenario analysis
    scenario_analysis = analyze_scenarios(results)
    
    # Performance correlations
    correlations = analyze_correlations(results)
    
    return {
        "summary_statistics": {
            "performance_score": {
                "mean": np.mean(performance_scores),
                "std": np.std(performance_scores),
                "min": np.min(performance_scores),
                "max": np.max(performance_scores),
                "median": np.median(performance_scores)
            },
            "completion_time": {
                "mean": np.mean(completion_times),
                "std": np.std(completion_times),
                "min": np.min(completion_times),
                "max": np.max(completion_times),
                "median": np.median(completion_times)
            },
            "error_rate": {
                "mean": np.mean(error_rates),
                "std": np.std(error_rates),
                "min": np.min(error_rates),
                "max": np.max(error_rates),
                "median": np.median(error_rates)
            },
            "comfort_rating": {
                "mean": np.mean(comfort_ratings),
                "std": np.std(comfort_ratings),
                "min": np.min(comfort_ratings),
                "max": np.max(comfort_ratings),
                "median": np.median(comfort_ratings)
            }
        },
        "demographic_analysis": demographic_analysis,
        "scenario_analysis": scenario_analysis,
        "correlations": correlations,
        "sample_size": len(results)
    }

def analyze_demographics(results: List[Dict]) -> Dict:
    """Analyze performance by demographic groups"""
    
    # Group by age ranges
    age_groups = {"18-25": [], "26-35": [], "36-45": [], "46-55": [], "56+": []}
    
    for result in results:
        age = result["user_demographics"]["age"]
        if age <= 25:
            age_groups["18-25"].append(result["overall_metrics"]["performance_score"])
        elif age <= 35:
            age_groups["26-35"].append(result["overall_metrics"]["performance_score"])
        elif age <= 45:
            age_groups["36-45"].append(result["overall_metrics"]["performance_score"])
        elif age <= 55:
            age_groups["46-55"].append(result["overall_metrics"]["performance_score"])
        else:
            age_groups["56+"].append(result["overall_metrics"]["performance_score"])
    
    age_analysis = {}
    for group, scores in age_groups.items():
        if scores:
            age_analysis[group] = {
                "mean_performance": np.mean(scores),
                "count": len(scores)
            }
    
    # Group by experience level
    experience_groups = {"novice": [], "intermediate": [], "expert": []}
    
    for result in results:
        experience = result["user_demographics"]["experience"]
        experience_groups[experience].append(result["overall_metrics"]["performance_score"])
    
    experience_analysis = {}
    for group, scores in experience_groups.items():
        if scores:
            experience_analysis[group] = {
                "mean_performance": np.mean(scores),
                "count": len(scores)
            }
    
    return {
        "age_groups": age_analysis,
        "experience_levels": experience_analysis
    }

def analyze_scenarios(results: List[Dict]) -> Dict:
    """Analyze performance by VR scenario"""
    
    scenarios = ["object_selection", "spatial_navigation", "hand_tracking", "menu_interaction", "3d_manipulation"]
    scenario_analysis = {}
    
    for scenario in scenarios:
        scenario_scores = []
        scenario_times = []
        scenario_errors = []
        scenario_comfort = []
        
        for result in results:
            for scenario_result in result["scenario_results"]:
                if scenario_result["scenario"] == scenario:
                    scenario_scores.append(scenario_result["performance_score"])
                    scenario_times.append(scenario_result["completion_time_seconds"])
                    scenario_errors.append(scenario_result["error_rate"])
                    scenario_comfort.append(scenario_result["comfort_rating"])
        
        if scenario_scores:
            scenario_analysis[scenario] = {
                "mean_performance": np.mean(scenario_scores),
                "mean_completion_time": np.mean(scenario_times),
                "mean_error_rate": np.mean(scenario_errors),
                "mean_comfort": np.mean(scenario_comfort),
                "sample_size": len(scenario_scores)
            }
    
    return scenario_analysis

def analyze_correlations(results: List[Dict]) -> Dict:
    """Analyze correlations between user characteristics and performance"""
    
    ages = []
    visual_acuities = []
    gaming_hours = []
    motion_sensitivities = []
    performances = []
    
    for result in results:
        demo = result["user_demographics"]
        ages.append(demo["age"])
        visual_acuities.append(demo["visual_acuity"])
        gaming_hours.append(demo["gaming_hours_per_week"])
        motion_sensitivities.append(demo["motion_sensitivity"])
        performances.append(result["overall_metrics"]["performance_score"])
    
    # Calculate correlations
    age_corr = np.corrcoef(ages, performances)[0, 1]
    visual_corr = np.corrcoef(visual_acuities, performances)[0, 1]
    gaming_corr = np.corrcoef(gaming_hours, performances)[0, 1]
    motion_corr = np.corrcoef(motion_sensitivities, performances)[0, 1]
    
    return {
        "age_performance_correlation": age_corr,
        "visual_acuity_performance_correlation": visual_corr,
        "gaming_hours_performance_correlation": gaming_corr,
        "motion_sensitivity_performance_correlation": motion_corr
    } 