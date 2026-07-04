import os
import random

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Atropos Trajectory Visualizer")

# Simple mock database of trajectories and simulated thinking steps for each env
SIM_DATA = {
    "epigenetic_neural_plasticity": {
        "think": "Evaluating weights with high gradient variance (1.5). According to Epigenetic Plasticity rule, high variance weight points require methylation to protect stable long-term memories from catastrophic forgetting. Ideal methylation range is near 0.8.",
        "assistant": "0.8",
    },
    "mycelnet_routing": {
        "think": "Analyzing connectivity matrices for data packets. Node 0 is target 5. Evaluating flux optimization under Physarum polycephalum equations. The highest conductivity path utilizes node 2 as next hop.",
        "assistant": "2",
    },
    "acoustic_protein_folding": {
        "think": "Decoding audio frequency profiles. The primary resonance peak is located exactly at 440Hz, indicating proper folding structures without denatured aggregation.",
        "assistant": "FOLDED",
    },
    "linguistic_paleontology": {
        "think": "Tracing usage frequencies of archaic term 'thee'. The usage gradient is highly negative, pointing towards linguistic extinction state. Estimating remaining survival rate.",
        "assistant": "0.01",
    },
    "cognitive_load_scheduler": {
        "think": "Comparing task queues. Task A (load 8) plus Task B (load 4) exceeds the cognitive limits of 10. We must reschedule to run Task B and Task C (load 1) first.",
        "assistant": "Schedule Task B, then Task C",
    },
    "origami_graph_algorithms": {
        "think": "Projecting node points (1.0, 1.0) on a 45 degree fold axis matrix. Applying Huzita-Hatori origami equations.",
        "assistant": "0.0",
    },
    "spiking_weather_net": {
        "think": "Calculating integrate-and-fire thresholds. Input currents [0.2, 0.4, 0.5, 0.8] accumulate to trigger action potential spikes at threshold 1.0.",
        "assistant": "0, 0, 1, 0",
    },
    "fermentation_distributed_compute": {
        "think": "Feeding substrate amount 8.0 units into yeast agent. Evaluating local load constraints and metabolic fatigue output.",
        "assistant": "8.0",
    },
    "temporal_knowledge_decay": {
        "think": "Evaluating relationship half-life timeline parameters. Computing survival exponential probability index.",
        "assistant": "0.707",
    },
    "swarm_drug_molecule_designer": {
        "think": "Optimizing ant colony pheromone paths in 3D binding pocket slots. Slot 3 contains highest pheromone deposit.",
        "assistant": "3",
    },
    "biolumiviz": {
        "think": "Mapping 3D coordinate vector distance to bioluminescent pulse rates.",
        "assistant": "2.38",
    },
    "synthetic_synesthesia_engine": {
        "think": "Translating RGB color values [255, 128, 64] to MIDI audio pitch scales.",
        "assistant": "63",
    },
    "dreamstate_cycle_predictor": {
        "think": "Analyzing motion sequence variance to evaluate sleep cycle REM phase dreaming likelihood.",
        "assistant": "0.005",
    },
    "gravitational_wave_pattern_ml": {
        "think": "Running template cross-correlation calculations on LIGO detector streams.",
        "assistant": "0.05",
    },
    "quantum_classical_hybrid_nas": {
        "think": "Navigating variational quantum circuit layout searching space to locate best configuration.",
        "assistant": "3_Quantum_2_Classical",
    },
}


@app.get("/", response_class=HTMLResponse)
def get_index():
    return HTMLResponse(
        content="""" + HTML_CONTENT.replace('"', '\"').replace('\n', '\\n') + """
    )


@app.get("/sim/{env_name}")
def simulate_trajectory(env_name: str, prompt: str = ""):
    data = SIM_DATA.get(
        env_name,
        {"think": "Processing custom simulated task details.", "assistant": "Verified"},
    )

    user_prompt = (
        prompt
        if prompt
        else f"Execute RL trajectory simulation loop for {env_name.replace('_', ' ')}."
    )

    # Construct simulated step trajectories
    steps = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "think": data["think"], "content": data["assistant"]},
    ]

    # Generate random analytics data for the dashboard chart
    rewards = [random.uniform(0.1, 0.9) for _ in range(5)] + [1.0]

    return {
        "env_name": env_name,
        "trajectory_steps": steps,
        "rewards": rewards,
        "final_reward": 1.0,
        "token_count": random.randint(120, 512),
    }
