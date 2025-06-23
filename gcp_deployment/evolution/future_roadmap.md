# evolution/future_roadmap.md

# VR Affordance Discovery v2.0 - Atropos Integration Plan

## Why Add Atropos in v2.0?

The current v1.0 system focuses on discovering **WHAT** visual cues are effective for different user segments by evolving parameters and evaluating their impact on a modeled 'touchability' or interaction probability. This is a crucial first step and can yield significant insights into optimal cue design.

Integrating a system like "Atropos" (representing deeper behavioral analysis, potentially through RL agents and trajectory mining) in v2.0 would allow us to understand **HOW** users decide to interact and **WHY** certain cues are more effective from a behavioral sequence perspective. This moves beyond static cue evaluation to dynamic interaction analysis.

**Key benefits of Atropos integration:**
-   **Micro-behavioral Analysis:** Instead of just a binary touch/no-touch or a probability, we can analyze detailed interaction trajectories: hesitation times, approach paths, gaze patterns (if simulated/tracked), correction behaviors, etc.
-   **Causal Understanding of Behavior:** Move closer to understanding the cognitive processes triggered by affordances.
-   **Adaptive Cues:** Lay groundwork for cues that adapt in real-time to user behavior or inferred user state.
-   **Richer Synthetic Users:** RL agents trained within Atropos can become more nuanced synthetic users for future cue evolution.

## Integration Points with Existing v1.0 System

1.  **Fitness Evaluation Replacement/Augmentation:**
    *   **Current (v1.0):** `evaluate_cue_fitness_for_user` (placeholder or simple model) or `evaluate_cue_fitness_for_user_with_padres` (external simulation call returning a score).
    *   **v2.0 with Atropos:** The fitness of a `VisualCue` would be determined by deploying it in a simulated environment where an RL agent (representing a synthetic user) attempts to complete a task. Fitness could be a combination of task success, efficiency, observed 'naturalness' of interaction, or specific behavioral markers identified by Atropos.

2.  **Synthetic User Evolution:**
    *   **Current (v1.0):** `SyntheticUser` dataclass with predefined (though diverse) characteristics.
    *   **v2.0 with Atropos:** Synthetic users could be RL agents themselves, with policies (neural networks) whose parameters could also be co-evolved or pre-trained based on different personas. Their 'decision-making process' becomes part of what's modeled.

3.  **Data Collection:**
    *   **Current (v1.0):** Primarily collects the resulting fitness score for a cue-user pair.
    *   **v2.0 with Atropos:** Would collect rich trajectory data: (state, action, reward, next_state) sequences for each interaction, detailed timings, simulated gaze data, etc. This data would be stored, potentially in GCS, for later mining.

4.  **Pattern Mining & Insight Generation:**
    *   **Current (v1.0):** `AffordancePatternMiner` looks for convergence in cue parameters. `InsightGenerator` forms heuristics from these.
    *   **v2.0 with Atropos:** Pattern mining would extend to sequential pattern mining on trajectories (e.g., common sequences of actions leading to touch, hesitation patterns before interacting with certain cues). Insights would include behavioral explanations.

## Architecture Changes for v2.0

*   **SyntheticUser Module (`evolution/synthetic_users.py`):**
    *   May need to define an `RLAgentUser` class inheriting from or replacing `SyntheticUser`.
    *   This class would encapsulate an RL policy (e.g., a neural network loaded from a checkpoint) and the logic to interact with the Atropos-enhanced simulation environment.
*   **Fitness Evaluation (`evolution/visual_cue_evolver.py` & `evolution/synthetic_users.py`):**
    *   The core `evaluate_multi_objective` (or single-objective equivalent) would need to orchestrate the interaction of an `RLAgentUser` with the simulated environment presenting the `VisualCue`.
    *   The environment itself would become an MDP (Markov Decision Process) where the RL agent operates. The reward function for the RL agent *within* this MDP is distinct from the evolutionary fitness of the cue (though related).
*   **Simulation Environment (Padres API or similar):**
    *   Would need to be enhanced to support RL agent interaction (observe state, take action, return reward/next_state).
    *   Must be capable of logging detailed trajectory data.
*   **Data Storage (`ResearchDataManager` in core pipeline):**
    *   Needs schema and storage solutions for large-scale trajectory data (e.g., JSONL for trajectories, Parquet for efficiency).
*   **New Analysis Modules:**
    *   `evolution/trajectory_miner.py`: For sequence mining, identifying common behavioral patterns.
    *   `evolution/rl_agent_manager.py`: For training, managing, and deploying different RL agent personas.

## High-Level Workflow for v2.0 Cycle

1.  **Evolutionary Algorithm (DEAP - outer loop):**
    *   Generates a population of `VisualCue` parameters.
2.  **For each Cue in population:**
    *   **For each `RLAgentUser` persona in synthetic population:**
        *   Initialize simulated VR environment with the current `VisualCue` applied to a target object.
        *   RL agent attempts a relevant task (e.g., "interact with the highlighted object if it seems appropriate").
        *   Atropos logs the full interaction trajectory (states, actions, internal agent states if available).
        *   A fitness score for the *cue* is derived from the RL agent's performance and observed behaviors (e.g., task completion, low hesitation, directness of approach).
    *   Aggregate fitness scores across personas for the current `VisualCue`.
3.  **DEAP uses aggregated fitness** to select parents, perform crossover/mutation, creating the next generation of cues.
4.  **Periodically (or post-run):**
    *   `TrajectoryMiner` analyzes stored trajectories for behavioral patterns associated with high/low-performing cues.
    *   `InsightGenerator` uses both cue parameter convergences and trajectory patterns to explain *why* certain cues are effective.
    *   `PaperGenerator` drafts sections incorporating these deeper behavioral insights.

## Timeline (Post v1.0 Launch)

*   **v1.0 Launch & Initial Validation (Current Plan - Next ~2 Weeks):**
    *   Complete current 14-day plan.
    *   Deploy v1.0 system.
    *   Run initial large-scale discovery cycles.
    *   Generate first paper based on v1.0 findings (parameter-based discoveries).
    *   Release initial Unity package.
*   **Month 1 (Post v1.0): Validate & Iterate v1.0**
    *   Gather feedback on the Unity package from developers.
    *   Conduct small-scale user studies (if possible) to validate a few key discoveries from v1.0 against real human behavior.
    *   Refine v1.0 fitness functions (Padres API integration or internal model) based on this initial validation.
    *   Submit v1.0 paper to a relevant workshop/conference (e.g., CHI/SIGGRAPH LBW/Poster, IEEE VR poster).
*   **Month 2-3: Plan & Develop Atropos Core (v2.0)**
    *   **Design RL Agent Personas:** Define 2-3 distinct RL agent profiles (e.g., cautious explorer, impatient gamer, accessibility-focused user).
    *   **Develop/Adapt RL Environment:** Enhance your VR simulation (Padres or other) to serve as an MDP for these agents. Define state/action spaces, and initial reward functions for *agent training* (distinct from cue fitness).
    *   **Train Initial RL Agents:** Train baseline policies for each persona. The discoveries from v1.0 (e.g., generally effective cues) can be used to create environments that help bootstrap agent training.
    *   **Trajectory Logging:** Implement robust logging of (state, action, reward) trajectories from agent interactions.
*   **Month 4-5: Integrate Atropos with Evolutionary Loop & Initial v2.0 Runs**
    *   Replace/augment `evaluate_cue_fitness_for_user` to use RL agent performance and trajectory features as the fitness signal for cues.
    *   Run initial v2.0 discovery cycles.
    *   Develop `TrajectoryMiner` to extract initial behavioral patterns (e.g., hesitation times, path efficiency related to cues).
*   **Month 6: v2.0 Analysis, Paper, and Release**
    *   Analyze v2.0 results, focusing on how different cues influence agent *behavior* and decision-making paths.
    *   Generate insights that link cue parameters to specific behavioral sequences.
    *   Draft v2.0 paper with these deeper, trajectory-based insights.
    *   Update Unity package and design guidelines with v2.0 findings.

This phased approach allows for continuous delivery of value and learning, with v1.0 providing a strong foundation and initial discoveries that inform the more complex v2.0 integration.
