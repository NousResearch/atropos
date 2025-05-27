import asyncio
import json
import os
import pandas as pd

# Assuming modules are in the 'evolution' package
from evolution.run_segmented_evolutions import main_segmented_runs # This itself is async
from evolution.pattern_miner import AffordancePatternMiner
from evolution.analysis_tools import discover_interaction_effects
from evolution.insight_generator import InsightGenerator
from evolution.artifact_generator import generate_unity_affordance_library
from evolution.paper_utils import (
    compile_research_data_for_paper, 
    EvolutionaryPaperGeneratorWrapper, 
    MockLLMResearcher, 
    MockAutomatedPaperGenerator,
    load_json_results
)

# Define file paths (consistent with paper_utils and other scripts)
SEGMENTED_RESULTS_FILE = "segmented_evolution_results.json"
CONVERGENCE_PATTERNS_FILE = "convergence_patterns.json" 
INTERACTION_EFFECTS_FILE = "interaction_effects.json"
UNITY_CODE_FILE = "DiscoveredAffordances_IntegrationTest.cs"
DESIGN_GUIDELINES_FILE = "design_guidelines_IntegrationTest.md"
COMPILED_DATA_FILE = "compiled_research_data_IntegrationTest.json"
PAPER_DRAFT_FILE = "paper_draft_IntegrationTest.json"

async def run_complete_discovery_cycle():
    """End-to-end test of the entire system, using file-based data passing."""
    
    print("🧬 Starting VR Affordance Discovery Integration Test")
    final_status = True
    all_results_summary = {}

    # --- 1. Run segmented evolutions --- 
    print("\n1️⃣ Running segmented evolutions (simulated - this would take time)..."    )
    # In a real test, you might run with very small gen/pop sizes or use pre-existing mock file.
    # For now, let's assume main_segmented_runs generates SEGMENTED_RESULTS_FILE
    try:
        # To make this test runnable without hours of evolution, we'll create mock results if the file doesn't exist
        if not os.path.exists(SEGMENTED_RESULTS_FILE):
            print(f"Mocking {SEGMENTED_RESULTS_FILE} as it's not found for integration test.")
            mock_seg_data = [{
                "segment_name": "mock_segment", "status": "completed", 
                "best_cues_pareto_front": [{"glow": 0.5, "pulse_hz": 2.1, "edge": 0.6, "animation_type": 1, "fitness_values": (0.6,0.5,0.3)}],
                "config": {"num_users": 10, "num_generations": 2, "population_size": 5},
                "logbook": [{'gen': 0, 'max_touch_rate': 0.5, 'avg_touch_rate': 0.4, 'avg_accessibility':0.3, 'avg_complexity':0.5}]
            }]
            with open(SEGMENTED_RESULTS_FILE, "w") as f: json.dump(mock_seg_data, f)
        # await main_segmented_runs() # This would be the actual call in a full test run
        print(f"Segmented evolutions step assumed complete (used/created {SEGMENTED_RESULTS_FILE}).")
    except Exception as e:
        print(f"Error in segmented evolutions step: {e}")
        final_status = False

    # --- 2. Mine patterns --- 
    print("\n2️⃣ Mining patterns...")
    patterns = {}
    if os.path.exists(SEGMENTED_RESULTS_FILE):
        try:
            with open(SEGMENTED_RESULTS_FILE, "r") as f: loaded_segmented_data = json.load(f)
            miner = AffordancePatternMiner(loaded_segmented_data) # Pass loaded data
            patterns = miner.analyze_convergence_of_best_cues()
            with open(CONVERGENCE_PATTERNS_FILE, "w") as f: json.dump(patterns, f, indent=2)
            print(f"Convergence patterns mined and saved to {CONVERGENCE_PATTERNS_FILE}.")
            if not patterns: print(" (No strong convergence patterns found based on mock/current data)")
        except Exception as e:
            print(f"Error in pattern mining step: {e}")
            final_status = False
    else:
        print(f"Skipping pattern mining as {SEGMENTED_RESULTS_FILE} not found.")

    # --- 3. Discover interactions --- 
    print("\n3️⃣ Discovering interaction effects...")
    interactions = []
    if os.path.exists(SEGMENTED_RESULTS_FILE):
        try:
            # Convert best_cues_pareto_front to a DataFrame for discover_interaction_effects
            all_best_cues_for_df = []
            with open(SEGMENTED_RESULTS_FILE, "r") as f: loaded_segmented_data_for_interactions = json.load(f)
            for run_res in loaded_segmented_data_for_interactions:
                if run_res.get("status") == "completed":
                    for cue_dict in run_res.get("best_cues_pareto_front", []):
                        # Add fitness if available (e.g., first objective as primary fitness)
                        fitness_vals = cue_dict.get("fitness_values")
                        if fitness_vals and isinstance(fitness_vals, (list, tuple)) and len(fitness_vals) > 0:
                            cue_dict['fitness'] = fitness_vals[0] # Use first objective as main fitness for interactions
                        all_best_cues_for_df.append(cue_dict)
            
            if all_best_cues_for_df:
                df_results = pd.DataFrame(all_best_cues_for_df)
                interactions = discover_interaction_effects(df_results, target_metric_column='fitness')
                with open(INTERACTION_EFFECTS_FILE, "w") as f: json.dump(interactions, f, indent=2)
                print(f"Interaction effects analyzed and saved to {INTERACTION_EFFECTS_FILE}.")
                if not interactions: print(" (No significant interaction effects found based on mock/current data)")
            else:
                print("No best cue data available to form DataFrame for interaction analysis.")
        except Exception as e:
            print(f"Error in discovering interaction effects: {e}")
            final_status = False
    else:
        print(f"Skipping interaction effects as {SEGMENTED_RESULTS_FILE} not found.")

    # --- 4. Generate insights --- 
    print("\n4️⃣ Generating insights...")
    insights = []
    # Load segmented cues again for insight generator (or pass from above)
    loaded_segmented_data_for_insights = load_json_results(SEGMENTED_RESULTS_FILE) if os.path.exists(SEGMENTED_RESULTS_FILE) else []
    try:
        # Prepare segmented_best_cues dict for InsightGenerator
        segmented_cues_for_insights = {}
        if loaded_segmented_data_for_insights:
            for run_res in loaded_segmented_data_for_insights:
                if run_res.get("status") == "completed":
                     segmented_cues_for_insights[run_res.get("segment_name")] = run_res.get("best_cues_pareto_front", [])

        generator = InsightGenerator(
            convergence_patterns=patterns,
            interaction_effects=interactions, 
            segmented_best_cues=segmented_cues_for_insights,
            all_runs_summary_data=loaded_segmented_data_for_insights # Pass full data for overall stats
        )
        insights = generator.generate_key_discoveries()
        print("Key insights generated.")
        for insight_text in insights:
            print(f"  - {insight_text}")
    except Exception as e:
        print(f"Error generating insights: {e}")
        final_status = False

    # --- 5. Create Unity package --- 
    print("\n5️⃣ Creating Unity package...")
    unity_code = ""
    # Create mock discoveries if patterns is empty, for artifact generation to proceed
    mock_discoveries_for_unity = patterns if patterns else {
        "MockUniversalCue": {"glow": 0.5, "pulse_hz": 2.0, "edge": 0.5, "animation_type": 1}
    }
    try:
        unity_code = generate_unity_affordance_library(mock_discoveries_for_unity)
        with open(UNITY_CODE_FILE, "w") as f: f.write(unity_code)
        print(f"Unity package code generated and saved to {UNITY_CODE_FILE} ({len(unity_code)} chars).")
    except Exception as e:
        print(f"Error creating Unity package: {e}")
        final_status = False

    # --- 6. Generate paper draft --- 
    print("\n6️⃣ Generating paper draft (using mock LLM for this test)..."    )
    try:
        # Use precomputed data for paper compilation
        compiled_data = compile_research_data_for_paper(
            precomputed_segmented_results=loaded_segmented_data_for_insights,
            precomputed_convergence=patterns,
            precomputed_interactions=interactions,
            unity_package_path=UNITY_CODE_FILE,
            design_guidelines_path=DESIGN_GUIDELINES_FILE # Assuming this would be created elsewhere or mocked
        )
        with open(COMPILED_DATA_FILE, "w") as f: json.dump(compiled_data, f, indent=2, default=str)
        print(f"Compiled research data saved to {COMPILED_DATA_FILE}")

        mock_llm = MockLLMResearcher()
        mock_paper_gen_instance = MockAutomatedPaperGenerator(researcher=mock_llm)
        evo_paper_gen = EvolutionaryPaperGeneratorWrapper(mock_paper_gen_instance, mock_llm)
        
        # Pass the convergence patterns and interaction effects to the paper generator through compiled_data
        # The EvolutionaryPaperGeneratorWrapper is designed to extract these from the research_data_summary
        drafted_paper = evo_paper_gen.generate_full_paper_draft(compiled_data)
        with open(PAPER_DRAFT_FILE, "w") as f: json.dump(drafted_paper, f, indent=2)
        print(f"Paper draft generated and saved to {PAPER_DRAFT_FILE}.")
    except Exception as e:
        print(f"Error generating paper draft: {e}")
        import traceback
        traceback.print_exc()
        final_status = False
    
    print("\n------------------------------------------")
    if final_status:
        print("✅✅✅ Integration Test Cycle Completed Successfully (using mocks/placeholders where needed) ✅✅✅")
    else:
        print("❌❌❌ Integration Test Cycle Completed with ERRORS ❌❌❌")
    print("------------------------------------------")
    
    print(f"   - Discovered {len(patterns.keys())} convergent parameters parameters (approx)")
    print(f"   - Found {len(insights)} key insights")
    print(f"   - Generated {len(unity_code)} chars of Unity code")
    # print(f"   - Compiled data for paper: {COMPILED_DATA_FILE}")
    # print(f"   - Paper draft: {PAPER_DRAFT_FILE}")
    
    return final_status

if __name__ == "__main__":
    # Create dummy design guidelines file if it doesn't exist for the test
    if not os.path.exists(DESIGN_GUIDELINES_FILE):
        with open(DESIGN_GUIDELINES_FILE, 'w') as f: f.write("# Mock Design Guidelines Content\n- Guideline 1: ...")
    asyncio.run(run_complete_discovery_cycle()) 