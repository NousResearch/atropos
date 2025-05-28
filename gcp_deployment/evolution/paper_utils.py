import json
from typing import Dict, List, Any
import os

# Assuming these modules are in the 'evolution' package or project root
from evolution.pattern_miner import AffordancePatternMiner # If used to load and provide data
from evolution.analysis_tools import discover_interaction_effects # If used to provide data
# from evolution.artifact_generator import generate_unity_affordance_library # For artifact paths

# Assuming your existing AutomatedPaperGenerator is accessible
# from paper_generator import AutomatedPaperGenerator 
# Assuming Production24x7Pipeline or its researcher component is accessible for LLM calls
# from production_research_pipeline import Production24x7Pipeline

DEFAULT_RESULTS_FILE = "segmented_evolution_results.json"
DEFAULT_CONVERGENCE_FILE = "convergence_patterns.json" # Assuming PatternMiner saves its output
DEFAULT_INTERACTIONS_FILE = "interaction_effects.json" # Assuming AnalysisTools saves its output
DEFAULT_UNITY_PACKAGE_FILE = "DiscoveredAffordances.cs"
DEFAULT_DESIGN_GUIDELINES_FILE = "design_guidelines.md"

def load_json_results(filepath: str) -> Any:
    """Helper to load JSON data from a file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Results file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {filepath}")
        return None

def compile_research_data_for_paper(
    segmented_results_filepath: str = DEFAULT_RESULTS_FILE,
    convergence_filepath: str = DEFAULT_CONVERGENCE_FILE,
    interactions_filepath: str = DEFAULT_INTERACTIONS_FILE,
    unity_package_path: str = DEFAULT_UNITY_PACKAGE_FILE,
    design_guidelines_path: str = DEFAULT_DESIGN_GUIDELINES_FILE,
    # You might pass already computed data instead of filepaths too
    precomputed_convergence: Dict = None,
    precomputed_interactions: List[Dict] = None,
    precomputed_segmented_results: List[Dict] = None
    ) -> Dict[str, Any]:
    """
    Gathers all experimental data, analyses, and paths to artifacts for the paper.
    It prioritizes precomputed data if provided, otherwise tries to load from files.
    """
    
    print(f"Compiling research data for paper...")
    
    # 1. Load Segmented Evolution Results
    segmented_runs_data = precomputed_segmented_results if precomputed_segmented_results is not None else load_json_results(segmented_results_filepath)
    if segmented_runs_data is None:
        segmented_runs_data = [] # Ensure it's a list for downstream processing
        print(f"Warning: Segmented evolution data not available from {segmented_results_filepath}.")

    total_experiments_run = 0
    num_successful_runs = 0
    total_pareto_solutions = 0
    # Basic stats from segmented runs
    for run in segmented_runs_data:
        if run.get("status") == "completed":
            num_successful_runs +=1
            total_pareto_solutions += len(run.get("best_cues_pareto_front", []))
        # Crude count; better would be to sum generations * population_size from config in each run if available
        total_experiments_run +=1 

    # 2. Load or use Precomputed Convergence Patterns
    if precomputed_convergence is not None:
        convergence_patterns_summary = precomputed_convergence
    else:
        convergence_patterns_summary = load_json_results(convergence_filepath)
        if convergence_patterns_summary is None:
            print(f"Warning: Convergence patterns not available from {convergence_filepath}. You may need to run AffordancePatternMiner first and save its output.")
            convergence_patterns_summary = {"message": "Data not available"}

    # 3. Load or use Precomputed Interaction Effects
    if precomputed_interactions is not None:
        interaction_effects_summary = precomputed_interactions
    else:
        interaction_effects_summary = load_json_results(interactions_filepath)
        if interaction_effects_summary is None:
            print(f"Warning: Interaction effects not available from {interactions_filepath}. You may need to run discover_interaction_effects first and save its output.")
            interaction_effects_summary = {"message": "Data not available"}

    # 4. Artifact Paths (check if they exist)
    artifacts = {}
    for artifact_name, path in [
        ('unity_package_path', unity_package_path),
        # ('unreal_package_path', "path/to/UnrealAffordances.uasset"), # Example
        ('design_guidelines_path', design_guidelines_path)
    ]:
        artifacts[artifact_name] = path if os.path.exists(path) else f"File not found: {path}"

    # 5. Overall summary statistics (can be expanded)
    # Placeholder for more detailed analysis e.g. number of users, average fitness improvements
    # These would typically come from deeper analysis of the logbooks within segmented_runs_data
    # For now, focusing on what's directly available or passed.
    
    research_summary = {
        'title': "Evolutionary Discovery of Universal and Demographic-Specific VR Affordances Through Diverse Synthetic User Populations",
        'total_evolution_runs_analyzed': len(segmented_runs_data),
        'num_successful_evolution_runs': num_successful_runs,
        'total_pareto_front_solutions_found': total_pareto_solutions,
        # Example: assumes a typical setup, should be derived from actual run configs if varied
        'typical_synthetic_users_per_run': segmented_runs_data[0]['config']['num_users'] if segmented_runs_data and 'config' in segmented_runs_data[0] else "N/A",
        'typical_generations_per_run': segmented_runs_data[0]['config']['num_generations'] if segmented_runs_data and 'config' in segmented_runs_data[0] else "N/A",
        
        'convergence_patterns_summary': convergence_patterns_summary,
        'interaction_effects_summary': interaction_effects_summary,
        'segmented_run_highlights': [
            {run.get('segment_name'): f"{len(run.get('best_cues_pareto_front',[]))} solutions, status: {run.get('status')}"} 
            for run in segmented_runs_data[:5] # Show highlights from first 5 segments
        ],
        'artifacts_generated': artifacts,
        'raw_data_pointers': {
            'segmented_evolution_results': segmented_results_filepath,
            'convergence_data': convergence_filepath,
            'interaction_data': interactions_filepath
        },
        # This field will be populated by InsightGenerator for the abstract/discussion
        'key_narrative_insights': "Placeholder for narrative insights from InsightGenerator."
    }
    print(f"Research data compilation complete.")
    return research_summary

# Placeholder for actual LLM interaction. In a real scenario, this would be your pipeline.researcher.llm_generate_text
class MockLLMResearcher:
    def llm_generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        print(f"\n--- LLM Prompt for Generation (max_tokens: {max_tokens}) ---")
        print(prompt)
        # Simulate LLM response based on prompt content
        if "Abstract" in prompt:
            return "This paper presents a novel evolutionary approach to discovering VR affordances. We found that pulse_hz converges to 2.3Hz and glow intensity around 0.4 are often optimal. Demographic analysis revealed distinct preferences."
        elif "Methodology" in prompt:
            return "We implemented a multi-objective genetic algorithm using DEAP. Visual cues were parameterized by glow, pulse_hz, etc. Fitness was evaluated against a diverse synthetic user population, optimizing for touch rate, accessibility, and low complexity."
        elif "Results" in prompt:
            return "Our experiments, totaling over 50,000 evaluations, showed strong convergence for pulse_hz (mean: 2.35Hz, strength: 0.85). Glow and edge parameters exhibited synergistic effects (corr: 0.45). Segmented evolution identified that younger users prefer higher particle densities."
        return f"Draft content for the section based on the prompt: '{prompt[:100]}...'"

class MockAutomatedPaperGenerator:
    def __init__(self, researcher):
        self.researcher = researcher
    # Add other methods if EvolutionaryPaperGeneratorWrapper relies on them

# --- End Placeholder ---

class EvolutionaryPaperGeneratorWrapper:
    def __init__(self, base_paper_generator: Any, llm_researcher: Any):
        """
        Wraps an existing paper generator to tailor content for evolutionary discoveries.
        Args:
            base_paper_generator: An instance of your existing AutomatedPaperGenerator (or a mock).
            llm_researcher: The LLM interaction component (e.g., pipeline.researcher).
        """
        self.base_generator = base_paper_generator
        self.researcher = llm_researcher # This will be used for actual LLM calls

    def _generate_section_prompt(self, section_title: str, research_data_summary: Dict, context: str = "") -> str:
        title = research_data_summary.get('title', "Evolutionary Discovery of VR Affordances")
        
        prompt = f"You are an AI assistant helping to write a research paper titled: '{title}'.\n"
        prompt += f"Your current task is to draft the '{section_title}' section.\n"
        if context:
            prompt += f"Additional context for this section: {context}\n"
        
        prompt += "\nKey data points to consider for this section (refer to them implicitly or explicitly as appropriate):\n"
        
        if section_title == "Abstract":
            prompt += f"- Overall goal: Discover optimal and segment-specific VR visual affordances using evolutionary algorithms.\n"
            prompt += f"- Method summary: Multi-objective GA (DEAP), {research_data_summary.get('typical_synthetic_users_per_run', 'N/A')} diverse synthetic users, {research_data_summary.get('total_evolution_runs_analyzed', 'N/A')} total runs.\n"
            prompt += f"- Key quantitative findings (examples): {research_data_summary.get('convergence_patterns_summary', {}).get('pulse_hz',{}).get('mean', 'N/A')} Hz for pulse; {research_data_summary.get('interaction_effects_summary', [])[0] if research_data_summary.get('interaction_effects_summary') else 'N/A'}.\n"
            prompt += f"- Key qualitative findings: {research_data_summary.get('key_narrative_insights', 'Specific preferences for user segments were identified.')}\n"
            prompt += f"- Artifacts: {list(research_data_summary.get('artifacts_generated', {}).keys())}\n"
            prompt += "\nDraft a concise and impactful abstract (around 200-250 words)."

        elif section_title == "Methodology":
            prompt += f"- Genetic Algorithm: DEAP library, multi-objective (NSGA-II), population size ~{research_data_summary.get('typical_population_size', 50)}, generations ~{research_data_summary.get('typical_generations_per_run', 30)}.\n"
            prompt += f"- VisualCue Parameters: Detail the parameters evolved (e.g., glow, pulse_hz, color_hue, particle_density, animation_type, etc. - {len(research_data_summary.get('convergence_patterns_summary', {}))} main params).\n"
            prompt += f"- Synthetic User Population: Describe diversity (age, VR experience, cultural background if modeled), total N = {research_data_summary.get('typical_synthetic_users_per_run', 'N/A')}.\n"
            prompt += f"- Fitness Evaluation: Multi-objective (touch_rate, accessibility_score, -complexity_score). Detail how these were calculated (placeholder or Padres API). Total evaluations: {research_data_summary.get('total_experiments_triggered', 'many')} * evaluations_per_run.\n"
            prompt += f"- Segmented Evolution: Approach for {len(research_data_summary.get('segmented_run_highlights',[]))} user segments.\n"
            prompt += "\nDescribe the experimental setup and evolutionary process in detail."

        elif section_title == "Results":
            prompt += f"- Convergence Analysis: Present findings from AffordancePatternMiner (e.g., {json.dumps(research_data_summary.get('convergence_patterns_summary', {}), indent=0)}).\n"
            prompt += f"- Interaction Effects: Detail synergistic/antagonistic effects found (e.g., {json.dumps(research_data_summary.get('interaction_effects_summary', []), indent=0)}).\n"
            prompt += f"- Demographic Preferences: Summarize key differences found from segmented runs (e.g., based on {json.dumps(research_data_summary.get('segmented_run_highlights', []), indent=0)}).\n"
            prompt += f"- Pareto Front examples: Briefly mention the nature of trade-offs found (e.g., high touch-rate vs. low complexity).\n"
            prompt += "\nPresent the main findings supported by data. Refer to figures hypothetically (e.g., 'Figure X shows...')."
        
        elif section_title == "Discussion":
            prompt += f"- Interpretation of key findings: What do the convergence patterns, interactions, and demographic differences mean for VR design?\n"
            prompt += f"- Link to prior work (briefly): How do these findings align or contrast with existing literature on affordances?\n"
            prompt += f"- Implications: For VR developers, designers, researchers.\n"
            prompt += f"- Limitations: Current fitness model (placeholder/Padres details), scope of synthetic users, range of cues explored.\n"
            prompt += f"- Future Work: Integration with real user studies, broader cue/user spaces, more complex interaction models.\n"
            prompt += "\nDiscuss the significance of the results, limitations, and future research directions."
        else:
            prompt += f"- General data summary: {json.dumps(research_data_summary, indent=0, default=str)}\n"
            prompt += "\nDraft this section of the paper."
        
        prompt += "\nWrite in a formal academic style suitable for a Human-Computer Interaction (HCI) or Computer Graphics conference (e.g., CHI, SIGGRAPH, IEEE VR). Ensure clarity, conciseness, and a scientific tone."
        return prompt

    def generate_full_paper_draft(self, research_data_summary: Dict[str, Any]) -> Dict[str, str]:
        """Generates a draft for all main sections of the paper."""
        paper_draft = {}
        
        # Ensure title is in the summary for consistent use in prompts
        if 'title' not in research_data_summary:
            research_data_summary['title'] = "Evolutionary Discovery of Universal and Demographic-Specific VR Affordances Through Diverse Synthetic User Populations"
        
        # Simple narrative insights based on mock example structure
        if not research_data_summary.get('key_narrative_insights') or research_data_summary['key_narrative_insights'] == "Placeholder for narrative insights from InsightGenerator.":
            narratives = []
            if self.convergence_patterns.get('pulse_hz',{}).get('convergence_strength',0) > 0.7:
                narratives.append(f"Pulse frequency strongly converged towards {self.convergence_patterns['pulse_hz']['mean']:.1f}Hz.")
            if any(eff['effect_type'] == 'synergistic' for eff in self.interaction_effects):
                narratives.append("Synergistic effects between some cue parameters were observed.")
            if self.segmented_best_cues: 
                narratives.append("Distinct cue preferences were noted across different user demographics.")
            research_data_summary['key_narrative_insights'] = " ".join(narratives) if narratives else "Key quantitative patterns were identified."

        sections_to_draft = {
            "Abstract": "",
            "Introduction": "Motivate VR affordance problem, introduce evolutionary approach, state contributions.",
            "Related Work": "Briefly cover VR affordances, computational discovery, evolutionary methods.",
            "Methodology": "Detail GA setup, cue/user parameterization, fitness, segmented runs.",
            "Results": "Present convergence, interaction, and demographic findings. Refer to placeholder figures.",
            "Discussion": "Interpret findings, implications, limitations, future work.",
            "Conclusion": "Summarize key contributions and main takeaways."
            # "Acknowledgements": "(Standard acknowledgements)",
            # "References": "(Placeholder for BibTeX)"
        }

        max_tokens_map = {
            "Abstract": 300,
            "Introduction": 700,
            "Related Work": 600,
            "Methodology": 1000,
            "Results": 1000,
            "Discussion": 800,
            "Conclusion": 300
        }

        for section, context in sections_to_draft.items():
            print(f"Drafting section: {section}...")
            prompt_for_section = self._generate_section_prompt(section, research_data_summary, context)
            max_tokens = max_tokens_map.get(section, 800)
            try:
                drafted_text = self.researcher.llm_generate_text(prompt_for_section, max_tokens=max_tokens)
                paper_draft[section] = drafted_text
                print(f"  Draft for {section} generated ({len(drafted_text)} chars).")
            except Exception as e:
                print(f"  Error generating draft for {section}: {e}")
                paper_draft[section] = f"Error generating content for {section}: {e}"
        
        # Placeholder for actual figure generation logic (e.g., using matplotlib/seaborn)
        paper_draft['_figures_todo'] = [
            "Figure 1: Convergence plot for key parameters (e.g., pulse_hz) over generations.",
            "Figure 2: Heatmap or bar chart illustrating cue parameter preferences for different demographic segments.",
            "Figure 3: Matrix/diagram showing significant interaction effects between cue parameters.",
            "Table 1: Summary of key discovered universal and segment-specific cue parameters."
        ]
        
        return paper_draft

if __name__ == '__main__':
    # Create dummy files for testing if they don't exist
    if not os.path.exists(DEFAULT_RESULTS_FILE):
        with open(DEFAULT_RESULTS_FILE, 'w') as f:
            # Example structure, actual data from run_segmented_evolutions.py
            json.dump([{
                "segment_name": "test_segment", "status": "completed", 
                "best_cues_pareto_front": [{"glow": 0.5, "pulse_hz": 2.0, "edge": 0.5}],
                "config": {"num_users": 50, "num_generations": 10},
                "logbook": []
            }], f)
    if not os.path.exists(DEFAULT_CONVERGENCE_FILE):
        with open(DEFAULT_CONVERGENCE_FILE, 'w') as f:
            json.dump({"pulse_hz": {"mean": 2.5, "std": 0.1, "convergence_strength": 0.9}}, f)
    if not os.path.exists(DEFAULT_INTERACTIONS_FILE):
        with open(DEFAULT_INTERACTIONS_FILE, 'w') as f:
            json.dump([{"parameter_pair": ["glow", "edge"], "correlation_with_target": 0.5}], f)
    if not os.path.exists(DEFAULT_UNITY_PACKAGE_FILE):
        with open(DEFAULT_UNITY_PACKAGE_FILE, 'w') as f: f.write("// Mock Unity Package Content")
    if not os.path.exists(DEFAULT_DESIGN_GUIDELINES_FILE):
        with open(DEFAULT_DESIGN_GUIDELINES_FILE, 'w') as f: f.write("# Mock Design Guidelines")

    compiled_data = compile_research_data_for_paper()
    print("\n--- Compiled Research Data for Paper ---")
    # Pretty print the json
    print(json.dumps(compiled_data, indent=2))

    # To make this truly useful, AffordancePatternMiner and discover_interaction_effects
    # would ideally save their outputs to JSON files that this function can then load,
    # or their main analysis methods would be called here if they operate on the loaded segmented_runs_data. 

    print("Testing EvolutionaryPaperGeneratorWrapper...")
    
    # Use a mock researcher for testing without actual LLM calls / API keys
    mock_researcher_instance = MockLLMResearcher()
    mock_base_paper_gen = MockAutomatedPaperGenerator(researcher=mock_researcher_instance)
    
    evo_paper_gen = EvolutionaryPaperGeneratorWrapper(
        base_paper_generator=mock_base_paper_gen, 
        llm_researcher=mock_researcher_instance
    )
    
    # Load or create mock compiled_research_data (from Day 11)
    # For testing, we can call the compile function from paper_utils if it's importable
    # Or use a simpler mock data structure here
    mock_research_data = {
        'title': "Evolutionary Discovery of VR Affordances (Test Paper)",
        'total_evolution_runs_analyzed': 50,
        'num_successful_evolution_runs': 45,
        'total_pareto_front_solutions_found': 250,
        'typical_synthetic_users_per_run': 100,
        'typical_generations_per_run': 30,
        'convergence_patterns_summary': {
            'pulse_hz': {'mean': 2.35, 'std': 0.2, 'convergence_strength': 0.85, 'min_observed': 2.0, 'max_observed': 2.8},
            'glow': {'mean': 0.38, 'std': 0.1, 'convergence_strength': 0.72, 'min_observed': 0.2, 'max_observed': 0.6}
        },
        'interaction_effects_summary': [
            {'parameter_pair': ('glow', 'edge'), 'correlation_with_target': 0.45, 'effect_type': 'synergistic'}
        ],
        'segmented_run_highlights': [
            {"young_gamers": "15 solutions, status: completed"},
            {"seniors_low_va": "12 solutions, status: completed"}
        ],
        'artifacts_generated': {'unity_package_path': 'DiscoveredAffordances.cs'},
        'key_narrative_insights': "Discovered optimal pulse around 2.3Hz and glow-edge synergy."
    }

    drafted_sections = evo_paper_gen.generate_full_paper_draft(mock_research_data)

    print("\n--- Drafted Paper Sections ---")
    for section, content in drafted_sections.items():
        if section != '_figures_todo':
            print(f"\n## {section}\n{content}")
    print("\nFigures to generate:")
    for fig_desc in drafted_sections.get('_figures_todo', []):
        print(f"- {fig_desc}") 