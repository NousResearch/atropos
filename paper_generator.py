import json  # For potentially handling raw_data if needed, though mostly for LLM interaction
import logging
import os
from datetime import datetime

from google.cloud import storage  # Import GCS client

# Assuming these classes are in the same directory or accessible via PYTHONPATH
# from bigquery_manager import ResearchDataManager
# from enhanced_padres_perplexity import SimplePadresResearch

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AutomatedPaperGenerator:
    """Generates automated research papers from experimental data."""

    def __init__(self, data_manager, researcher):
        """
        Initializes the AutomatedPaperGenerator.
        Args:
            data_manager: An instance of ResearchDataManager.
            researcher: An instance of SimplePadresResearch (or a similar class with call_claude and search_perplexity).
        """
        if not data_manager:
            logger.error("CRITICAL: ResearchDataManager instance is required.")
            raise ValueError("ResearchDataManager cannot be None.")
        if not researcher:
            logger.error(
                "CRITICAL: Researcher instance (e.g., SimplePadresResearch) is required."
            )
            raise ValueError("Researcher instance cannot be None.")

        self.data_manager = data_manager
        self.researcher = researcher
        self.gcs_bucket_name = os.getenv("PAPER_OUTPUT_GCS_BUCKET")
        if self.gcs_bucket_name:
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket(self.gcs_bucket_name)
                logger.info(
                    f"AutomatedPaperGenerator initialized to save papers to GCS bucket: {self.gcs_bucket_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize GCS client or bucket {self.gcs_bucket_name}: {e}. Papers will be saved locally.",
                    exc_info=True,
                )
                self.gcs_bucket_name = None  # Fallback to local saving
                self.storage_client = None
                self.bucket = None
        else:
            logger.warning(
                "PAPER_OUTPUT_GCS_BUCKET env var not set. Papers will be saved locally to the container."
            )
            self.storage_client = None
            self.bucket = None
        logger.info("AutomatedPaperGenerator initialization complete.")

    def _call_llm_for_section(self, prompt: str, section_name: str) -> str:
        """Helper function to call the configured LLM and handle potential errors."""
        try:
            logger.info(f"Generating section: {section_name} using configured LLM...")
            # Assuming self.researcher now has a generic call_llm_for_text method
            response = self.researcher.call_llm_for_text(prompt)
            logger.info(f"Successfully generated section: {section_name}.")
            return response
        except Exception as e:
            logger.error(
                "Error generating section "{section_name}' with LLM: {e}",
                exc_info=True,
            )
            return f"[Error generating {section_name}. Details: {e}]"

    def _generate_abstract(self, all_experiments_data: list) -> str:
        """Generates the abstract section of the research paper using all available data."""
        if not all_experiments_data:
            return "No data available to generate an abstract."

        num_experiments = len(all_experiments_data)
        successful_experiments = sum(
            1 for exp in all_experiments_data if exp.get("padres_success")
        )
        success_rate = (
            (successful_experiments / num_experiments) * 100
            if num_experiments > 0
            else 0
        )

        trend_info = "Consistent perfect performance (100% success, score 1.0) was observed across all data segments analyzed."

        prompt = """
        Write a concise and compelling research paper abstract (150-250 words) for a study on Large Language Model (LLM) spatial reasoning capabilities, evaluated through interactions in dynamic physical simulations.

        Key information from the conducted experiments (full dataset):
        - Total experiments conducted: {num_experiments}
        - Overall success rate: {success_rate:.2f}% (achieving perfect scores and zero distance metrics).
        - Performance Trend: {trend_info}

        The abstract should cover:
        1.  The context and importance of LLM spatial reasoning.
        2.  The novel methodology used (e.g., AI analyzing and interacting with physical simulations in the 'Padres Spatial RL Environment').
        3.  The key quantitative finding of consistently perfect performance across all {num_experiments} experiments.
        4.  Briefly state the implications of these striking findings, suggesting that while promising, they warrant further investigation into task complexity, environmental factors, and the LLM's interaction paradigm.

        Maintain an academic tone. Focus on the automated analysis and continuous experimentation aspect.
        """
        return self._call_llm_for_section(prompt, "Abstract")

    def _generate_introduction(self) -> str:
        """Generates the introduction section, prompting for more detail."""
        prompt = """
        Write a compelling introduction for a research paper titled 'Automated Analysis of Large Language Model Spatial Reasoning Capabilities in Dynamic Physical Simulations'.
        The introduction should:
        1. Establish the importance of spatial reasoning for AI and LLMs, particularly in dynamic contexts.
        2. Briefly review existing approaches and their limitations in evaluating dynamic spatial understanding, noting the challenges of manual or static evaluations.
        3. Introduce the novel approach of this research: a continuous, automated experimentation pipeline utilizing the 'Padres Spatial RL Environment' to assess a Gemini-based LLM (gemini-1.5-flash). Emphasize the goal of achieving scalable and rigorous evaluation.
        4. State the main objectives: to present this evaluation framework, report on the LLM's performance across a large set of diverse spatial tasks, and discuss the implications of the findings.
        5. Provide a roadmap for the rest of the paper.

        Aim for 3-4 well-developed paragraphs. Maintain an academic tone.
        """
        logger.info("Generating Introduction section with prompts for more detail.")
        return self._call_llm_for_section(prompt, "Introduction")

    def _generate_methodology(self) -> str:
        """Generates the methodology section, prompting for more detail."""
        prompt = """
        Write a detailed Methodology section for a research paper on LLM spatial reasoning in physical simulations.
        Describe the following aspects with as much specificity as a generative model can infer or create based on the context of an advanced automated research pipeline:
        1.  **Experimental Setup: The 'Padres Spatial RL Environment'**:
            *   Describe its core principles (e.g., 2D/3D, physics engine basis if any like PyBullet, grid-based or continuous).
            *   Detail the types of objects, their properties (e.g., mass, friction, visual appearance if relevant to textual description), and their typical interactions.
            *   Explain the action space available to the LLM (e.g., discrete commands like 'move north', 'pick up object X', or more continuous parameters).
        2.  **Task Design and Generation**:
            *   Elaborate on the categories of spatial tasks (e.g., navigation, object manipulation - picking, placing, stacking, pathfinding, relative positioning).
            *   Explain how the {num_experiments_placeholder} distinct tasks were generated. Were they procedurally generated? If so, what parameters were varied to ensure diversity in goals, initial conditions, and object configurations? Provide illustrative examples if possible.
            *   Define how 'task success,' 'score' (e.g., 0.0 to 1.0 scale, criteria for achieving 1.0), and 'distance metric' (e.g., Euclidean distance to target, relevance if 0.00) were precisely defined, measured, and logged for each task.
        3.  **LLM Integration and Interaction Loop**:
            *   Specify the LLM used: "a Gemini-based model (gemini-1.5-flash)".
            *   Detail how the environment's state was translated into the textual description provided to the LLM. Was it a raw dump of coordinates, a structured summary, or a narrative description? Emphasize that this prompt was "concise and unambiguous."
            *   Explain how the LLM's natural language output was parsed and translated into actionable commands for the simulation.
        4.  **Data Collection and Automated Pipeline**:
            *   Briefly describe the 24/7 automated system, highlighting its role in continuous experimentation.
            *   Mention the parameters logged for each experiment (success, score, distance, task completion, raw interaction logs, LLM analysis text, Perplexity context) and their storage in GCS (JSONL format).

        Ensure clarity and provide enough detail to allow a reader to understand the rigor of the experimental process. Maintain an academic tone. Use subheadings for clarity if appropriate.
        (Replace {num_experiments_placeholder} with the actual number of experiments when this prompt is used, if possible, otherwise the LLM should treat it as a variable to be filled contextually).
        """
        logger.info("Generating Methodology section with prompts for more detail.")
        return self._call_llm_for_section(prompt, "Methodology")

    def _generate_results(self, all_experiments_data: list) -> str:
        """Generates the results section, including some basic statistics from all available data."""
        if not all_experiments_data:
            return "No experimental data available to generate results."

        num_experiments = len(all_experiments_data)
        successful_experiments = sum(
            1 for exp in all_experiments_data if exp.get("padres_success")
        )
        success_rate = (
            (successful_experiments / num_experiments) * 100
            if num_experiments > 0
            else 0
        )

        scores = [
            exp.get("score", 0)
            for exp in all_experiments_data
            if exp.get("score") is not None
            and isinstance(exp.get("score"), (int, float))
        ]
        avg_score = sum(scores) / len(scores) if scores else 0

        distances = [
            exp.get("distance", 0)
            for exp in all_experiments_data
            if exp.get("distance") is not None
            and isinstance(exp.get("distance"), (int, float))
        ]
        avg_distance = sum(distances) / len(distances) if distances else 0

        tasks_completed = sum(
            1 for exp in all_experiments_data if exp.get("task_completed")
        )

        trend_info = "Consistent perfect performance (100% success, score 1.0, distance 0.0) was observed across all data segments and metrics, indicating no degradation or variation in performance over the dataset."

        prompt = """
        Write the Results section for a research paper on LLM spatial reasoning, based on the full dataset of {num_experiments} experiments.

        Quantitative summary of all experiments:
        - Total number of experiments conducted: {num_experiments}
        - Number of successful experiments (padres_success = True): {successful_experiments} (This constitutes {success_rate:.2f}% of the total)
        - Overall success rate: {success_rate:.2f}%
        - Average experiment score: {avg_score:.3f}
        - Average distance metric: {avg_distance:.3f}
        - Total tasks marked as completed: {tasks_completed}
        - Performance trend insights: {trend_info}

        The Results section should:
        1.  Clearly and objectively present these key quantitative findings.
        2.  Emphasize the consistency of the perfect performance across the entire dataset.
        3.  Avoid interpretation or speculation (save that for the Discussion section). Stick to reporting the outcomes as reflected in the data.

        Maintain an objective and academic tone. Use precise language.
        """
        return self._call_llm_for_section(prompt, "Results")

    def _generate_related_work(self) -> str:
        """Generates the related work section using Perplexity AI for literature search."""
        logger.info(
            "Initiating literature search for Related Work section using Perplexity AI..."
        )
        research_query = """
        Recent academic research (2023-2025) on:
        1. Large Language Models (LLMs) and spatial reasoning capabilities, including models like Gemini and Claude.
        2. Evaluation of LLMs in interactive or simulated physical environments.
        3. Reinforcement learning approaches for spatial tasks using LLMs.
        4. Benchmarks and methodologies for assessing spatial understanding in AI.
        5. Automated AI research pipelines and their application in LLM studies.
        Focus on peer-reviewed papers, conference proceedings (e.g., NeurIPS, ICML, ICLR, CVPR, ACL), and prominent arXiv preprints.
        """
        try:
            # Assuming self.researcher has a method like search_perplexity
            search_results_summary = self.researcher.search_perplexity(research_query)
            logger.info("Perplexity AI search completed.")
        except Exception as e:
            logger.error(f"Error during Perplexity AI search: {e}", exc_info=True)
            return "[Error conducting literature search with Perplexity AI. Related Work section cannot be generated.]"

        prompt = """
        Based on the following summary of recent academic research, write a comprehensive 'Related Work' section for a research paper on automated analysis of LLM spatial reasoning in dynamic physical simulations.

        Research Summary from Perplexity AI:
        --- BEGIN PERPLEXITY SUMMARY ---
        {search_results_summary}
        --- END PERPLEXITY SUMMARY ---

        The 'Related Work' section should be structured to cover:
        1.  **LLMs and Spatial Reasoning**: Discuss existing research on the spatial abilities and limitations of current LLMs (mentioning models like Gemini where appropriate if the summary includes them).
        2.  **Simulation-Based Evaluation**: Review studies that use simulated environments to test AI, particularly for physical or spatial tasks.
        3.  **Automated Experimentation and RL in LLM Contexts**: Briefly touch upon relevant work in automated AI research or RL applied to LLMs for task performance, if covered in the summary.
        4.  **Benchmarking Spatial AI**: Mention existing benchmarks or common methodologies for evaluating spatial intelligence.
        5.  **Positioning Current Work**: Subtly highlight how the current paper's approach (continuous, automated analysis in dynamic simulations using a Gemini-based model) builds upon or differs from the cited works.

        Organize the section logically with clear paragraphs. Maintain an academic tone and cite works appropriately (e.g., by referring to them as 'Author et al. (Year) found...' or similar, based on the style of the summary provided by Perplexity).
        This section should synthesize the provided information, not just list it.
        """
        return self._call_llm_for_section(prompt, "Related Work")

    def _generate_discussion(self, all_experiments_data: list) -> str:
        """Generates the discussion section, critically reflecting on the 100% success rate."""
        num_experiments = len(all_experiments_data) if all_experiments_data else 0
        avg_score_all_time = 0
        if all_experiments_data:
            scores = [
                exp.get("score", 0)
                for exp in all_experiments_data
                if exp.get("score") is not None
                and isinstance(exp.get("score"), (int, float))
            ]
            if scores:
                avg_score_all_time = sum(scores) / len(scores)

        prompt = """
        Write a critical and insightful Discussion section for a research paper that has reported a 100% success rate across {num_experiments} spatial reasoning experiments conducted with a Gemini-based LLM (gemini-1.5-flash) in the 'Padres Spatial RL Environment'. The average score was {avg_score_all_time:.3f} and average distance metric was 0.00.

        This section must thoughtfully address this extraordinary result. Key aspects to cover:
        1.  **Interpreting the Perfect Performance**:
            *   Acknowledge the reported 100% success rate as a striking and unexpected outcome, potentially challenging conventional views of LLM capabilities in dynamic spatial tasks if the tasks were sufficiently complex.
            *   Discuss potential factors contributing to this perfect performance. Consider:
                *   **Task and Environment Design**: Could the 'Padres Spatial RL Environment' or the specific nature of the {num_experiments} tasks (e.g., navigation, object manipulation, pathfinding as mentioned in abstract/methodology) be simpler or more constrained than typical real-world spatial challenges?
                *   **State Representation & Prompting**: How might the "concise and unambiguous textual description of the environment's state" provided to the LLM have influenced performance? Is it possible this explicitness guided the LLM effectively, reducing the reasoning burden?
                *   **LLM Capabilities**: Is it plausible that the Gemini-1.5-flash model possesses an unexpectedly high intrinsic capability for these specific types of simulated spatial tasks when provided with clear textual inputs/outputs?
        2.  **Comparison with Broader LLM Research**:
            *   Contrast this finding of 100% success with the general understanding from existing literature on LLM spatial reasoning, which often highlights ongoing challenges, variability, and limitations, especially in dynamic or zero-shot settings.
        3.  **Strengths and Limitations of the Study**:
            *   Strengths: Highlight the value of the automated pipeline for large-scale, consistent experimentation.
            *   Limitations: Critically evaluate the generalizability of these perfect results. Emphasize that 100% success in this specific simulated environment with its particular task set does not necessarily translate to universal spatial mastery. Discuss the need for validation on more diverse, complex, and potentially ambiguous benchmarks.
        4.  **Implications of the Research**:
            *   If the high performance is genuine and replicable under more challenging conditions, what are the implications for embodied AI and robotics?
            *   If potentially due to the experimental setup, what does this teach us about designing effective benchmarks and evaluation methods for LLM spatial reasoning?
        5.  **Future Work**:
            *   Suggest specific directions to rigorously test the observed proficiency. This MUST include:
                *   Introducing tasks with higher complexity, ambiguity, or requiring multi-step inferential reasoning.
                *   Varying the fidelity and nature of the simulation.
                *   Testing with less explicit state representations or more complex natural language commands.
                *   Comparative studies with other LLMs and on established external benchmarks.
                *   Probing for robustness against minor perturbations or noise in the environment.

        Maintain a balanced, critical, and academic tone. The goal is not to outright dismiss the results but to explore them thoughtfully and scientifically, acknowledging both their potential significance and the need for cautious interpretation and further validation. Avoid making definitive claims that are not supported by a critical analysis of the perfect score itself.
        """
        logger.info(
            "Generating Discussion section with critical reflection on 100% success."
        )
        return self._call_llm_for_section(prompt, "Discussion")

    def _generate_conclusion(self, all_experiments_data: list) -> str:
        """Generates the conclusion section based on all available data."""
        num_experiments = len(all_experiments_data) if all_experiments_data else 0
        # Simplified trend for conclusion
        overall_success_rate = 0
        if num_experiments > 0:
            successful_experiments = sum(
                1 for exp in all_experiments_data if exp.get("padres_success")
            )
            overall_success_rate = (successful_experiments / num_experiments) * 100

        prompt = """
        Write a Conclusion section for the research paper. This section should summarize the main contributions, key findings, and future outlook of the research on automated LLM (Gemini-based) spatial reasoning analysis, based on all {num_experiments} experiments conducted.
        Overall success rate was {overall_success_rate:.2f}%.

        The Conclusion should concisely:
        1.  Restate the primary objectives of the study.
        2.  Summarize the key findings regarding the LLM's performance in dynamic spatial tasks, as determined by the automated pipeline over the full dataset.
        3.  Reiterate the significance of the automated evaluation methodology.
        4.  Offer a final perspective on the future of LLMs in spatial reasoning and the role of continuous, automated research in advancing this area.

        Avoid introducing new information. Keep it focused and impactful (1-2 paragraphs).
        """
        logger.info("Generating Conclusion section based on all available data.")
        return self._call_llm_for_section(prompt, "Conclusion")

    def _combine_paper_sections(
        self, sections: dict, num_total_experiments: int
    ) -> str:
        """Combines all generated sections into a single Markdown formatted paper string."""
        paper_template = """
# Automated Analysis of Large Language Model Spatial Reasoning Capabilities in Dynamic Physical Simulations

**Date Generated:** {datetime.now().strftime("%B %d, %Y %H:%M:%S UTC")}
**Reporting Period:** Entire dataset analysis as of {datetime.now().strftime("%Y-%m-%d")}
**LLM Used for Analysis:** Gemini-based model (e.g., gemini-1.5-flash)
**Total Experiments Analyzed:** {num_total_experiments}

## Abstract
{sections.get('abstract', '[Abstract not generated]')}

## 1. Introduction
{sections.get('introduction', '[Introduction not generated]')}

## 2. Methodology
{sections.get('methodology', '[Methodology not generated]')}

## 3. Results
{sections.get('results', '[Results not generated]')}

## 4. Related Work
{sections.get('related_work', '[Related Work not generated]')}

## 5. Discussion
{sections.get('discussion', '[Discussion not generated]')}

## 6. Conclusion
{sections.get('conclusion', '[Conclusion not generated]')}

---
*This paper was automatically generated by an AI research system. This iteration analyzed data from {num_total_experiments} experiments conducted.*
"""
        return paper_template

    def generate_weekly_paper(self) -> str:
        """Generates a complete research paper from all accumulated data."""
        logger.info(
            "Initiating research paper generation process (using Gemini for text generation), analyzing ALL available data..."
        )

        try:
            # 1. Gather all data
            logger.info("Fetching all data from ResearchDataManager...")
            all_experiments_data = self.data_manager.get_all_experiments_for_paper()
            logger.info(f"Data fetched: {len(all_experiments_data)} total experiments.")

            if not all_experiments_data:
                logger.warning(
                    "No data available from GCS. Paper generation will be very minimal or skipped."
                )
                return "[No paper generated due to no data found in GCS.]"

            # 2. Generate paper sections using all data
            paper_sections = {}
            # Pass all_experiments_data to relevant section generators
            paper_sections["abstract"] = self._generate_abstract(all_experiments_data)
            paper_sections["introduction"] = (
                self._generate_introduction()
            )  # Introduction is generic
            paper_sections["methodology"] = (
                self._generate_methodology()
            )  # Methodology is generic
            paper_sections["results"] = self._generate_results(all_experiments_data)
            paper_sections["related_work"] = (
                self._generate_related_work()
            )  # Related work is generic for now
            paper_sections["discussion"] = self._generate_discussion(
                all_experiments_data
            )
            paper_sections["conclusion"] = self._generate_conclusion(
                all_experiments_data
            )

            # 3. Combine into full paper
            full_paper_md = self._combine_paper_sections(
                paper_sections, len(all_experiments_data)
            )

            # 4. Save paper to a Markdown file
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"spatial_ai_research_paper_gemini_{timestamp_str}.md"

            if self.bucket:  # If GCS bucket is configured
                blob = self.bucket.blob(
                    f"generated_papers/{base_filename}"
                )  # Store in a folder
                blob.upload_from_string(full_paper_md, content_type="text/markdown")
                gcs_path = (
                    f"gs://{self.gcs_bucket_name}/generated_papers/{base_filename}"
                )
                logger.info(
                    f"✅ Research paper uploaded successfully to GCS: {gcs_path}"
                )
                return gcs_path
            else:  # Fallback to local saving (within container's temp fs)
                # Ensure a directory exists if saving to a subdirectory, e.g., 'generated_papers/'
                # os.makedirs("generated_papers", exist_ok=True)
                # filepath = os.path.join("generated_papers", base_filename)
                filepath = base_filename
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(full_paper_md)
                logger.info(
                    f"✅ Research paper saved locally (in container) as: {filepath}"
                )
                return filepath  # This path is only valid within the container instance

        except Exception as e:
            logger.error(
                f"Critical error during research paper generation: {e}", exc_info=True
            )
            # Depending on desired behavior, could save a partial paper or just log
            return f"[Failed to generate paper. Error: {e}]"


# Example usage (for local testing)
if __name__ == "__main__":
    from dotenv import load_dotenv

    from bigquery_manager import (  # Assuming it's in the same directory for local test
        ResearchDataManager,
    )
    from enhanced_padres_perplexity import (  # Assuming for local test
        SimplePadresResearch,
    )

    load_dotenv()
    logger.info(
        "--- Running AutomatedPaperGenerator local test (with Gemini & GCS) ---"
    )

    gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    gemini_api_key = os.getenv("GEMINI_API_KEY")  # Changed from ANTHROPIC_API_KEY
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    # For local test of GCS saving, you might want to set PAPER_OUTPUT_GCS_BUCKET in your .env
    # Ensure your local ADC (gcloud auth application-default login) has permissions to write to that bucket.
    # Example for .env: PAPER_OUTPUT_GCS_BUCKET="your-test-bucket-name"

    if not all([gcp_project_id, gemini_api_key, perplexity_api_key]):
        logger.error(
            "Missing one or more required environment variables for local test: "
            "GOOGLE_CLOUD_PROJECT, GEMINI_API_KEY, PERPLEXITY_API_KEY"
        )
    else:
        try:
            # Initialize dependencies
            mock_data_manager = ResearchDataManager(project_id=gcp_project_id)
            mock_researcher = SimplePadresResearch()  # Assumes it loads keys from env

            # Create a dummy experiment to ensure some data exists for testing
            # In a real scenario, data would already be in BigQuery
            logger.info("Storing a dummy experiment for testing paper generation...")
            dummy_exp_id = (
                f"paper_gen_dummy_exp_gemini_{int(datetime.now().timestamp())}"
            )
            mock_data_manager.store_experiment(
                {
                    "experiment_id": dummy_exp_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "padres_success": True,
                    "score": 0.8,
                    "distance": 5.0,
                    "task_completed": True,
                    "claude_analysis": "[This would be Gemini analysis now]",  # Field name in BQ if it was 'claude_analysis'
                    "llm_analysis": "Dummy Gemini analysis for paper gen.",  # New field used by SimplePadresResearch
                    "perplexity_research": "Dummy Perplexity research.",
                    "raw_data": {"test": "data"},
                }
            )

            paper_generator_instance = AutomatedPaperGenerator(
                mock_data_manager, mock_researcher
            )

            logger.info("Starting research paper generation test...")
            generated_file_path = paper_generator_instance.generate_weekly_paper()
            logger.info(
                f"Paper generation test completed. File/Path: {generated_file_path}"
            )

        except Exception as e:
            logger.error(
                f"Error during AutomatedPaperGenerator local test: {e}", exc_info=True
            )
    logger.info("--- AutomatedPaperGenerator local test finished ---")
