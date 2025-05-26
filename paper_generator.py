import os
import logging
from datetime import datetime
import json # For potentially handling raw_data if needed, though mostly for LLM interaction
from google.cloud import storage # Import GCS client

# Assuming these classes are in the same directory or accessible via PYTHONPATH
# from bigquery_manager import ResearchDataManager
# from enhanced_padres_perplexity import SimplePadresResearch

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
            logger.error("CRITICAL: Researcher instance (e.g., SimplePadresResearch) is required.")
            raise ValueError("Researcher instance cannot be None.")
            
        self.data_manager = data_manager
        self.researcher = researcher
        self.gcs_bucket_name = os.getenv("PAPER_OUTPUT_GCS_BUCKET")
        if self.gcs_bucket_name:
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket(self.gcs_bucket_name)
                logger.info(f"AutomatedPaperGenerator initialized to save papers to GCS bucket: {self.gcs_bucket_name}")
            except Exception as e:
                logger.error(f"Failed to initialize GCS client or bucket {self.gcs_bucket_name}: {e}. Papers will be saved locally.", exc_info=True)
                self.gcs_bucket_name = None # Fallback to local saving
                self.storage_client = None
                self.bucket = None
        else:
            logger.warning("PAPER_OUTPUT_GCS_BUCKET env var not set. Papers will be saved locally to the container.")
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
            logger.error(f"Error generating section '{section_name}' with LLM: {e}", exc_info=True)
            return f"[Error generating {section_name}. Details: {e}]"

    def _generate_abstract(self, all_experiments_data: list) -> str:
        """Generates the abstract section of the research paper using all available data."""
        if not all_experiments_data:
            return "No data available to generate an abstract."
        
        num_experiments = len(all_experiments_data)
        successful_experiments = sum(1 for exp in all_experiments_data if exp.get('padres_success'))
        success_rate = (successful_experiments / num_experiments) * 100 if num_experiments > 0 else 0
        
        # Simple trend: success rate of first half vs second half if enough data
        trend_info = "Trend data not specifically analyzed for this abstract version."
        if num_experiments >= 10:
            first_half_success = sum(1 for exp in all_experiments_data[:num_experiments//2] if exp.get('padres_success'))
            second_half_success = sum(1 for exp in all_experiments_data[num_experiments//2:] if exp.get('padres_success'))
            trend_info = f"Success rate in first half of dataset: {first_half_success/(num_experiments//2)*100:.2f}%, second half: {second_half_success/(num_experiments - num_experiments//2)*100:.2f}%."

        prompt = f"""
        Write a concise and compelling research paper abstract (150-250 words) for a study on Large Language Model (LLM) spatial reasoning capabilities, evaluated through interactions in dynamic physical simulations.

        Key information from the conducted experiments (full dataset):
        - Total experiments conducted: {num_experiments}
        - Overall success rate: {success_rate:.2f}%
        - Brief trend summary: {trend_info}
        
        The abstract should cover:
        1.  The context and importance of LLM spatial reasoning.
        2.  The novel methodology used (e.g., AI analyzing and interacting with physical simulations).
        3.  Key quantitative performance insights and findings from the experiments.
        4.  Briefly state the implications of these findings for the field of spatial AI and LLM development.
        
        Maintain an academic tone and structure. Focus on the automated analysis and continuous experimentation aspect.
        """
        return self._call_llm_for_section(prompt, "Abstract")

    def _generate_introduction(self) -> str:
        """Generates the introduction section. (Placeholder - requires significant prompt engineering)."""
        prompt = """
        Write a compelling introduction for a research paper titled 'Automated Analysis of Large Language Model Spatial Reasoning Capabilities in Dynamic Physical Simulations'.
        The introduction should:
        1. Establish the importance of spatial reasoning for AI and LLMs.
        2. Briefly review existing approaches and their limitations in evaluating dynamic spatial understanding.
        3. Introduce the novel approach of this research: continuous, automated experimentation in simulated physical environments.
        4. State the main objectives and contributions of the paper (e.g., to present a framework for such evaluation and report on ongoing findings).
        5. Provide a roadmap for the rest of the paper.
        
        Aim for 3-4 paragraphs. Maintain an academic tone.
        """
        # This is a placeholder. Real introduction generation is complex.
        logger.info("Generating placeholder for Introduction section.")
        return self._call_llm_for_section(prompt, "Introduction")

    def _generate_methodology(self) -> str:
        """Generates the methodology section. (Placeholder)."""
        prompt = """
        Write the Methodology section for a research paper on LLM spatial reasoning in physical simulations.
        Describe the following aspects:
        1.  **Experimental Setup**: The simulated environment (e.g., "Padres Spatial RL Environment"), key tasks, and objectives within the simulation.
        2.  **LLM Integration**: How the LLM interacts with the simulation (e.g., processing observations, generating actions).
        3.  **Data Collection**: Parameters logged for each experiment (e.g., success, score, distance, task completion, raw interaction logs).
        4.  **Automated Pipeline**: Briefly describe the 24/7 automated system for running experiments, collecting data (BigQuery), and triggering analysis.
        5.  **Performance Metrics**: Define how success, score, and other key metrics are calculated and interpreted.
        6.  **LLM Used**: Specify the type of LLM used (e.g., "a Gemini-based model such as gemini-1.5-flash").
        
        Ensure clarity and provide enough detail for reproducibility. Maintain an academic tone.
        """
        logger.info("Generating placeholder for Methodology section.")
        return self._call_llm_for_section(prompt, "Methodology")

    def _generate_results(self, all_experiments_data: list) -> str:
        """Generates the results section, including some basic statistics from all available data."""
        if not all_experiments_data:
            return "No experimental data available to generate results."

        num_experiments = len(all_experiments_data)
        successful_experiments = sum(1 for exp in all_experiments_data if exp.get('padres_success'))
        success_rate = (successful_experiments / num_experiments) * 100 if num_experiments > 0 else 0
        
        scores = [exp.get('score', 0) for exp in all_experiments_data if exp.get('score') is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        distances = [exp.get('distance', 0) for exp in all_experiments_data if exp.get('distance') is not None]
        avg_distance = sum(distances) / len(distances) if distances else 0
        
        tasks_completed = sum(1 for exp in all_experiments_data if exp.get('task_completed'))

        # Simplified trend for results - could be expanded
        trend_info = "Detailed trend analysis over time is recommended for future work."
        if num_experiments >= 50: # Example: provide some breakdown if many experiments
             # Calculate success rate over segments of the data, e.g., quartiles
            segment_size = num_experiments // 4
            if segment_size > 10: # Only if segments are meaningful
                trends = []
                for i in range(4):
                    segment_data = all_experiments_data[i*segment_size:(i+1)*segment_size]
                    seg_success = sum(1 for exp in segment_data if exp.get('padres_success'))
                    seg_rate = (seg_success / len(segment_data) * 100) if segment_data else 0
                    trends.append(f"Segment {i+1} success: {seg_rate:.2f}%")
                trend_info = "Segmented success rates: " + ", ".join(trends)


        prompt = f"""
        Write the Results section for a research paper on LLM spatial reasoning, based on the full dataset of experiments.

        Quantitative summary of all experiments:
        - Total number of experiments conducted: {num_experiments}
        - Number of successful experiments (padres_success = True): {successful_experiments}
        - Overall success rate: {success_rate:.2f}%
        - Average experiment score: {avg_score:.3f} (if applicable)
        - Average distance metric: {avg_distance:.2f} (if applicable)
        - Total tasks marked as completed: {tasks_completed}
        - Performance trend insights: {trend_info}

        The Results section should:
        1.  Present the key quantitative findings clearly and concisely based on the provided statistics.
        2.  Discuss any observable patterns or trends in performance if discernible from the overall dataset or segmented analysis.
        3.  Avoid interpretation or discussion (save that for the Discussion section). Stick to reporting the outcomes.
        
        Maintain an objective and academic tone. Use precise language.
        """
        return self._call_llm_for_section(prompt, "Results")

    def _generate_related_work(self) -> str:
        """Generates the related work section using Perplexity AI for literature search."""
        logger.info("Initiating literature search for Related Work section using Perplexity AI...")
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

        prompt = f"""
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
        """Generates the discussion section using all available data."""
        avg_score_all_time = 0
        num_experiments = 0
        if all_experiments_data:
            num_experiments = len(all_experiments_data)
            scores = [exp.get('score', 0) for exp in all_experiments_data if exp.get('score') is not None]
            if scores:
                avg_score_all_time = sum(scores) / len(scores)

        prompt = f"""
        Write a Discussion section for the research paper. This section should interpret the findings presented in the Results section, in the broader context of LLM spatial reasoning (using a Gemini-class model).
        Consider data from all {num_experiments} experiments (e.g., average score around {avg_score_all_time:.3f} if available).

        Key aspects to cover:
        1.  **Interpretation of Key Findings**: What do the overall success rates, scores, and any observed trends from the Results section imply about the LLM's current spatial reasoning capabilities in these dynamic simulations?
        2.  **Comparison with Expectations/Related Work**: How do these findings align with or differ from general expectations or findings discussed in the Related Work section?
        3.  **Strengths and Limitations of the Current Study/Methodology**: Discuss the advantages of the automated, continuous evaluation approach with the current LLM. Also, acknowledge limitations (e.g., simulation fidelity, scope of tasks, specific model version nuances, nature of the full dataset analysis vs. time-series).
        4.  **Implications of the Research**: What are the broader implications for developing more spatially aware LLMs (like Gemini)? How can these findings inform future LLM architecture or training?
        5.  **Future Work**: Suggest specific directions for future research. This could include more complex tasks, different simulations, comparative studies with other LLMs, or improvements to the LLM's interaction model, potentially deeper time-series analysis of the collected data.
        
        Maintain a critical and insightful tone. This section should not just restate results but provide deeper meaning.
        """
        logger.info("Generating Discussion section based on all available data.")
        return self._call_llm_for_section(prompt, "Discussion")

    def _generate_conclusion(self, all_experiments_data: list) -> str:
        """Generates the conclusion section based on all available data."""
        num_experiments = len(all_experiments_data) if all_experiments_data else 0
        # Simplified trend for conclusion
        overall_success_rate = 0
        if num_experiments > 0:
            successful_experiments = sum(1 for exp in all_experiments_data if exp.get('padres_success'))
            overall_success_rate = (successful_experiments / num_experiments) * 100

        prompt = f"""
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

    def _combine_paper_sections(self, sections: dict, num_total_experiments: int) -> str:
        """Combines all generated sections into a single Markdown formatted paper string."""
        paper_template = f"""
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
        logger.info("Initiating research paper generation process (using Gemini for text generation), analyzing ALL available data...")
        
        try:
            # 1. Gather all data
            logger.info("Fetching all data from ResearchDataManager...")
            all_experiments_data = self.data_manager.get_all_experiments_for_paper()
            logger.info(f"Data fetched: {len(all_experiments_data)} total experiments.")

            if not all_experiments_data:
                logger.warning("No data available from GCS. Paper generation will be very minimal or skipped.")
                return "[No paper generated due to no data found in GCS.]"

            # 2. Generate paper sections using all data
            paper_sections = {}
            # Pass all_experiments_data to relevant section generators
            paper_sections['abstract'] = self._generate_abstract(all_experiments_data)
            paper_sections['introduction'] = self._generate_introduction() # Introduction is generic
            paper_sections['methodology'] = self._generate_methodology() # Methodology is generic
            paper_sections['results'] = self._generate_results(all_experiments_data)
            paper_sections['related_work'] = self._generate_related_work() # Related work is generic for now
            paper_sections['discussion'] = self._generate_discussion(all_experiments_data)
            paper_sections['conclusion'] = self._generate_conclusion(all_experiments_data)
            
            # 3. Combine into full paper
            full_paper_md = self._combine_paper_sections(paper_sections, len(all_experiments_data))
            
            # 4. Save paper to a Markdown file
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"spatial_ai_research_paper_gemini_{timestamp_str}.md"

            if self.bucket: # If GCS bucket is configured
                blob = self.bucket.blob(f"generated_papers/{base_filename}") # Store in a folder
                blob.upload_from_string(full_paper_md, content_type='text/markdown')
                gcs_path = f"gs://{self.gcs_bucket_name}/generated_papers/{base_filename}"
                logger.info(f"✅ Research paper uploaded successfully to GCS: {gcs_path}")
                return gcs_path
            else: # Fallback to local saving (within container's temp fs)
                # Ensure a directory exists if saving to a subdirectory, e.g., 'generated_papers/'
                # os.makedirs("generated_papers", exist_ok=True)
                # filepath = os.path.join("generated_papers", base_filename)
                filepath = base_filename 
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(full_paper_md)
                logger.info(f"✅ Research paper saved locally (in container) as: {filepath}")
                return filepath # This path is only valid within the container instance
            
        except Exception as e:
            logger.error(f"Critical error during research paper generation: {e}", exc_info=True)
            # Depending on desired behavior, could save a partial paper or just log
            return f"[Failed to generate paper. Error: {e}]"

# Example usage (for local testing)
if __name__ == '__main__':
    from dotenv import load_dotenv
    from bigquery_manager import ResearchDataManager # Assuming it's in the same directory for local test
    from enhanced_padres_perplexity import SimplePadresResearch # Assuming for local test

    load_dotenv()
    logger.info("--- Running AutomatedPaperGenerator local test (with Gemini & GCS) ---")

    gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    gemini_api_key = os.getenv("GEMINI_API_KEY") # Changed from ANTHROPIC_API_KEY
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    # For local test of GCS saving, you might want to set PAPER_OUTPUT_GCS_BUCKET in your .env
    # Ensure your local ADC (gcloud auth application-default login) has permissions to write to that bucket.
    # Example for .env: PAPER_OUTPUT_GCS_BUCKET="your-test-bucket-name"

    if not all([gcp_project_id, gemini_api_key, perplexity_api_key]):
        logger.error("Missing one or more required environment variables for local test: "
                     "GOOGLE_CLOUD_PROJECT, GEMINI_API_KEY, PERPLEXITY_API_KEY")
    else:
        try:
            # Initialize dependencies
            mock_data_manager = ResearchDataManager(project_id=gcp_project_id)
            mock_researcher = SimplePadresResearch() # Assumes it loads keys from env

            # Create a dummy experiment to ensure some data exists for testing
            # In a real scenario, data would already be in BigQuery
            logger.info("Storing a dummy experiment for testing paper generation...")
            dummy_exp_id = f"paper_gen_dummy_exp_gemini_{int(datetime.now().timestamp())}"
            mock_data_manager.store_experiment({
                "experiment_id": dummy_exp_id,
                "timestamp": datetime.utcnow().isoformat(),
                "padres_success": True, "score": 0.8, "distance": 5.0, "task_completed": True,
                "claude_analysis": "[This would be Gemini analysis now]", # Field name in BQ if it was 'claude_analysis'
                "llm_analysis": "Dummy Gemini analysis for paper gen.", # New field used by SimplePadresResearch
                "perplexity_research": "Dummy Perplexity research.",
                "raw_data": {"test": "data"}
            })

            paper_generator_instance = AutomatedPaperGenerator(mock_data_manager, mock_researcher)
            
            logger.info("Starting research paper generation test...")
            generated_file_path = paper_generator_instance.generate_weekly_paper()
            logger.info(f"Paper generation test completed. File/Path: {generated_file_path}")

        except Exception as e:
            logger.error(f"Error during AutomatedPaperGenerator local test: {e}", exc_info=True)
    logger.info("--- AutomatedPaperGenerator local test finished ---") 