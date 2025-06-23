import json  # For logging the full result if needed
import logging
import os
from datetime import datetime

from bigquery_manager import ResearchDataManager  # Needs GOOGLE_CLOUD_PROJECT

# Assuming these files are in the same directory or accessible via PYTHONPATH
from enhanced_padres_perplexity import (  # Needs ANTHROPIC_API_KEY, PERPLEXITY_API_KEY from env
    SimplePadresResearch,
)
from paper_generator import AutomatedPaperGenerator  # Uses researcher and data_manager

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Production24x7Pipeline:
    def __init__(self):
        """Initializes all components of the research pipeline."""
        logger.info("Initializing Production24x7Pipeline...")

        # Get GOOGLE_CLOUD_PROJECT for DataManager
        self.gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.gcp_project_id:
            logger.error("CRITICAL: GOOGLE_CLOUD_PROJECT environment variable not set.")
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable is required for BigQuery operations."
            )

        # ANTHROPIC_API_KEY and PERPLEXITY_API_KEY are expected to be set as environment variables.
        # SimplePadresResearch is assumed to use os.getenv() internally for these keys.
        # If they are not set, SimplePadresResearch should handle that (e.g., raise error or log warning).
        # We check for their presence here to provide an early warning if expected.
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning(
                "ANTHROPIC_API_KEY environment variable not set. Claude functionality may be affected."
            )
        if not os.getenv("PERPLEXITY_API_KEY"):
            logger.warning(
                "PERPLEXITY_API_KEY environment variable not set. Perplexity functionality may be affected."
            )

        try:
            self.researcher = SimplePadresResearch()
            logger.info("SimplePadresResearch initialized.")

            self.data_manager = ResearchDataManager(project_id=self.gcp_project_id)
            logger.info("ResearchDataManager initialized.")

            self.paper_generator = AutomatedPaperGenerator(
                data_manager=self.data_manager, researcher=self.researcher
            )
            logger.info("AutomatedPaperGenerator initialized.")

            logger.info("Production24x7Pipeline initialized successfully.")

        except Exception as e:
            logger.error(
                f"Error during Production24x7Pipeline initialization: {e}",
                exc_info=True,
            )
            # Depending on the severity, you might want to re-raise or handle gracefully
            raise RuntimeError(f"Failed to initialize core components: {e}")

    def run_experiment_batch(self, batch_size: int = 10):
        """Runs a batch of experiments and stores results in BigQuery."""
        logger.info(f"Starting experiment batch of size {batch_size}.")
        batch_results_summary = []

        # 1. Perform Perplexity search once for the entire batch
        master_research_query = (
            "Latest spatial reasoning AI research 2024 2025 LLM physics simulation"
        )
        logger.info(
            f"Performing batch-level Perplexity search for query: {master_research_query}"
        )
        batch_perplexity_result = self.researcher.search_perplexity(
            master_research_query
        )
        if "Error:" in str(
            batch_perplexity_result
        ):  # Check if perplexity call resulted in an error string
            logger.warning(
                f"Batch-level Perplexity search failed: {batch_perplexity_result}. This will be noted in stored data."
            )

        for i in range(batch_size):
            experiment_number = i + 1
            logger.info(f"🧪 Running Experiment {experiment_number}/{batch_size}")
            try:
                # 2. Run individual experiment (Padres -> Gemini analysis)
                experiment_core_data = self.researcher.run_research_experiment()

                # Prepare the sub-dictionary for the 'raw_data' field in BigQuery
                # Ensure all potentially complex objects are definitely strings here
                padres_data_for_raw = experiment_core_data.get(
                    "padres_api_response", {}
                )
                try:
                    # If padres_data_for_raw is already a dict and JSON serializable, this is fine
                    # If not, json.dumps will make it a string.
                    padres_data_str = json.dumps(padres_data_for_raw, default=str)
                except TypeError:
                    padres_data_str = str(padres_data_for_raw)  # Fallback

                gemini_analysis_str = str(experiment_core_data.get("llm_analysis", ""))
                perplexity_research_str = (
                    str(batch_perplexity_result) if batch_perplexity_result else ""
                )
                master_query_str = str(master_research_query)

                raw_data_payload = {
                    "padres_api_response": padres_data_str,  # Store as string
                    "gemini_analysis_full": gemini_analysis_str,
                    "perplexity_research_full": perplexity_research_str,
                    "perplexity_query_used": master_query_str,
                }

                # This is the comprehensive data structure to be stored for this experiment
                final_data_for_bq = {
                    "experiment_id": experiment_core_data.get("experiment_id"),
                    "timestamp": experiment_core_data.get("timestamp"),
                    "padres_success": experiment_core_data.get("padres_success"),
                    "llm_analysis": gemini_analysis_str,  # For the main BQ column (Gemini's output)
                    "perplexity_research": perplexity_research_str,  # For the main BQ column
                    "score": experiment_core_data.get("score"),
                    "distance": experiment_core_data.get("distance"),
                    "task_completed": experiment_core_data.get("task_completed"),
                    "raw_data": raw_data_payload,  # This is the dictionary for the JSON field
                }

                # Log the complete data that will be attempted for BigQuery
                logger.info(
                    f"Complete data for BigQuery (experiment_id: {final_data_for_bq.get('experiment_id')}): {json.dumps(final_data_for_bq, default=str)}"
                )

                # 4. Store the fully combined result in BigQuery
                self.data_manager.store_experiment(final_data_for_bq)

                batch_results_summary.append(
                    {
                        "experiment_number": experiment_number,
                        "status": "success",
                        "experiment_id": experiment_core_data.get(
                            "experiment_id", "N/A"
                        ),
                    }
                )
                logger.info(
                    f"Experiment {experiment_number} processed and data storage initiated."
                )

            except Exception as e:
                logger.error(
                    f"❌ Experiment {experiment_number} failed during pipeline processing: {e}",
                    exc_info=True,
                )
                batch_results_summary.append(
                    {
                        "experiment_number": experiment_number,
                        "status": "error",
                        "error_message": str(e),
                    }
                )

        logger.info(
            f"Experiment batch completed. Processed {len(batch_results_summary)} experiments."
        )
        return {
            "message": f"Experiment batch of {batch_size} processed.",
            "processed_count": len(batch_results_summary),
            "details": batch_results_summary,
        }

    def analyze_recent_data(self):
        """Analyzes experimental data from the last 7 days."""
        logger.info("📊 Initiating analysis of recent experimental data (last 7 days).")
        try:
            recent_experiments = self.data_manager.get_recent_experiments(days=7)
            trends = (
                self.data_manager.get_success_rate_trends()
            )  # Assumes this looks at a relevant period, e.g., 30 days

            if not recent_experiments:
                logger.info("No experiments found in the last 7 days to analyze.")
                return {
                    "message": "No experiments found in the last 7 days.",
                    "analysis": "Not applicable.",
                    "experiment_count": 0,
                    "trends": trends,  # Still might be useful to return overall trends
                }

            analysis_prompt = """
            Analyze the following AI research data from spatial reasoning experiments conducted over the last 7 days.
            Focus on identifying performance patterns, potential research insights, areas for further investigation, and any novel findings.
            The experiments involve an AI interacting with physical simulations, with primary analysis by a Gemini model and contextual research by Perplexity.

            Summary of data for the last 7 days:
            - Total Experiments Conducted: {len(recent_experiments)}

            Recent Performance Trends (e.g., daily success rates, scores from the last 5 reporting periods):
            {str(trends[:5]) if trends else 'No trend data available.'}

            Please generate a concise research summary based on this information, suitable for inclusion in a progress report or preliminary findings section of a paper.
            Highlight key takeaways and suggest next steps if any patterns are discernible.
            """

            # call_claude is assumed to exist in SimplePadresResearch
            analysis_text = self.researcher.call_claude(analysis_prompt)
            logger.info("Data analysis generated successfully via Claude.")

            return {
                "message": "Data analysis completed.",
                "analysis": analysis_text,
                "experiment_count_last_7_days": len(recent_experiments),
                "trends_summary": trends,
            }
        except Exception as e:
            logger.error(f"Failed to analyze recent data: {e}", exc_info=True)
            return {"message": "Failed to analyze recent data.", "error": str(e)}

    def generate_paper(self):
        """Generates a research paper based on accumulated data."""
        logger.info("📝 Initiating research paper generation.")
        try:
            # generate_weekly_paper is assumed to exist in AutomatedPaperGenerator
            # and to save the file, returning its name or path.
            paper_filename = self.paper_generator.generate_weekly_paper()
            logger.info(f"Research paper generated successfully: {paper_filename}")
            return {
                "message": "Research paper generated successfully.",
                "filename": paper_filename,
                # Consider returning a GCS URL if uploaded to Cloud Storage by paper_generator
            }
        except Exception as e:
            logger.error(f"Failed to generate research paper: {e}", exc_info=True)
            return {"message": "Failed to generate research paper.", "error": str(e)}


# Example of local usage (not run when imported by FastAPI app)
if __name__ == "__main__":
    # This block is for local testing and demonstration.
    # Ensure .env file is present with GOOGLE_CLOUD_PROJECT, ANTHROPIC_API_KEY, PERPLEXITY_API_KEY
    from dotenv import load_dotenv

    load_dotenv()

    logger.info(
        "Running Production24x7Pipeline locally (with batch Perplexity call)..."
    )
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        print("Error: GOOGLE_CLOUD_PROJECT not set.")
    elif not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set.")
    elif not os.getenv("PERPLEXITY_API_KEY"):
        print("Error: PERPLEXITY_API_KEY not set.")
    elif not os.getenv("PADRES_API_URL"):
        print("Error: PADRES_API_URL not set.")
    else:
        pipeline = Production24x7Pipeline()
        logger.info("\n--- Testing run_experiment_batch (batch_size=2) ---")
        experiment_results = pipeline.run_experiment_batch(batch_size=2)
        logger.info(
            f"Experiment Batch Results: {json.dumps(experiment_results, indent=2)}"
        )

        # logger.info("\n--- Testing analyze_recent_data ---")
        # analysis_results = pipeline.analyze_recent_data()
        # logger.info(f"Analysis Results: {json.dumps(analysis_results, indent=2)}")

        # logger.info("\n--- Testing generate_paper ---")
        # paper_results = pipeline.generate_paper()
        # logger.info(f"Paper Generation Results: {json.dumps(paper_results, indent=2)}")

        logger.info("Local testing finished. Uncomment other tests to run them.")
