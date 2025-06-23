# research_automation/research_assistant.py

# This service will likely be a Cloud Function, triggered by Pub/Sub messages (e.g., experiment completion).

# from ..backend_services.mcp_manager import MCPManager # Adjust import based on final structure
# from google.cloud import bigquery, pubsub_v1 # Will be needed later

import asyncio  # For main test block example
import datetime
import json  # For main test block example
import logging
import uuid  # Added for generating unique IDs
from pathlib import Path  # Added for loading config in main
from typing import Any, Dict, List, Optional

# Ensure MCPManager can be imported correctly
# This assumes the script is run from the project root or PYTHONPATH is set up
try:
    from backend_services.mcp_manager import MCPManager
except ImportError:
    # Fallback for cases where the script might be run directly and path isn't set
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from backend_services.mcp_manager import MCPManager


class ResearchAssistant:
    """
    Analyzes experiment results and generates insights or recommendations
    using the MCP framework to interact with LLMs, BigQuery, etc.
    """

    def __init__(self, mcp_manager, config=None):
        """
        Initialize the ResearchAssistant.

        Args:
            mcp_manager: An instance of MCPManager
            config (dict, optional): Configuration for the ResearchAssistant
        """
        self.mcp_manager = mcp_manager
        self.config = config or {}
        self.default_llm_mcp = self.config.get(
            "default_llm_mcp_server_name", "gemini_pro_main"
        )
        self.default_bigquery_mcp = self.config.get(
            "default_bigquery_mcp_server_name", "bq_main"
        )
        self.default_firestore_mcp = self.config.get(
            "default_firestore_mcp_server_name", "firestore_main"
        )
        self.default_perplexity_mcp = self.config.get(
            "default_perplexity_mcp_server_name", "perplexity_main"
        )

        # Configuration for data sources and sinks
        self.bq_project_id = self.config.get("bigquery_project_id")
        self.bq_dataset_id = self.config.get("bigquery_dataset_id")
        self.bq_trial_results_table = self.config.get("bigquery_trial_results_table")
        self.firestore_insights_collection = self.config.get(
            "firestore_insights_collection", "research_assistant_outputs"
        )

        self.low_score_threshold = self.config.get(
            "low_score_threshold", 0.7
        )  # Example specific config

        logging.info(
            f"ResearchAssistant initialized with LLM: {self.default_llm_mcp}, BQ: {self.default_bigquery_mcp}, Firestore: {self.default_firestore_mcp}"
        )
        if not all(
            [self.bq_project_id, self.bq_dataset_id, self.bq_trial_results_table]
        ):
            logging.warning(
                "BigQuery connection details (project, dataset, table) are not fully configured."
            )

    async def analyze_experimental_results(
        self, experiment_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Analyzes results from one or more experiments to generate insights.

        Args:
            experiment_ids: List of experiment IDs to analyze

        Returns:
            Dict containing insights and recommendations
        """
        logging.info(f"Analyzing results for experiments: {experiment_ids}")

        if not experiment_ids:
            logging.warning("No experiment IDs provided for analysis.")
            return {"status": "error", "error": "No experiment IDs provided."}

        try:
            results_data = await self._fetch_experiment_results(experiment_ids)
            if not results_data or results_data.get("error"):
                logging.error(
                    f"Failed to fetch experiment results: {results_data.get('error', 'No data returned')}"
                )
                return {
                    "status": "error",
                    "error": f"Failed to fetch experiment results: {results_data.get('error', 'No data returned')}",
                }

            insights = await self._generate_insights(results_data, experiment_ids)
            if not insights or insights.get("error"):
                logging.error(
                    f"Failed to generate insights: {insights.get('error', 'No insights returned')}"
                )
                return {
                    "status": "error",
                    "error": f"Failed to generate insights: {insights.get('error', 'No insights returned')}",
                }

            recommendations = await self._generate_recommendations(
                insights, experiment_ids
            )
            if not recommendations or recommendations.get("error"):
                logging.error(
                    f"Failed to generate recommendations: {recommendations.get('error', 'No recommendations returned')}"
                )
                # Continue to store insights even if recommendations fail
                pass  # Or return error: return {"status": "error", "error": f"Failed to generate recommendations: {recommendations.get('error', 'No recommendations returned')}"}

            storage_success = await self._store_insights(
                insights, recommendations, experiment_ids
            )
            if not storage_success or storage_success.get("error"):
                logging.error(
                    f"Failed to store insights: {storage_success.get('error', 'Storage failed')}"
                )
                # Non-fatal, proceed with returning generated content

            return {
                "status": "success",
                "insights": insights.get("insights_generated", []),
                "recommendations": recommendations.get("recommendations_generated", []),
                "storage_info": storage_success,
            }
        except Exception as e:
            logging.exception(
                f"Critical error in analyze_experimental_results for {experiment_ids}: {e}"
            )
            return {"status": "error", "error": str(e)}

    async def _fetch_experiment_results(
        self, experiment_ids: List[str]
    ) -> Dict[str, Any]:
        """Fetch experiment results from BigQuery via MCP."""
        logging.info(f"Fetching results for experiments: {experiment_ids}")
        if not all(
            [self.bq_project_id, self.bq_dataset_id, self.bq_trial_results_table]
        ):
            return {
                "error": "BigQuery connection details (project, dataset, table) are not configured."
            }

        experiment_ids_quoted = [""{eid}'" for eid in experiment_ids]
        query = """
            SELECT *
            FROM `{self.bq_project_id}.{self.bq_dataset_id}.{self.bq_trial_results_table}`
            WHERE experiment_id IN ({','.join(experiment_ids_quoted)})
        """

        try:
            bq_response = await self.mcp_manager.call_tool(
                server_name=self.default_bigquery_mcp,
                tool_name="run_query",
                parameters={
                    "query": query,
                    "to_dataframe": False,
                },  # Assuming returns list of dicts
            )

            if bq_response and "error" not in bq_response:
                # The BigQueryMCPServer mock returns a list of rows directly if successful
                # For a real BQ server, it might be a dict like {"rows": [...]}
                # Adjust based on actual BQ MCP server's response structure
                if isinstance(bq_response, list):  # Mock server direct list
                    return {"experiment_data": bq_response}
                elif (
                    isinstance(bq_response, dict) and "rows" in bq_response
                ):  # More standard BQ response
                    return {"experiment_data": bq_response["rows"]}
                else:  # Unexpected format
                    logging.warning(
                        f"Unexpected BigQuery response format: {bq_response}"
                    )
                    return {
                        "experiment_data": []
                    }  # Return empty if format is weird but no explicit error
            else:
                logging.error(
                    f"Error from BigQuery MCP: {bq_response.get('error', 'Unknown error')}"
                )
                return {
                    "error": f"BigQuery MCP error: {bq_response.get('error', 'Unknown error')}"
                }
        except Exception as e:
            logging.exception(
                f"Exception fetching experiment results from BigQuery: {e}"
            )
            return {"error": f"Exception querying BigQuery: {str(e)}"}

    async def _generate_insights(
        self, results_data: Dict[str, Any], experiment_ids: List[str]
    ) -> Dict[str, Any]:
        """Generate insights using LLM via MCP."""
        logging.info(f"Generating insights for experiments: {experiment_ids}")

        prompt = """
        Analyze the following experimental results and generate insights.
        Experiment IDs: {', '.join(experiment_ids)}
        Data:
        {json.dumps(results_data.get("experiment_data", []), indent=2)}

        Provide insights in a structured JSON format, like this:
        {{
            "insights_generated": [
                {{
                    "experiment_id": "specific_experiment_id_or_all",
                    "insight_type": "performance_summary | anomaly_detection | pattern_recognition | improvement_suggestion",
                    "description": "Detailed finding here.",
                    "severity": "low | medium | high (optional)",
                    "confidence": "low | medium | high (optional)",
                    "supporting_data_points": [ {{ "metric": "...", "value": "..." }} ] (optional)
                }}
            ]
        }}
        Focus on key performance indicators, anomalies, and potential areas for improvement.
        If data is sparse or missing, note that in the insights.
        """

        try:
            llm_response = await self.mcp_manager.call_tool(
                server_name=self.default_llm_mcp,
                tool_name="generate_text",
                parameters={
                    "prompt": prompt,
                    "max_tokens": 1024,
                },  # Adjust max_tokens as needed
            )

            if llm_response and not (
                isinstance(llm_response, dict) and llm_response.get("error")
            ):
                # Assuming LLM returns a JSON string that needs parsing
                try:
                    # The Claude mock server returns a string, not a dict with "text"
                    # The Gemini mock server returns a dict {"text": "..."}
                    # Need to standardize or handle based on actual server response
                    if isinstance(llm_response, dict) and "text" in llm_response:
                        insights_content = llm_response["text"]
                    elif isinstance(
                        llm_response, str
                    ):  # Direct string response (like Claude mock)
                        insights_content = llm_response
                    else:
                        logging.error(
                            f"Unexpected LLM response type for insights: {type(llm_response)}"
                        )
                        return {"error": "Unexpected LLM response type for insights."}

                    # LLMs might sometimes return markdown ```json ... ```
                    if insights_content.strip().startswith("```json"):
                        insights_content = insights_content.strip()[7:-3].strip()
                    elif insights_content.strip().startswith("```"):
                        insights_content = insights_content.strip()[3:-3].strip()

                    parsed_insights = json.loads(insights_content)
                    return parsed_insights  # Expected: {"insights_generated": [...]}
                except json.JSONDecodeError as jde:
                    logging.error(
                        f"Failed to parse JSON from LLM for insights: {jde}. Response was: {llm_response[:500]}"
                    )
                    # Fallback: return the raw text if JSON parsing fails, wrapped in a standard structure
                    return {
                        "insights_generated": [
                            {
                                "insight_type": "raw_text",
                                "description": str(llm_response),
                            }
                        ],
                        "warning": "LLM response was not valid JSON.",
                    }
                except Exception as e:
                    logging.exception(
                        f"Error processing LLM response for insights: {e}"
                    )
                    return {
                        "error": f"Error processing LLM response for insights: {str(e)}"
                    }

            else:
                err_msg = (
                    llm_response.get("error", "Unknown LLM error")
                    if isinstance(llm_response, dict)
                    else "Unknown LLM error"
                )
                logging.error(f"Error from LLM MCP for insights: {err_msg}")
                return {"error": f"LLM MCP error for insights: {err_msg}"}
        except Exception as e:
            logging.exception(f"Exception generating insights via LLM: {e}")
            return {"error": f"Exception calling LLM for insights: {str(e)}"}

    async def _generate_recommendations(
        self, insights_data: Dict[str, Any], experiment_ids: List[str]
    ) -> Dict[str, Any]:
        """Generate recommendations for new experiments via MCP based on insights."""
        logging.info(
            f"Generating recommendations based on insights for experiments: {experiment_ids}"
        )

        prompt = """
        Based on the following insights from experiments ({', '.join(experiment_ids)}):
        {json.dumps(insights_data.get("insights_generated", []), indent=2)}

        Suggest recommendations for future experiments or modifications to existing ones.
        Provide recommendations in a structured JSON format, like this:
        {{
            "recommendations_generated": [
                {{
                    "recommendation_type": "new_experiment_idea | parameter_tuning | prompt_refinement | data_augmentation",
                    "description": "Detailed recommendation here.",
                    "based_on_insight_ids": ["id_of_insight_if_applicable"], (optional)
                    "suggested_parameters": {{ "param1": "value1" }} (optional),
                    "expected_impact": "Briefly describe the expected outcome." (optional)
                }}
            ]
        }}
        If no specific recommendations can be made, state that.
        """

        try:
            llm_response = await self.mcp_manager.call_tool(
                server_name=self.default_llm_mcp,
                tool_name="generate_text",
                parameters={"prompt": prompt, "max_tokens": 1024},  # Adjust as needed
            )

            if llm_response and not (
                isinstance(llm_response, dict) and llm_response.get("error")
            ):
                try:
                    if isinstance(llm_response, dict) and "text" in llm_response:
                        recommendations_content = llm_response["text"]
                    elif isinstance(llm_response, str):
                        recommendations_content = llm_response
                    else:
                        logging.error(
                            f"Unexpected LLM response type for recommendations: {type(llm_response)}"
                        )
                        return {
                            "error": "Unexpected LLM response type for recommendations."
                        }

                    if recommendations_content.strip().startswith("```json"):
                        recommendations_content = recommendations_content.strip()[
                            7:-3
                        ].strip()
                    elif recommendations_content.strip().startswith("```"):
                        recommendations_content = recommendations_content.strip()[
                            3:-3
                        ].strip()

                    parsed_recommendations = json.loads(recommendations_content)
                    return parsed_recommendations  # Expected: {"recommendations_generated": [...]}
                except json.JSONDecodeError as jde:
                    logging.error(
                        f"Failed to parse JSON from LLM for recommendations: {jde}. Response was: {llm_response[:500]}"
                    )
                    return {
                        "recommendations_generated": [
                            {
                                "recommendation_type": "raw_text",
                                "description": str(llm_response),
                            }
                        ],
                        "warning": "LLM response was not valid JSON.",
                    }
                except Exception as e:
                    logging.exception(
                        f"Error processing LLM response for recommendations: {e}"
                    )
                    return {
                        "error": f"Error processing LLM response for recommendations: {str(e)}"
                    }
            else:
                err_msg = (
                    llm_response.get("error", "Unknown LLM error")
                    if isinstance(llm_response, dict)
                    else "Unknown LLM error"
                )
                logging.error(f"Error from LLM MCP for recommendations: {err_msg}")
                return {"error": f"LLM MCP error for recommendations: {err_msg}"}
        except Exception as e:
            logging.exception(f"Exception generating recommendations via LLM: {e}")
            return {"error": f"Exception calling LLM for recommendations: {str(e)}"}

    async def _store_insights(
        self,
        insights_data: Optional[Dict[str, Any]],
        recommendations_data: Optional[Dict[str, Any]],
        experiment_ids: List[str],
    ) -> Dict[str, Any]:
        """Store insights and recommendations in Firestore via MCP."""

        # Ensure insights_data and recommendations_data are dicts, even if empty
        insights_data = insights_data if insights_data else {}
        recommendations_data = recommendations_data if recommendations_data else {}

        # Use the first experiment ID for naming or a generic ID if list is empty
        primary_experiment_id = (
            experiment_ids[0] if experiment_ids else "batch_analysis"
        )
        # Generate a unique ID for this set of insights/recommendations.
        # Could also be tied to a specific analysis run ID if that exists.
        insight_doc_id = f"{primary_experiment_id}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        logging.info(
            f"Storing insights and recommendations in Firestore under doc ID: {insight_doc_id} in collection {self.firestore_insights_collection}"
        )

        data_to_store = {
            "assistant_run_id": insight_doc_id,
            "analyzed_experiment_ids": experiment_ids,
            "insights": insights_data.get("insights_generated", []),
            "recommendations": recommendations_data.get(
                "recommendations_generated", []
            ),
            "generation_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "llm_server_used": self.default_llm_mcp,
            "errors_during_generation": {  # Log any non-fatal errors from prior steps
                "insights_error": insights_data.get("error"),
                "insights_warning": insights_data.get("warning"),
                "recommendations_error": recommendations_data.get("error"),
                "recommendations_warning": recommendations_data.get("warning"),
            },
        }

        try:
            fs_response = await self.mcp_manager.call_tool(
                server_name=self.default_firestore_mcp,
                tool_name="set_document",
                parameters={
                    "collection_id": self.firestore_insights_collection,
                    "document_id": insight_doc_id,
                    "data": data_to_store,
                },
            )

            if fs_response and fs_response.get("status", "").startswith("success"):
                logging.info(
                    f"Successfully stored insights in Firestore: {insight_doc_id}"
                )
                return {
                    "status": "success",
                    "document_id": insight_doc_id,
                    "collection_id": self.firestore_insights_collection,
                    "message": "Insights stored successfully.",
                }
            else:
                error_msg = (
                    fs_response.get("error", "Unknown Firestore error")
                    if isinstance(fs_response, dict)
                    else "Unknown Firestore error"
                )
                logging.error(f"Error storing insights in Firestore: {error_msg}")
                return {"error": f"Firestore MCP error: {error_msg}"}
        except Exception as e:
            logging.exception(f"Exception storing insights in Firestore: {e}")
            return {"error": f"Exception storing insights: {str(e)}"}


# Cloud Function entry point (example)
# This is how it might be structured if deployed as a Google Cloud Function
# triggered by a Pub/Sub message about experiment completion.
#
# def gcf_research_assistant_trigger(event, context):
#     """
#     Background Cloud Function to be triggered by Pub/Sub.
#     Args:
#          event (dict):  The dictionary with data specific to this type of event.
#                         The `data` field contains the Pub/Sub message data, base64-encoded.
#          context (google.cloud.functions.Context): Metadata of triggering event.
#     """
#     import base64
#     import json
#
#     print(f"GCF Research Assistant Triggered. Event ID: {context.event_id}, Type: {context.event_type}")
#
#     if 'data' in event:
#         pubsub_message_str = base64.b64decode(event['data']).decode('utf-8')
#         pubsub_message = json.loads(pubsub_message_str)
#         print(f"Pub/Sub Message: {pubsub_message}")
#
#         experiment_id = pubsub_message.get("experiment_id")
#         if experiment_id:
#             # Initialize dependencies (MCPManager, config)
#             # This is a simplified example; in a real GCF, you'd handle config and client instantiation robustly.
#             # from backend_services.mcp_manager import MCPManager # Ensure correct path or packaging
#             class MockMCPManager: # Replace with actual MCPManager setup
#                 def call_tool(self, sn, tn, p): return {}
#
#             config = { # Load from env vars or bundled config file
#                 "bigquery_table": "your_project.your_dataset.trial_results",
#                 "bigquery_mcp_server": "bigquery_main",
#                 "perplexity_mcp_server": "perplexity_main",
#                 "llm_mcp_server": "claude_main"
#             }
#             assistant = ResearchAssistant(mcp_manager=MockMCPManager(), config=config)
#             assistant.process_experiment_results(experiment_id, event_data=pubsub_message)
#         else:
#             print("Error: 'experiment_id' not found in Pub/Sub message.")
#     else:
#         print("Error: No data in event.")


# Example Usage (conceptual, direct call):
async def main_test():
    print("\nTesting ResearchAssistant with live MCPManager...")

    # Determine project root to reliably load config
    # Assumes this script is in research_automation/
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "pipeline_config.json"

    if not config_path.exists():
        print(f"ERROR: Configuration file not found at {config_path}")
        print("Please ensure 'config/pipeline_config.json' exists in the project root.")
        return

    with open(config_path, "r") as f:
        pipeline_config = json.load(f)

    print(f"Loaded pipeline config from {config_path}")
    # Ensure mock_mode is true for all relevant MCPs for this test if no real credentials
    print(f"Pipeline mock_mode (global): {pipeline_config.get('mock_mode')}")
    print(
        f"Claude mock_mode: {pipeline_config.get('mcp_server_configurations', {}).get('claude_main', {}).get('config', {}).get('mock_mode')}"
    )
    print(
        f"BigQuery mock_mode: {pipeline_config.get('mcp_server_configurations', {}).get('bq_main', {}).get('config', {}).get('mock_mode')}"
    )
    print(
        f"Firestore mock_mode: {pipeline_config.get('mcp_server_configurations', {}).get('firestore_main', {}).get('config', {}).get('mock_mode')}"
    )

    # Initialize MCP Manager - the core of our system
    # It will use the "mcp_server_configurations" from pipeline_config
    mcp_manager = MCPManager(config=pipeline_config)

    # Initialize ResearchAssistant
    ra_config = pipeline_config.get("research_assistant_config", {})
    if not ra_config:
        print(
            "Warning: 'research_assistant_config' not found in pipeline_config.json. RA may not work correctly."
        )

    assistant = ResearchAssistant(mcp_manager=mcp_manager, config=ra_config)

    # Example experiment ID present in Firestore mock data in pipeline_config.json
    # This is mostly for the orchestrator; RA fetches from BQ.
    # Let's use a conceptual experiment ID that BQ mock might respond to.
    # The current BigQueryMCPServer mock always returns dummy data for ANY query.
    test_experiment_id_completed = "exp_spatial_reasoning_001_completed"

    print(f"\nProcessing completed experiment event: {test_experiment_id_completed}")
    # Call the main method of ResearchAssistant
    assistant_run_result = await assistant.analyze_experimental_results(
        [test_experiment_id_completed]
    )
    print(
        f"Research Assistant run result:\n{json.dumps(assistant_run_result, indent=2)}"
    )

    # Test with an experiment ID that might yield no data from BQ mock (or empty list)
    test_experiment_id_no_data = "exp_002_no_data"
    print(
        f"\nProcessing event for experiment expected to have no data: {test_experiment_id_no_data}"
    )
    no_data_run_result = await assistant.analyze_experimental_results(
        [test_experiment_id_no_data]
    )
    print(
        f"Research Assistant run result (no data):\n{json.dumps(no_data_run_result, indent=2)}"
    )

    # Test with empty list of experiment IDs
    print("\nProcessing event for empty list of experiment IDs")
    empty_list_run_result = await assistant.analyze_experimental_results([])
    print(
        f"Research Assistant run result (empty list):\n{json.dumps(empty_list_run_result, indent=2)}"
    )

    # Clean up MCPManager (if it holds resources like HTTP clients)
    # This depends on MCPManager's design; adding a close method might be good practice
    # For now, individual MCP servers (like Padres) have close methods.
    # We can iterate and close them if they have a 'close_client' method.
    for server_instance in mcp_manager.mcp_servers.values():
        if hasattr(server_instance, "close_client") and asyncio.iscoroutinefunction(
            server_instance.close_client
        ):
            await server_instance.close_client()
        elif hasattr(server_instance, "close_client"):  # if not async
            server_instance.close_client()

    print("\nResearchAssistant with MCPManager conceptual test complete.")


if __name__ == "__main__":
    # Setup basic logging for the test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )
    asyncio.run(main_test())
