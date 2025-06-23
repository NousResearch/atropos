# research_automation/analysis_engine.py
import asyncio  # For main test block
import datetime
import json  # For main test block

# import numpy as np # Not used in current mocks, but could be for real analysis
# from ..backend_services.mcp_manager import MCPManager # Adjust import for actual structure
# from google.cloud import bigquery # Used by the BQ MCP server, not directly here
import logging
import uuid  # Added for generating unique IDs
from pathlib import Path  # Added for loading config in main
from typing import Any, Dict, List, Optional

import pandas as pd

# Ensure MCPManager can be imported correctly
try:
    from backend_services.mcp_manager import MCPManager
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from backend_services.mcp_manager import MCPManager


class AnalysisEngine:
    """
    Performs deeper, potentially scheduled, statistical analysis on data aggregated in BigQuery,
    fetched via MCPManager. Uses Pandas, NumPy, etc.
    Generates summary statistics, identifies trends, and performs comparative analyses.
    """

    def __init__(self, mcp_manager, config=None):
        """
        Initializes the AnalysisEngine.
        Args:
            mcp_manager (MCPManager): An instance of MCPManager.
            config (dict, optional): Engine-specific configurations, e.g.,
                                     default_bigquery_mcp_server_name,
                                     bigquery_project_id (for context, though MCP handles it),
                                     bigquery_dataset_id (for context),
                                     bigquery_analysis_results_table (for storing outputs).
        """
        if mcp_manager is None:
            raise ValueError("MCPManager instance is required for AnalysisEngine.")
        self.mcp_manager = mcp_manager
        self.config = config or {}
        self.default_bigquery_mcp = self.config.get(
            "default_bigquery_mcp_server_name", "bq_main"
        )
        # mock_mode is determined by the individual MCP servers now
        # self.mock_mode = self.config.get("mock_mode", False)

        self.bq_project_id = self.config.get(
            "bigquery_project_id"
        )  # For context if needed, MCP handles actual project
        self.bq_dataset_id = self.config.get("bigquery_dataset_id")  # For context
        self.bq_analysis_results_table = self.config.get(
            "bigquery_analysis_results_table", "analysis_outputs"
        )

        logging.info(
            f"AnalysisEngine initialized with BigQuery MCP: {self.default_bigquery_mcp}"
        )
        if not self.bq_analysis_results_table:
            logging.warning("BigQuery analysis results table ID is not configured.")

    async def perform_deep_analysis(
        self, analysis_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Performs deep analysis on research data based on the provided configuration.

        Args:
            analysis_config: Configuration for the analysis task

        Returns:
            Dict containing analysis results
        """
        analysis_name = analysis_config.get("name", "Unnamed Analysis")
        logging.info(f"Performing deep analysis: {analysis_name}")

        # Mock mode behavior is now handled by the underlying MCP servers.
        # The `_run_analysis_query`, `_process_analysis_data`, `_store_analysis_results`
        # methods will rely on MCP calls which might be to mock servers.
        # The previous top-level if self.mock_mode block is removed.

        try:
            query_result_data = await self._run_analysis_query(analysis_config)
            if not query_result_data or query_result_data.get("error"):
                logging.error(
                    f"Failed to run analysis query: {query_result_data.get('error', 'No data returned')}"
                )
                return {
                    "status": "error",
                    "analysis_name": analysis_name,
                    "error": f"Failed to run analysis query: {query_result_data.get('error', 'No data returned')}",
                }

            # _process_analysis_data expects a list of dicts (rows)
            # The BQ MCP mock returns a list of dicts directly. A real one might be {"rows": [...]}
            data_for_processing = []
            if isinstance(query_result_data, list):
                data_for_processing = query_result_data
            elif isinstance(query_result_data, dict) and "rows" in query_result_data:
                data_for_processing = query_result_data["rows"]
            else:
                logging.warning(
                    f"Unexpected format from _run_analysis_query for {analysis_name}: {type(query_result_data)}"
                )
                # Proceed with empty data if format is unrecognized but no explicit error

            processed_results = await self._process_analysis_data(
                data_for_processing, analysis_config
            )
            if not processed_results or processed_results.get("error"):
                logging.error(
                    f"Failed to process analysis data: {processed_results.get('error', 'Processing failed')}"
                )
                return {
                    "status": "error",
                    "analysis_name": analysis_name,
                    "error": f"Failed to process analysis data: {processed_results.get('error', 'Processing failed')}",
                }

            storage_info = await self._store_analysis_results(
                processed_results, analysis_config
            )
            if not storage_info or storage_info.get("error"):
                logging.error(
                    f"Failed to store analysis results: {storage_info.get('error', 'Storage failed')}"
                )
                # Non-fatal for returning results, but log it.

            return {
                "status": "success",
                "analysis_name": analysis_name,
                "summary": "Analysis "{analysis_name}' completed successfully.",
                "results": processed_results,
                "storage_info": storage_info,
            }
        except Exception as e:
            logging.exception(
                "Critical error in perform_deep_analysis for "{analysis_name}': {e}"
            )
            return {"status": "error", "analysis_name": analysis_name, "error": str(e)}

    async def _run_analysis_query(self, analysis_config: Dict[str, Any]) -> Any:
        """Run the analysis query in BigQuery via MCP. Returns raw response from MCP call."""
        query = analysis_config.get("query", "")
        if not query:
            logging.error("No query provided in analysis_config.")
            return {"error": "No query provided in analysis_config."}

        logging.info(
            "Running analysis query for "{analysis_config.get('name')}': {query[:100]}..."
        )

        try:
            # Assuming BQ MCP's run_query returns list of dicts (rows) or {"rows": [...]} or {"error": ...}
            # The mock BigQueryMCPServer directly returns a list of rows or an error dict.
            bq_response = await self.mcp_manager.call_tool(
                server_name=self.default_bigquery_mcp,
                tool_name="run_query",
                parameters={
                    "query": query,
                    "to_dataframe": False,
                },  # Fetch as list of dicts
            )

            if isinstance(bq_response, dict) and "error" in bq_response:
                logging.error(
                    f"Error from BigQuery MCP during query execution: {bq_response['error']}"
                )
                return {"error": f"BigQuery MCP error: {bq_response['error']}"}

            # If no error, bq_response is expected to be the data (e.g., list of rows for mock)
            # or a dict containing rows for a more standard server.
            return bq_response  # Return the direct response (list of dicts or dict with rows/error)

        except Exception as e:
            logging.exception(f"Exception running analysis query via MCP: {e}")
            return {"error": f"Exception querying BigQuery via MCP: {str(e)}"}

    async def _process_analysis_data(
        self, query_rows: List[Dict[str, Any]], analysis_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the raw data (list of dicts) from BigQuery into useful insights."""
        analysis_name = analysis_config.get("name", "Unnamed Analysis")
        logging.info(
            "Processing analysis data for "{analysis_name}'. Received {len(query_rows)} rows."
        )

        # If no data, return a specific structure
        if not query_rows:
            logging.warning(
                "No data rows received for processing for analysis "{analysis_name}'."
            )
            return {
                "metrics": {},
                "charts_data": [],
                "findings": ["No data returned from query to process."],
                "summary": "Query returned no data.",
            }

        # Convert to DataFrame for easier processing, if not empty
        try:
            df = pd.DataFrame(query_rows)
        except Exception as e:
            logging.exception(
                "Error converting query rows to DataFrame for "{analysis_name}': {e}"
            )
            return {"error": f"Could not process query data into DataFrame: {str(e)}"}

        if df.empty:
            logging.warning(
                "DataFrame is empty after converting query rows for "{analysis_name}'."
            )
            return {
                "metrics": {},
                "charts_data": [],
                "findings": [
                    "Query data resulted in an empty dataset after processing."
                ],
                "summary": "Processed data is empty.",
            }

        # Example: Generic descriptive statistics (can be expanded based on analysis_config type)
        # This part remains somewhat mock/example, but operates on real (mocked MCP) data.
        results = {"metrics": {}, "charts_data": [], "findings": []}
        try:
            # Calculate overall average score if 'score' or 'avg_score' column exists
            score_col = None
            if "score" in df.columns:
                score_col = "score"
            elif "avg_score" in df.columns:
                score_col = "avg_score"

            if score_col:
                # Attempt to convert to numeric, coercing errors to NaN
                df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
                avg_overall_score = df[score_col].mean()
                results["metrics"]["avg_overall_score"] = (
                    round(avg_overall_score, 3)
                    if not pd.isna(avg_overall_score)
                    else None
                )
                results["metrics"]["num_valid_scores"] = int(df[score_col].count())
                results["findings"].append(
                    f"Average score across {df.shape[0]} entries: {results['metrics']['avg_overall_score']:.2f}"
                    if results["metrics"]["avg_overall_score"] is not None
                    else "Average score could not be calculated."
                )

            # Example: Generate a simple chart data if there's a time column and a value column
            time_col = analysis_config.get("time_column_name", "timestamp")
            value_col = analysis_config.get(
                "value_column_name", score_col if score_col else "value"
            )  # default to score_col if available

            if (
                time_col in df.columns
                and value_col in df.columns
                and value_col != time_col
            ):
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
                df_sorted = df.dropna(subset=[time_col, value_col]).sort_values(
                    by=time_col
                )
                if not df_sorted.empty:
                    results["charts_data"].append(
                        {
                            "chart_type": "line",
                            "title": f"{value_col.replace('_', ' ').title()} Over {time_col.replace('_', ' ').title()}",
                            "x_axis_label": time_col,
                            "y_axis_label": value_col,
                            "data_points": df_sorted[[time_col, value_col]].to_dict(
                                orient="records"
                            ),
                        }
                    )
                    results["findings"].append(
                        "Trend analysis performed on "{value_col}' over '{time_col}'."
                    )

            results["summary"] = (
                f"Processed {df.shape[0]} data points. Found {len(results['findings'])} findings."
            )
            if not results["findings"]:
                results["findings"].append(
                    "Basic descriptive analysis performed. No specific pre-canned findings generated."
                )

        except Exception as e:
            logging.exception(
                "Error during pandas processing for "{analysis_name}': {e}"
            )
            results["error"] = f"Error during data processing: {str(e)}"
            results["findings"].append(
                f"Error during data processing: {str(e)}. Some results might be incomplete."
            )

        return results

    async def _store_analysis_results(
        self, processed_results: Dict[str, Any], analysis_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store the analysis results (summary, not raw data) in BigQuery via MCP."""
        analysis_name = analysis_config.get("name", "Unnamed Analysis")
        analysis_id = analysis_config.get(
            "analysis_id",
            f"{analysis_name.replace(' ', '_').lower()}_{uuid.uuid4().hex[:8]}",
        )
        output_table_id = self.bq_analysis_results_table

        if not output_table_id:
            logging.error(
                "BigQuery analysis results table ID is not configured. Cannot store results."
            )
            return {"error": "BQ analysis results table ID not configured."}

        logging.info(
            "Storing analysis results for "{analysis_name}' (ID: {analysis_id}) in BQ table: {output_table_id}"
        )

        # We are storing the JSON summary of results, not re-inserting all raw data.
        # The schema of this output table should accommodate this structure.
        # For example, a table with columns: analysis_id (STRING), analysis_name (STRING),
        # timestamp (TIMESTAMP), results_summary (JSON/STRING), config_details (JSON/STRING)
        row_to_insert = {
            "analysis_id": analysis_id,
            "analysis_name": analysis_name,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "results_summary_json": json.dumps(
                processed_results
            ),  # Store the whole results dict as JSON string
            "analysis_config_json": json.dumps(
                analysis_config
            ),  # Store the config for reproducibility
        }

        try:
            insert_response = await self.mcp_manager.call_tool(
                server_name=self.default_bigquery_mcp,
                tool_name="insert_rows",
                parameters={
                    "table_id": output_table_id,
                    "rows": [row_to_insert],  # insert_rows expects a list of rows
                    "dataset_id": self.bq_dataset_id,  # BQ MCP might need dataset_id for insert
                },
            )

            # Mock BQ server's insert_rows returns: {"status": "success", "inserted_count": N} or {"error": ...}
            if (
                insert_response
                and insert_response.get("status") == "success"
                and insert_response.get("inserted_count", 0) > 0
            ):
                logging.info(
                    "Successfully stored analysis results for "{analysis_id}' in BigQuery table '{output_table_id}'."
                )
                return {
                    "status": "success",
                    "table_id": output_table_id,
                    "dataset_id": self.bq_dataset_id,
                    "analysis_id_stored": analysis_id,
                    "rows_inserted": insert_response.get("inserted_count"),
                }
            else:
                error_msg = (
                    insert_response.get("error", "Unknown error during BQ insert")
                    if isinstance(insert_response, dict)
                    else "Unknown BQ insert error"
                )
                logging.error(
                    f"Failed to store analysis results in BigQuery: {error_msg}"
                )
                return {"error": f"BigQuery MCP insert error: {error_msg}"}
        except Exception as e:
            logging.exception(f"Exception storing analysis results in BigQuery: {e}")
            return {"error": f"Exception storing analysis results in BQ: {str(e)}"}


# Example Usage (conceptual) - Needs an async context to run (e.g., asyncio.run())
async def run_ae_test_with_mcp():
    print("\nTesting AnalysisEngine with live MCPManager (async)..")

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "pipeline_config.json"

    if not config_path.exists():
        print(f"ERROR: pipeline_config.json not found at {config_path}")
        return

    with open(config_path, "r") as f:
        pipeline_config = json.load(f)
    print(f"Loaded pipeline config from {config_path}")
    print(
        f"BigQuery MCP mock_mode: {pipeline_config.get('mcp_server_configurations', {}).get('bq_main', {}).get('config', {}).get('mock_mode')}"
    )

    mcp_manager = MCPManager(config=pipeline_config)

    ae_config_from_pipeline = pipeline_config.get("analysis_engine_config", {})
    if not ae_config_from_pipeline:
        print("Warning: 'analysis_engine_config' not in pipeline_config.json.")

    engine = AnalysisEngine(mcp_manager=mcp_manager, config=ae_config_from_pipeline)

    # This query will go to the BQ MCP server (mock or real)
    # The mock BQ server returns a fixed set of rows for any query currently.
    general_task = {
        "name": "Experiment Performance Summary (via MCP)",
        "query": "SELECT experiment_id, AVG(final_score) as avg_score, COUNT(*) as num_trials FROM mock_project.mock_dataset.trial_results GROUP BY experiment_id",
        "analysis_type": "general_summary",  # For _process_analysis_data logic if it uses it
        "output_storage_method": "bigquery",  # For context
    }
    print(f"\nSubmitting General Task: {general_task.get('name')}")
    results1 = await engine.perform_deep_analysis(general_task)
    print(
        "Results for "{general_task.get('name')}':\n{json.dumps(results1, indent=2, default=str)}"
    )

    # Another example, perhaps for trend analysis (mock BQ will return same data)
    trend_task = {
        "name": "Score Trend Over Time (via MCP)",
        "query": "SELECT trial_timestamp as timestamp, final_score as score FROM mock_project.mock_dataset.trial_results ORDER BY trial_timestamp",
        "analysis_type": "trend_detection",
        "time_column_name": "timestamp",  # Used by _process_analysis_data
        "value_column_name": "score",  # Used by _process_analysis_data
        "output_storage_method": "log_only",  # For context
    }
    print(f"\nSubmitting Trend Task: {trend_task.get('name')}")
    results2 = await engine.perform_deep_analysis(trend_task)
    print(
        "Results for "{trend_task.get('name')}':\n{json.dumps(results2, indent=2, default=str)}"
    )

    # Test with an intentionally bad query to see error handling
    bad_query_task = {
        "name": "Bad Query Test (via MCP)",
        "query": "SELECT * FROM non_existent_table",
    }
    print(f"\nSubmitting Bad Query Task: {bad_query_task.get('name')}")
    results_bad_query = await engine.perform_deep_analysis(bad_query_task)
    print(
        "Results for "{bad_query_task.get('name')}':\n{json.dumps(results_bad_query, indent=2, default=str)}"
    )

    # Test with an empty query string
    empty_query_task = {
        "name": "Empty Query Test (via MCP)",
        "query": "",
    }
    print(f"\nSubmitting Empty Query Task: {empty_query_task.get('name')}")
    results_empty_query = await engine.perform_deep_analysis(empty_query_task)
    print(
        "Results for "{empty_query_task.get('name')}':\n{json.dumps(results_empty_query, indent=2, default=str)}"
    )

    # Clean up MCP client resources if any
    for server_instance in mcp_manager.mcp_servers.values():
        if hasattr(server_instance, "close_client") and asyncio.iscoroutinefunction(
            server_instance.close_client
        ):
            await server_instance.close_client()

    print("\nAnalysisEngine with MCPManager (async) conceptual test complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )
    asyncio.run(run_ae_test_with_mcp())
