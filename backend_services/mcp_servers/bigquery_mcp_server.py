import pandas as pd  # For converting query results to DataFrame
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from .base_mcp_server import BaseMCPServer


class BigQueryMCPServer(BaseMCPServer):
    """
    MCP Server implementation for interacting with Google BigQuery.
    """

    def __init__(self, server_name: str, config: dict):
        """
        Initializes the BigQueryMCPServer.
        Expected config keys:
            - "project_id": Your Google Cloud Project ID.
            - "dataset_id" (optional): Default dataset ID to use if not specified in calls.
            - "location" (optional): Default location for BigQuery jobs (e.g., "US").
        """
        self.client: Optional[bigquery.Client] = None
        super().__init__(server_name, config)

    def _validate_config(self):
        """Validates that 'project_id' is in the config."""
        if "project_id" not in self.config:
            raise ValueError(
                "Missing "project_id' in config for BigQueryMCPServer: {self.server_name}"
            )
        self.project_id = self.config["project_id"]
        self.default_dataset_id = self.config.get("dataset_id")
        self.default_location = self.config.get("location")

    def _initialize_client(self):
        """Initializes the BigQuery client."""
        try:
            self.client = bigquery.Client(
                project=self.project_id, location=self.default_location
            )
            # Test connection by listing datasets (lightweight call)
            print(
                "BigQuery client initialized for project "{self.project_id}' in location '{self.client.location}' for server '{self.server_name}'. Attempting to list datasets (max 1)..."
            )
            datasets = list(self.client.list_datasets(max_results=1))
            if datasets:
                print(
                    f"  Successfully listed at least one dataset: {datasets[0].dataset_id}"
                )
            else:
                print(
                    "  No datasets found or not permitted to list, but client initialized."
                )
        except Exception as e:
            print(
                f"CRITICAL: Error initializing BigQuery client for {self.server_name}: {e}"
            )
            # Allow to proceed without a client for now, calls will fail. Or raise to stop startup.
            self.client = None  # Ensure client is None if init fails
            # raise # Uncomment to make BQ client initialization critical for startup

    def _run_query(
        self,
        query: str,
        query_parameters: list = None,
        to_dataframe: bool = False,
        target_dataset_id: str = None,
        target_table_id: str = None,
        write_disposition: str = "WRITE_EMPTY",
        create_disposition: str = "CREATE_IF_NEEDED",
    ) -> any:
        """
        Executes a SQL query in BigQuery. Can also write results to a destination table.
        """
        if not self.client:
            return {
                "error": f"BigQuery client not initialized for {self.server_name}. Query cannot run."
            }

        print(f"  {self.server_name}._run_query called. To DataFrame: {to_dataframe}")
        print("    Query (first 100 chars): "{query[:100]}...'")
        if query_parameters:
            print(f"    Query Parameters: {query_parameters}")

        job_config = bigquery.QueryJobConfig()
        if query_parameters:
            # Convert simple list of dicts to BigQuery ScalarQueryParameter objects
            # Example: [{"name": "param_name", "type": "STRING", "value": "val"}]
            bq_params = []
            for p in query_parameters:
                if isinstance(p, dict) and all(
                    k in p for k in ["name", "type", "value"]
                ):
                    bq_params.append(
                        bigquery.ScalarQueryParameter(
                            p["name"], p["type"].upper(), p["value"]
                        )
                    )
                else:
                    # Or log a warning and skip, or raise error depending on strictness
                    print(f"    Warning: Invalid query parameter format, skipping: {p}")
            job_config.query_parameters = bq_params

        if target_table_id:
            if not target_dataset_id and not self.default_dataset_id:
                return {
                    "error": "target_dataset_id or default_dataset_id required when target_table_id is specified."
                }
            dest_dataset_ref = self.client.dataset(
                target_dataset_id or self.default_dataset_id
            )
            job_config.destination = dest_dataset_ref.table(target_table_id)
            job_config.write_disposition = (
                bigquery.WriteDisposition.WRITE_TRUNCATE
                if write_disposition.upper() == "WRITE_TRUNCATE"
                else (
                    bigquery.WriteDisposition.WRITE_APPEND
                    if write_disposition.upper() == "WRITE_APPEND"
                    else bigquery.WriteDisposition.WRITE_EMPTY
                )
            )
            job_config.create_disposition = (
                bigquery.CreateDisposition.CREATE_NEVER
                if create_disposition.upper() == "CREATE_NEVER"
                else bigquery.CreateDisposition.CREATE_IF_NEEDED
            )
            print(
                f"    Query results will be written to: {job_config.destination}. Write: {job_config.write_disposition}, Create: {job_config.create_disposition}"
            )

        try:
            query_job = self.client.query(query, job_config=job_config)  # API request
            print(f"    Query job {query_job.job_id} started. Waiting for results...")
            results_iterator = query_job.result()  # Waits for query to finish
            print(
                f"    Query job {query_job.job_id} finished. Rows affected: {query_job.num_dml_affected_rows}, Rows returned (if not DDL/DML): {results_iterator.total_rows if results_iterator else 'N/A'}"
            )

            if query_job.destination:  # If results were written to a table
                dest_table = self.client.get_table(query_job.destination)
                return {
                    "status": "success",
                    "destination_table": str(dest_table.reference),
                    "rows_written": dest_table.num_rows,
                }

            if to_dataframe:
                df = results_iterator.to_dataframe()
                print(f"    Converted results to DataFrame. Shape: {df.shape}")
                return df
            else:
                # Convert rows to a list of dicts for easier JSON serialization
                rows = [dict(row) for row in results_iterator]
                print(f"    Returning {len(rows)} rows as list of dicts.")
                return rows
        except NotFound as nf_error:
            print(f"BigQuery NotFound error for {self.server_name}: {nf_error}")
            return {"error": "BigQuery resource not found", "details": str(nf_error)}
        except Exception as e:
            print(f"Error running query with BigQuery for {self.server_name}: {e}")
            # Consider more specific error handling for different BQ exceptions
            return {
                "error": f"BigQuery query execution failed for {self.server_name}",
                "details": str(e),
            }

    def _insert_rows(
        self,
        table_id: str,
        rows: list[dict],
        dataset_id: str = None,
        skip_invalid_rows: bool = False,
        ignore_unknown_values: bool = False,
    ) -> dict:
        """
        Inserts rows into a BigQuery table.
        """
        if not self.client:
            return {
                "error": f"BigQuery client not initialized for {self.server_name}. Cannot insert rows."
            }

        target_dataset_str = dataset_id or self.default_dataset_id
        if not target_dataset_str:
            return {
                "error": "Dataset ID must be provided either in server config ("dataset_id') or call for insert_rows in {self.server_name}."
            }

        full_table_id = f"{self.project_id}.{target_dataset_str}.{table_id}"
        print(
            f"  {self.server_name}._insert_rows called for table '{full_table_id}'. Num rows: {len(rows)}."
        )
        if not rows:
            return {
                "status": "success",
                "message": "No rows provided to insert.",
                "inserted_count": 0,
                "errors": [],
            }
        print(f"    First row (sample): {str(rows[0])[:200]}...")

        try:
            table_ref = self.client.dataset(target_dataset_str).table(table_id)
            # Get table to ensure it exists, or to let API create it if schema is also provided (not done here)
            # self.client.get_table(table_ref) # This would raise NotFound if table doesn't exist and not creating

            job_config = bigquery.LoadJobConfig()
            job_config.skip_leading_rows = (
                0  # Assuming rows is a list of dicts, not CSV data
            )
            job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
            # job_config.autodetect = True # If schema is not perfectly matching or table doesn't exist
            # For more control, define schema explicitly: job_config.schema = [ bigquery.SchemaField(...), ... ]
            job_config.ignore_unknown_values = ignore_unknown_values
            job_config.skip_invalid_rows = skip_invalid_rows

            # Using insert_rows_json for simplicity with list of dicts
            # For larger loads, using client.load_table_from_json or load_table_from_uri is better.
            errors = self.client.insert_rows_json(
                table_ref, rows, job_config=None
            )  # job_config applies to load jobs, not insert_rows_json directly in this way
            # For insert_rows options, they are direct params to insert_rows itself if available, or often default.
            # For `skip_invalid_rows` and `ignore_unknown_values` in `insert_rows_json`, they are parameters to the method call.
            # errors = self.client.insert_rows_json(table_ref, rows, skip_invalid_rows=skip_invalid_rows, ignore_unknown_values=ignore_unknown_values)

            if not errors:  # If errors is an empty list
                print(
                    f"    Successfully inserted {len(rows)} rows into {full_table_id}."
                )
                return {"status": "success", "inserted_count": len(rows), "errors": []}
            else:
                print(
                    f"    Encountered errors while inserting rows to {full_table_id}: {errors}"
                )
                # Try to provide a count of successfully inserted rows if possible/meaningful
                # This is tricky with insert_rows_json as it's more all-or-nothing per batch for atomicity of the call itself.
                return {"status": "failure", "errors": errors, "inserted_count": 0}
        except NotFound as nf_error:
            print(
                "BigQuery table "{full_table_id}' not found during insert for {self.server_name}: {nf_error}"
            )
            return {
                "error": "BigQuery table "{full_table_id}' not found",
                "details": str(nf_error),
            }
        except Exception as e:
            print(f"Error inserting rows with BigQuery for {self.server_name}: {e}")
            return {
                "error": f"BigQuery insert_rows failed for {self.server_name}",
                "details": str(e),
            }

    def call_tool(self, tool_name: str, parameters: dict) -> any:
        """
        Executes a tool supported by the BigQueryMCPServer.
        """
        if tool_name == "run_query":
            query = parameters.get("query")
            if not query:
                raise ValueError(
                    "Missing "query' parameter for 'run_query' tool in {self.server_name}."
                )
            return self._run_query(
                query=query,
                query_parameters=parameters.get("query_parameters"),
                to_dataframe=parameters.get("to_dataframe", False),
                target_dataset_id=parameters.get("target_dataset_id"),
                target_table_id=parameters.get("target_table_id"),
                write_disposition=parameters.get("write_disposition", "WRITE_EMPTY"),
                create_disposition=parameters.get(
                    "create_disposition", "CREATE_IF_NEEDED"
                ),
            )
        elif tool_name == "insert_rows":
            table_id = parameters.get("table_id")
            rows_to_insert = parameters.get("rows")
            if not table_id or rows_to_insert is None:  # rows can be an empty list
                raise ValueError(
                    "Missing "table_id' or 'rows' parameter for 'insert_rows' tool in {self.server_name}."
                )
            return self._insert_rows(
                table_id=table_id,
                rows=rows_to_insert,
                dataset_id=parameters.get("dataset_id"),
                skip_invalid_rows=parameters.get("skip_invalid_rows", False),
                ignore_unknown_values=parameters.get("ignore_unknown_values", False),
            )
        else:
            raise NotImplementedError(
                "Tool "{tool_name}' is not supported by {self.server_name} ({self.__class__.__name__})."
            )


# Example Usage (conceptual)
if __name__ == "__main__":
    print(
        "Testing BigQueryMCPServer with actual client (requires GCP auth & project)..."
    )
    # THIS TEST WILL FAIL IF YOU DON'T HAVE GCP AUTHENTICATION SET UP
    # AND A VALID PROJECT ID IN THE CONFIG
    sample_bq_config_real = {
        "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT"),  # Tries to use env var
        "dataset_id": "generic_test_dataset",  # This dataset should exist or be creatable
        "location": "US",
    }
    if not sample_bq_config_real["project_id"]:
        print(
            "SKIPPING real BigQueryMCPServer test: GOOGLE_CLOUD_PROJECT env var not set."
        )
    else:
        print(f"Using GCP Project: {sample_bq_config_real['project_id']} for test.")
        bq_server_real = BigQueryMCPServer(
            server_name="bq_real_test", config=sample_bq_config_real
        )

        if bq_server_real.client:  # Only proceed if client initialized
            test_table_id = "mcp_test_table_inserts"
            # Test insert_rows
            rows_data = [
                {
                    "name": "alpha",
                    "value": 10,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                },
                {
                    "name": "beta",
                    "value": 25,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                },
                {
                    "name": "gamma",
                    "value": 15,
                    "invalid_extra_field": True,
                },  # Test ignore_unknown_values
            ]
            insert_params_real = {
                "table_id": test_table_id,
                "rows": rows_data,
                "dataset_id": sample_bq_config_real["dataset_id"],
                "ignore_unknown_values": True,
            }
            print(
                "\nSimulating insert_rows to table "{sample_bq_config_real['dataset_id']}.{test_table_id}'"
            )
            insert_status_real = bq_server_real.call_tool(
                "insert_rows", insert_params_real
            )
            print(f"Insert Status (real client):\n{insert_status_real}")

            # Test run_query (reading back what was inserted if successful)
            if insert_status_real.get("status") == "success":
                query_params_real = {
                    "query": f"SELECT name, value FROM `{sample_bq_config_real['project_id']}.{sample_bq_config_real['dataset_id']}.{test_table_id}` ORDER BY value DESC LIMIT 5",
                    "to_dataframe": True,
                }
                print("\nSimulating run_query from table "{test_table_id}'")
                query_results_real = bq_server_real.call_tool(
                    "run_query", query_params_real
                )
                print(f"Query Results (real client, DataFrame):\n{query_results_real}")

                # Test query with results to destination table
                dest_table_query = f"CREATE OR REPLACE TABLE `{sample_bq_config_real['project_id']}.{sample_bq_config_real['dataset_id']}.{test_table_id}_summary` AS SELECT name, SUM(value) as total_value FROM `{sample_bq_config_real['project_id']}.{sample_bq_config_real['dataset_id']}.{test_table_id}` GROUP BY name"
                query_to_table_params = {
                    "query": dest_table_query,
                    # Not using target_dataset_id/target_table_id here because the query itself specifies destination via DDL
                    # If it were a SELECT query whose results we want in a new table, then we'd set target_table_id.
                }
                print("\nSimulating run_query with DDL to create summary table...")
                ddl_result = bq_server_real.call_tool(
                    "run_query", query_to_table_params
                )
                print(f"DDL Query Result (real client):\n{ddl_result}")

            # Test a simple public dataset query if inserts failed or for broader test
            public_query_params = {
                "query": "SELECT COUNT(*) as num_ shakespeare_corpus FROM `bigquery-public-data.samples.shakespeare`",
            }
            print("\nSimulating run_query on public dataset...")
            public_results = bq_server_real.call_tool("run_query", public_query_params)
            print(f"Public Query Results (real client):\n{public_results}")

        else:
            print("Skipping real BigQuery client tests as client did not initialize.")

    print("\nBigQueryMCPServer test complete.")

# Need to import datetime for test data
import datetime
import os
from typing import Optional  # For self.client type hint

# Example Usage (conceptual)
if __name__ == "__main__":
    print("Testing BigQueryMCPServer...")
    sample_bq_config = {
        "project_id": "your-gcp-project-id-placeholder",
        "dataset_id": "my_research_data",
        "location": "US",
    }

    try:
        bq_server = BigQueryMCPServer(
            server_name="bq_main_test", config=sample_bq_config
        )

        # Test run_query
        query_params = {
            "query": "SELECT name, SUM(number) as total_count FROM `my_dataset.my_table` WHERE score > @min_score GROUP BY name;",
            "query_parameters": [
                {
                    "name": "min_score",
                    "parameterType": {"type": "FLOAT64"},
                    "parameterValue": {"value": 0.75},
                }
            ],  # Example BQ param format
        }
        print(f"\nSimulating run_query with params: {query_params}")
        query_results = bq_server.call_tool("run_query", query_params)
        print(f"Query Results (mock):\n{query_results}")

        # Test insert_rows
        insert_params = {
            "table_id": "experiment_trials",
            "rows": [
                {"trial_id": "t001", "score": 0.85, "params": '{"variant": "A"}'},
                {"trial_id": "t002", "score": 0.92, "params": '{"variant": "B"}'},
            ],
            # "dataset_id": "optional_override_dataset" # Can also be specified here
        }
        print("\nSimulating insert_rows to table "{insert_params['table_id']}'")
        insert_status = bq_server.call_tool("insert_rows", insert_params)
        print(f"Insert Status (mock):\n{insert_status}")

        status = bq_server.get_status()
        print(f"\nServer Status: {status}")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\nBigQueryMCPServer test complete.")
