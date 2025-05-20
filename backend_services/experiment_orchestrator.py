# backend_services/experiment_orchestrator.py

import datetime
import json
import asyncio # For main test block if needed
# from .mcp_manager import MCPManager # Assume MCPManager is available
# For actual GCP calls, these would be needed by the respective MCP servers:
# from google.cloud import firestore, bigquery, pubsub_v1

class ExperimentOrchestrator:
    """
    A Python class responsible for:
    - Fetching experiment configurations (from Firestore via MCP).
    - Iteratively running trials by interacting with:
        - The Padres environment (on Cloud Run, via a padres-mcp).
        - LLM MCPs (via MCPManager).
    - Storing detailed trial data in BigQuery (via MCPManager).
    """

    def __init__(self, mcp_manager, config=None):
        """
        Initializes the ExperimentOrchestrator.
        Args:
            mcp_manager (MCPManager): An instance of MCPManager.
            config (dict, optional): Orchestrator-specific configurations, e.g.,
                                     default_padres_mcp_server_name, 
                                     default_firestore_mcp_server_name,
                                     default_bigquery_mcp_server_name,
                                     default_pubsub_mcp_server_name,
                                     gcp_project_id,
                                     pubsub_topic_experiment_completed.
        """
        if mcp_manager is None:
            raise ValueError("MCPManager instance is required for ExperimentOrchestrator.")
        self.mcp_manager = mcp_manager
        self.config = config if config else {}
        self.default_padres_mcp = self.config.get("default_padres_mcp_server_name", "padres_main")
        self.default_firestore_mcp = self.config.get("default_firestore_mcp_server_name", "firestore_main")
        self.default_bigquery_mcp = self.config.get("default_bigquery_mcp_server_name", "bigquery_main")
        self.default_pubsub_mcp = self.config.get("default_pubsub_mcp_server_name", "pubsub_main")
        print("ExperimentOrchestrator initialized.")

    async def fetch_experiment_config(self, experiment_id: str) -> dict:
        """
        Fetches an experiment configuration from Firestore via an MCP server.
        """
        print(f"Fetching experiment config for '{experiment_id}' using MCP ({self.default_firestore_mcp}).")
        try:
            config_data = await self.mcp_manager.call_tool(
                server_name=self.default_firestore_mcp,
                tool_name="get_document",
                parameters={
                    "collection_id": self.config.get("firestore_experiment_collection", "experiment_configs"),
                    "document_id": experiment_id
                }
            )
            if config_data and not config_data.get("error"):
                return config_data
            else:
                raise ValueError(f"Experiment config '{experiment_id}' not found or error fetching: {config_data.get('error')}")
        except Exception as e:
            print(f"Error fetching experiment config '{experiment_id}': {e}")
            # Depending on desired behavior, could return a default, raise, or return None
            raise

    async def run_experiment_batch(self, batch_config: dict):
        """
        Runs a batch of experiments based on the provided configuration.
        Batch config could list experiment_ids or define a template for generating them.
        """
        batch_name = batch_config.get('name', 'Unnamed Batch')
        print(f"Running experiment batch: {batch_name}.")
        experiment_ids_to_run = batch_config.get("experiment_ids", [])

        for exp_id in experiment_ids_to_run:
            try:
                print(f"Starting experiment '{exp_id}' from batch '{batch_name}'.")
                config = await self.fetch_experiment_config(exp_id)
                await self.run_single_experiment(config)
                print(f"Successfully completed experiment '{exp_id}'.")
            except Exception as e:
                print(f"Failed to run experiment '{exp_id}' from batch '{batch_name}': {e}")
                # Log error, potentially mark experiment as failed
        print(f"Experiment batch '{batch_name}' finished.")

    async def run_single_experiment(self, experiment_config: dict):
        """
        Runs a single experiment based on its configuration using MCP manager for interactions.
        """
        exp_id = experiment_config.get("id", "unknown_experiment")
        print(f"Running single experiment: {exp_id}.")
        num_trials = experiment_config.get("num_trials", 1)
        llm_mcp_server_name = experiment_config.get("llm_mcp_server_name", "default_llm")

        for i in range(num_trials):
            trial_num = i + 1
            print(f"  Starting trial {trial_num} for experiment {exp_id}.")
            trial_successful = False
            try:
                # 1. Setup Padres environment (via padres-mcp)
                padres_task_name = experiment_config.get("padres_task_name", "default_task")
                padres_params = experiment_config.get("padres_task_parameters", {})
                print(f"    Setting up Padres env ({self.default_padres_mcp}) for task '{padres_task_name}'.")
                env_setup_response = await self.mcp_manager.call_tool(
                    server_name=self.default_padres_mcp,
                    tool_name="setup_environment",
                    parameters={"task_name": padres_task_name, "task_params": padres_params}
                )
                if env_setup_response.get("error") or not env_setup_response.get("environment_id"):
                    raise RuntimeError(f"Failed to setup Padres environment: {env_setup_response.get('error')}")
                current_env_state = env_setup_response.get("initial_state")
                mock_environment_id = env_setup_response.get("environment_id")
                print(f"    Padres environment setup complete. Env ID: {mock_environment_id}, State: {current_env_state}")

                # 2. Format prompt (using templates from config or another source - TBD)
                prompt_template = experiment_config.get("prompt_template", "Default prompt: {task_description}")
                prompt = prompt_template.format(task_description=experiment_config.get("description", padres_task_name))
                print(f"    Formatted prompt (first 80 chars): '{prompt[:80]}...'")

                # 3. Call the appropriate LLM MCP to get actions
                print(f"    Calling LLM ({llm_mcp_server_name}) for action.")
                action_response_dict = await self.mcp_manager.call_tool(
                    server_name=llm_mcp_server_name,
                    tool_name="generate_content",
                    parameters={"prompt_or_messages": prompt, "generation_config": {"max_output_tokens": 50}}
                )
                if not isinstance(action_response_dict, dict) or action_response_dict.get("error") or action_response_dict.get("status") != "success":
                    raise RuntimeError(f"LLM call failed: {action_response_dict}")
                action_from_llm = action_response_dict.get("text")
                print(f"    LLM proposed action: {action_from_llm}")

                # 4. Execute actions in Padres (via padres-mcp)
                print(f"    Executing action in Padres env ({self.default_padres_mcp}).")
                execution_result = await self.mcp_manager.call_tool(
                    server_name=self.default_padres_mcp,
                    tool_name="execute_action",
                    parameters={"environment_id": mock_environment_id, "action_name": action_from_llm}
                )
                if not isinstance(execution_result, dict) or execution_result.get("error"):
                    raise RuntimeError(f"Padres action execution failed: {execution_result}")
                print(f"    Padres execution result: {execution_result}")
                trial_successful = execution_result.get("status") == "success"

                # 5. Collect results and scores
                trial_data = {
                    "experiment_id": exp_id,
                    "trial_number": trial_num,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "config_used": experiment_config,
                    "prompt_sent_to_llm": prompt,
                    "action_proposed_by_llm": action_from_llm,
                    "llm_response_details": action_response_dict,
                    "action_execution_result": execution_result,
                    "final_score_for_trial": execution_result.get("score_delta") if isinstance(execution_result, dict) else execution_result.get("score"),
                    "trial_successful": trial_successful
                }
                
                # 6. Store detailed trial data in BigQuery (via bigquery-mcp)
                await self._store_trial_data(trial_data)
                print(f"  Trial {trial_num} for experiment {exp_id} completed successfully.")

            except Exception as e:
                print(f"  Error during trial {trial_num} for experiment {exp_id}: {type(e).__name__} - {e}")
                # Log error for this trial, it might not have completed successfully
                # Potentially store partial failure information if desired
        
        print(f"Experiment {exp_id} finished processing all trials.")
        await self._publish_completion_event(exp_id, num_trials)

    async def _store_trial_data(self, trial_data: dict):
        """
        Stores a single trial's data into BigQuery via the MCPManager.
        """
        print(f"  Storing trial data for {trial_data['experiment_id']} (trial {trial_data['trial_number']}) using MCP ({self.default_bigquery_mcp}).")
        try:
            storage_response = await self.mcp_manager.call_tool(
                server_name=self.default_bigquery_mcp,
                tool_name="insert_rows",
                parameters={
                    "table_id": self.config.get("bigquery_trial_results_table", "trial_results"),
                    "rows": [trial_data]
                }
            )
            if not (isinstance(storage_response, dict) and storage_response.get("status") == "success"):
                 print(f"    Warning: Storing trial data via MCP might have failed or had errors: {storage_response}")
            else:
                print(f"    Trial data storage successful. Response: {storage_response}")
        except Exception as e:
            print(f"  Error storing trial data via MCP: {e}")
            # Handle storage failure (e.g., retry, log to a fallback)

    async def _publish_completion_event(self, experiment_id: str, num_trials_processed: int):
        """
        Publishes an event to Pub/Sub (via MCP) when an experiment is completed.
        """
        print(f"Publishing completion for {experiment_id} (processed {num_trials_processed} trials) via MCP ({self.default_pubsub_mcp}).")
        message_payload = {
            "experiment_id": experiment_id, 
            "status": "completed", 
            "num_trials_processed": num_trials_processed,
            "completion_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        try:
            pubsub_response = await self.mcp_manager.call_tool(
                server_name=self.default_pubsub_mcp,
                tool_name="publish_message",
                parameters={
                    "topic_id": self.config.get("pubsub_topic_experiment_completed", "experiment-completed"),
                    "message_data": message_payload
                }
            )
            if not (isinstance(pubsub_response, dict) and pubsub_response.get("status") == "success"):
                print(f"  Warning: Pub/Sub publish may have failed: {pubsub_response}")
            else:
                print(f"  Pub/Sub message published successfully for {experiment_id}. Message ID: {pubsub_response.get('message_id')}")
        except Exception as e:
            print(f"  Error publishing completion event via MCP: {e}")

# Example Usage (conceptual) - Needs an async context to run
async def run_orchestrator_test():
    # Mock MCPManager and its call_tool for testing ExperimentOrchestrator
    class MockMCPManagerForOrchestrator:
        def __init__(self, available_servers=None):
            self.available_servers = available_servers if available_servers else ["firestore_main", "llm_gemini_test", "bigquery_main", "padres_main", "pubsub_main"]
            self.config = { "experiment_orchestrator_config": {"default_llm_mcp_server_name": "llm_gemini_test"}} # For llm_mcp_server_name fallback
            print(f"MockMCPManagerForOrchestrator initialized. Available servers: {self.available_servers}")

        async def call_tool(self, server_name: str, tool_name: str, parameters: dict) -> any: # Made async
            print(f"MockMCPManager: Async Called {server_name}.{tool_name} with {parameters}")
            await asyncio.sleep(0.01) # Simulate async work
            if server_name not in self.available_servers:
                return {"error": f"Mock server {server_name} not available."}

            if server_name == "firestore_main" and tool_name == "get_document":
                exp_id = parameters.get("document_id")
                if exp_id == "exp_001_valid":
                    return {
                        "id": "exp_001_valid", 
                        "description": "Valid fetched experiment config", 
                        "llm_mcp_server_name": "llm_gemini_test",
                        "padres_task_name": "find_object",
                        "num_trials": 1,
                        "prompt_template": "Please find the {object_name}."
                    }
                return {"error": f"Document {exp_id} not found in mock Firestore."}
            
            elif server_name == "llm_gemini_test" and tool_name == "generate_content":
                return {"text": f"Mock LLM action for prompt: '{parameters.get('prompt_or_messages', '')[:20]}...'", "status": "success"}

            elif server_name == "bigquery_main" and tool_name == "insert_rows":
                return {"status": "success", "inserted_count": len(parameters.get("rows", [])), "errors": []}
            
            elif server_name == "padres_main":
                 if tool_name == "setup_environment":
                     return {"environment_id": "mock_env_123", "initial_state": {"status": "ready"}, "message": "Mock env ready"}
                 elif tool_name == "execute_action":
                     return {"status": "success", "new_state": {"status": "action_done"}, "score_delta": 0.9, "observation": "Action performed mockly"}
            
            elif server_name == "pubsub_main" and tool_name == "publish_message":
                return {"status": "success", "message_id": "mock_pubsub_msg_id_123"}

            return {"error": f"Mock tool '{tool_name}' not implemented for server '{server_name}'."}

    print("\nTesting ExperimentOrchestrator (async)..")
    mock_mcp_mgr = MockMCPManagerForOrchestrator()
    
    orchestrator_config = {
        "default_padres_mcp_server_name": "padres_main",
        "default_firestore_mcp_server_name": "firestore_main",
        "default_bigquery_mcp_server_name": "bigquery_main",
        "default_pubsub_mcp_server_name": "pubsub_main",
        "firestore_experiment_collection": "experiment_configurations_test",
        "bigquery_trial_results_table": "trial_results_test",
        "pubsub_topic_experiment_completed": "test-experiment-completed"
    }
    orchestrator = ExperimentOrchestrator(mcp_manager=mock_mcp_mgr, config=orchestrator_config)
    
    try:
        print("\nAttempting to fetch valid experiment config (async)... ")
        valid_config = await orchestrator.fetch_experiment_config("exp_001_valid")
        print(f"Fetched config: {valid_config}")

        print("\nAttempting to run single experiment (async)... ")
        await orchestrator.run_single_experiment(valid_config)

    except ValueError as ve:
        print(f"ValueError during orchestrator test: {ve}")
    except RuntimeError as rte:
        print(f"RuntimeError during orchestrator test: {rte}")
    except Exception as e:
        print(f"Unexpected Exception during orchestrator test: {type(e).__name__} - {e}")

    sample_batch_config = {
        "name": "MyAsyncTestBatch_01",
        "experiment_ids": ["exp_001_valid"]
    }
    print("\nAttempting to run an experiment batch (async)... ")
    await orchestrator.run_experiment_batch(sample_batch_config)
    
    print("\nExperimentOrchestrator async conceptual test complete.")

if __name__ == "__main__":
    asyncio.run(run_orchestrator_test()) # Run the async test function 