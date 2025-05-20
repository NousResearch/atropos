import os
import json # ensure json is imported if used in __main__

print("--- PYTHON SCRIPT ENVIRONMENT VARIABLES ---")
for key, value in os.environ.items():
    if "API_KEY" in key: # Print only relevant keys for brevity and security
        print(f'{key}={value[:5]}...{value[-5:] if len(value) > 10 else "[TOO SHORT OR EMPTY]"}')
print("--- END OF API_KEY ENV VARS ---")

from run_single_padres_test import PadresTest
import requests
import json
from datetime import datetime
import os # For API keys

# Attempt to import BigQuery, will be used later
try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    print("Warning: google-cloud-bigquery not installed. BigQuery functionality will be disabled.")

class DirectResearchPipeline:
    def __init__(self, perplexity_api_key, anthropic_api_key, gcp_project_id=None, bigquery_dataset_id=None):
        anthropic_key_for_debug = os.getenv('ANTHROPIC_API_KEY') # Still get from env for debug
        perplexity_env_key = os.getenv('PERPLEXITY_API_KEY')

        print(f"DEBUG (DirectResearchPipeline init): ANTHROPIC_API_KEY from os.getenv: {anthropic_key_for_debug[:5]}...{anthropic_key_for_debug[-5:] if anthropic_key_for_debug and len(anthropic_key_for_debug) > 10 else '[ENV VAR INVALID OR TOO SHORT]'}")
        print(f"DEBUG (DirectResearchPipeline init): ANTHROPIC_API_KEY passed to init: {anthropic_api_key[:5]}...{anthropic_api_key[-5:] if anthropic_api_key and len(anthropic_api_key) > 10 else '[PARAM INVALID OR TOO SHORT]'}")
        print(f"DEBUG (DirectResearchPipeline init): PERPLEXITY_API_KEY from env: {perplexity_env_key[:5]}...{perplexity_env_key[-5:] if perplexity_env_key and len(perplexity_env_key) > 10 else '[ENV VAR INVALID OR TOO SHORT]'}")
        print(f"DEBUG (DirectResearchPipeline init): Perplexity API key passed to init: {perplexity_api_key[:5]}...{perplexity_api_key[-5:] if perplexity_api_key and len(perplexity_api_key) > 10 else '[PARAM INVALID OR TOO SHORT]'}")

        if not anthropic_api_key:
            print("Warning: Anthropic API key not provided to DirectResearchPipeline. Claude analysis will fail.")
        
        # Pass the key directly to PadresTest constructor
        self.padres = PadresTest(use_llm=True, anthropic_api_key_override=anthropic_api_key)
        self.perplexity_key = perplexity_api_key
        self.gcp_project_id = gcp_project_id
        self.bigquery_dataset_id = bigquery_dataset_id
        
        if BIGQUERY_AVAILABLE and self.gcp_project_id:
            try:
                self.bigquery_client = bigquery.Client(project=self.gcp_project_id)
                print(f"BigQuery client initialized for project {self.gcp_project_id}")
            except Exception as e:
                print(f"Warning: Failed to initialize BigQuery client: {e}")
                self.bigquery_client = None
        else:
            self.bigquery_client = None
            if not BIGQUERY_AVAILABLE:
                print("BigQuery client not initialized because google-cloud-bigquery is not available.")
            elif not self.gcp_project_id:
                print("BigQuery client not initialized because GCP_PROJECT_ID was not provided.")


    def research_with_perplexity(self, query, model="sonar-small-online"): # Using a generally available model
        """Direct Perplexity API call"""
        if not self.perplexity_key:
            print("Error: Perplexity API key not provided.")
            return {"error": "Perplexity API key missing"}
            
        headers = {
            "Authorization": f"Bearer {self.perplexity_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model, # Updated to a common, fast model
            "messages": [{"role": "user", "content": query}]
        }
        
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status() # Raise an exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Perplexity API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    return {"error": str(e), "details": e.response.json()}
                except json.JSONDecodeError:
                    return {"error": str(e), "details": e.response.text}
            return {"error": str(e)}
        except Exception as e:
            print(f"An unexpected error occurred with Perplexity API: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

    def store_in_bigquery(self, data_to_store, table_id):
        """Store data in Google BigQuery using direct API calls."""
        if not self.bigquery_client:
            print("Error: BigQuery client not initialized. Cannot store data.")
            return {"error": "BigQuery client not initialized."}
        if not self.bigquery_dataset_id:
            print("Error: BigQuery dataset ID not configured. Cannot store data.")
            return {"error": "BigQuery dataset ID not configured."}

        try:
            dataset_ref = self.bigquery_client.dataset(self.bigquery_dataset_id)
            table_ref = dataset_ref.table(table_id)

            # Attempt to get table to see if it exists, if not, create it (simplified schema)
            try:
                self.bigquery_client.get_table(table_ref)
            except Exception: # google.cloud.exceptions.NotFound
                print(f"Table {table_id} not found in dataset {self.bigquery_dataset_id}. Attempting to create.")
                # Define a simple schema based on common keys in combined_result
                # This is a very basic schema, might need adjustment based on actual data structure
                schema = [
                    bigquery.SchemaField("experiment_status", "STRING"),
                    bigquery.SchemaField("experiment_llm_analysis", "STRING"),
                    bigquery.SchemaField("research_query_content", "STRING"),
                    bigquery.SchemaField("research_model_used", "STRING"),
                    bigquery.SchemaField("research_response_content", "STRING"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP"),
                ]
                bq_table = bigquery.Table(table_ref, schema=schema)
                self.bigquery_client.create_table(bq_table)
                print(f"Table {table_id} created in dataset {self.bigquery_dataset_id}.")

            # Prepare row data (flattening complex structures for simplicity)
            row_to_insert = {
                "experiment_status": json.dumps(data_to_store.get('experiment', {}).get('status')),
                "experiment_llm_analysis": str(data_to_store.get('experiment', {}).get('llm_analysis', '')),
                "research_query_content": data_to_store.get('research', {}).get('request', {}).get('messages', [{}])[0].get('content', ''),
                "research_model_used": data_to_store.get('research', {}).get('model', ''),
                "research_response_content": str(data_to_store.get('research', {}).get('choices', [{}])[0].get('message', {}).get('content', '')),
                "timestamp": data_to_store.get('timestamp')
            }
            
            errors = self.bigquery_client.insert_rows_json(table_ref, [row_to_insert])
            if not errors:
                print(f"Data successfully inserted into BigQuery table: {self.bigquery_dataset_id}.{table_id}")
                return {"status": "success", "table_id": f"{self.bigquery_dataset_id}.{table_id}"}
            else:
                print(f"Errors encountered while inserting data into BigQuery: {errors}")
                return {"error": "Failed to insert data into BigQuery", "details": errors}

        except Exception as e:
            print(f"Error storing data in BigQuery: {e}")
            return {"error": str(e)}

    def run_enhanced_experiment(self, research_query_content="latest spatial reasoning AI research papers 2024"):
        """Run experiment + research + analysis + storage"""
        
        # 1. Run spatial experiment
        print("\n=== Running Spatial Experiment ===")
        experiment_results = self.padres.test_padres_api()
        
        # 2. Research related work
        print("\n=== Researching with Perplexity ===")
        perplexity_research_results = self.research_with_perplexity(research_query_content)
        
        # Extract the actual query sent to Perplexity for logging, if available
        # Assuming data structure from Perplexity API
        actual_query_sent_to_perplexity = research_query_content # Fallback
        if perplexity_research_results and 'model' in perplexity_research_results: # Check if it's a valid response
             # This is a placeholder as the provided perplexity call structure doesn't directly echo back the 'messages' part of request in response
             pass


        # 3. Combined analysis
        print("\n=== Combining Results ===")
        combined_result = {
            'experiment': experiment_results,
            'research': {
                'request': {'messages': [{'role': 'user', 'content': actual_query_sent_to_perplexity}]}, # Log what was asked
                'model': perplexity_research_results.get('model') if isinstance(perplexity_research_results, dict) else None,
                'choices': perplexity_research_results.get('choices') if isinstance(perplexity_research_results, dict) else None,
                'error': perplexity_research_results.get('error') if isinstance(perplexity_research_results, dict) else None,
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 4. Store in BigQuery
        print("\n=== Storing Results in BigQuery ===")
        storage_info = self.store_in_bigquery(combined_result, "padres_experiments_log")
        combined_result['bigquery_storage_info'] = storage_info
        
        print("\n=== Enhanced Experiment Complete ===")
        return combined_result

# Example Usage (assuming ANTHROPIC_API_KEY and PERPLEXITY_API_KEY are set as environment variables)
# And GOOGLE_APPLICATION_CREDENTIALS is set for BigQuery
if __name__ == "__main__":
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    anthropic_key_main = os.getenv("ANTHROPIC_API_KEY") # Get Anthropic key here
    gcp_project = os.getenv("GCP_PROJECT_ID")
    bq_dataset = "padres_research_data"

    if not perplexity_key:
        print("PERPLEXITY_API_KEY environment variable not found. Please set it to run the example.")
    elif not anthropic_key_main:
        print("ANTHROPIC_API_KEY environment variable not found. Please set it for Claude analysis.")
    elif not gcp_project:
        print("GCP_PROJECT_ID environment variable not found. Please set it for BigQuery integration.")
    else:
        pipeline = DirectResearchPipeline(
            perplexity_api_key=perplexity_key,
            anthropic_api_key=anthropic_key_main, # Pass it here
            gcp_project_id=gcp_project,
            bigquery_dataset_id=bq_dataset
        )
        
        # You might want to create the BigQuery dataset manually first if it doesn't exist
        # or add a method to create it in the DirectResearchPipeline class.
        # For this example, let's assume the dataset 'padres_research_data' exists in your GCP_PROJECT_ID.
        
        enhanced_results = pipeline.run_enhanced_experiment(
            research_query_content="What are the latest advancements in combining LLMs with physics simulators for spatial reasoning?"
        )
        
        output_filename = f"enhanced_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_filename, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str) # Use default=str for any non-serializable objects like TextBlock
        print(f"\nEnhanced experiment results saved to: {output_filename}")

        if enhanced_results.get('research', {}).get('error'):
            print("\n--- Perplexity Research Error ---")
            print(json.dumps(enhanced_results['research']['error'], indent=2))
            if enhanced_results['research'].get('details'):
                 print(json.dumps(enhanced_results['research']['details'], indent=2))


        if enhanced_results.get('bigquery_storage_info', {}).get('error'):
            print("\n--- BigQuery Storage Error ---")
            print(json.dumps(enhanced_results['bigquery_storage_info'], indent=2)) 