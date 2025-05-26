import requests
import json
import os
import logging # Use logging instead of print for consistency
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger(__name__) # Setup logger for this module

# Optional imports for LLM integration
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class PadresTest:
    def __init__(self, use_llm=True, anthropic_api_key_override=None):
        self.base_url = os.getenv('PADRES_API_URL')
        if not self.base_url:
            logger.error("CRITICAL: PADRES_API_URL environment variable is not set. Padres API calls will fail.")
            # Or raise an error to prevent initialization if Padres is critical
            # raise ValueError("PADRES_API_URL must be set in the environment.")
            self.base_url = "http://invalid-url-due-to-missing-env-var" # Ensure it won't accidentally hit localhost
        else:
            logger.info(f"PadresTest initialized with base_url: {self.base_url}")

        # Anthropic related setup (will be skipped if use_llm=False from SimplePadresResearch)
        actual_anthropic_key = anthropic_api_key_override if anthropic_api_key_override else os.getenv('ANTHROPIC_API_KEY')
        # logger.debug(f"DEBUG (PadresTest init): ANTHROPIC_API_KEY_OVERRIDE: ...") # Keep debugs minimal or conditional
        # logger.debug(f"DEBUG (PadresTest init): ACTUAL_ANTHROPIC_KEY: ...")
        
        self.use_llm_internal = use_llm # Renamed to avoid confusion
        if self.use_llm_internal:
            if os.getenv('ANTHROPIC_API_KEY'): # Only attempt if key is present
                try:
                    from anthropic import Anthropic # Keep import local if only used here
                    self.anthropic = Anthropic(api_key=actual_anthropic_key)
                    logger.info("PadresTest: Anthropic client initialized (if use_llm_internal=True).")
                except Exception as e:
                    logger.warning(f"PadresTest: Failed to initialize Anthropic client: {e}")
                    self.use_llm_internal = False
            else:
                logger.warning("PadresTest: ANTHROPIC_API_KEY not found, internal LLM analysis will be skipped.")
                self.use_llm_internal = False

    def _default_error_results(self, error_type: str, message: str) -> dict:
        """Returns a structured dictionary for error states."""
        return {
            'status': {"api_status": error_type, "error_message": message},
            'setup': {"status": "ERROR", "message": message},
            'action': {"status": "ERROR", "message": message, "reward": 0, "done": False},
            'observation': message,
            'llm_analysis': f"LLM analysis skipped due to {error_type}: {message}"
        }

    def test_padres_api(self):
        logger.info(f"=== Testing Padres API at {self.base_url} ===")
        results = self._default_error_results("INITIAL_STATE", "Experiment not fully run.")

        if "invalid-url" in self.base_url or not self.base_url.startswith(("http://", "https://")):
            msg = f"Invalid PADRES_API_URL configured: {self.base_url}"
            logger.error(msg)
            return self._default_error_results("INVALID_URL", msg)

        try:
            logger.info(f"Attempting GET {self.base_url}/status")
            status_response = requests.get(f"{self.base_url}/status", timeout=10)
            status_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            results['status'] = status_response.json()
            logger.info(f"Padres API Status: {results['status']}")
            
            logger.info(f"Attempting POST {self.base_url}/setup_environment")
            setup_response = requests.post(f"{self.base_url}/setup_environment", timeout=15)
            setup_response.raise_for_status()
            results['setup'] = setup_response.json()
            logger.info(f"Padres Environment Setup: {results['setup']}")
            
            logger.info(f"Attempting POST {self.base_url}/execute_action")
            action_response = requests.post(f"{self.base_url}/execute_action", timeout=30) # Longer timeout for action
            action_response.raise_for_status()
            results['action'] = action_response.json()
            logger.info(f"Padres Action Result: {results['action']}")
            
            results['observation'] = self.format_observation_for_llm(results['action'])
            logger.info(f"Formatted Observation for LLM: {results['observation'][:500]}...") # Log snippet
            
            if self.use_llm_internal: # This is the internal Anthropic call, if enabled
                logger.info("PadresTest: Attempting internal LLM analysis (Anthropic)...")
                results['llm_analysis'] = self.get_llm_analysis(results['observation'])
                if isinstance(results['llm_analysis'], list) and len(results['llm_analysis']) > 0:
                    results['llm_analysis'] = results['llm_analysis'][0].text # Example for Anthropic SDK
                logger.info("PadresTest: Internal LLM Analysis complete.")
            else:
                logger.info("PadresTest: Skipping internal LLM analysis (use_llm_internal=False or client unavailable).")
                results['llm_analysis'] = "Internal LLM analysis skipped by PadresTest."
            
            return results 
            
        except requests.exceptions.ConnectionError as e:
            msg = f"ConnectionError: Could not connect to Padres API at {self.base_url}. Details: {e}"
            logger.error(msg)
            return self._default_error_results("PADRES_CONNECTION_ERROR", msg)
        except requests.exceptions.Timeout as e:
            msg = f"Timeout: Request to Padres API at {self.base_url} timed out. Details: {e}"
            logger.error(msg)
            return self._default_error_results("PADRES_TIMEOUT_ERROR", msg)
        except requests.exceptions.HTTPError as e:
            msg = f"HTTPError: Padres API request failed with status {e.response.status_code} for {e.request.url}. Response: {e.response.text[:200]}"
            logger.error(msg)
            return self._default_error_results("PADRES_HTTP_ERROR", msg)
        except json.JSONDecodeError as e:
            msg = f"JSONDecodeError: Failed to parse JSON response from Padres API at {self.base_url}. Details: {e}"
            logger.error(msg)
            return self._default_error_results("PADRES_JSON_DECODE_ERROR", msg)
        except Exception as e: # Catch-all for other unexpected errors
            msg = f"Unexpected error during Padres API interaction with {self.base_url}. Details: {e}"
            logger.error(msg, exc_info=True) # Log full traceback for unexpected errors
            return self._default_error_results("PADRES_UNEXPECTED_ERROR", msg)

    def format_observation_for_llm(self, action_data: dict):
        # ... (format_observation_for_llm method as before, ensure it handles action_data being potentially from an error state) ...
        # Safely access keys from action_data
        if not isinstance(action_data, dict): # Handle if action_data is not a dict due to earlier error
            logger.warning("format_observation_for_llm received non-dict action_data. Returning raw data.")
            return json.dumps(action_data) if action_data is not None else "No action data available."
        try:
            action_applied = action_data.get('action_applied', {})
            observation = action_data.get('observation', 'No specific observation text.')
            reward = action_data.get('reward', 0)
            done = action_data.get('done', False)
            full_outcome = action_data.get('full_outcome_debug', {})
            object_states = full_outcome.get('new_state_viz', [])
            formatted_observation = f"""Spatial Simulation Result:\n- Action performed: {action_applied.get('action_type', 'unknown')} on {action_applied.get('object_id', 'unknown')}\n- System observation: {observation}\n- Reward achieved: {reward}\n- Task completed: {done}\n- Current object positions:\n"""
            for obj in object_states:
                pos = obj.get('position', [0, 0, 0])
                formatted_observation += f"  * {obj.get('id', 'unknown_id')}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}\n"
            return formatted_observation.strip()
        except Exception as e:
            logger.error(f"Error formatting observation: {str(e)}", exc_info=True)
            return f"Error formatting observation. Raw action_data: {json.dumps(action_data) if isinstance(action_data, dict) else str(action_data)}"

    def get_llm_analysis(self, observation: str):
        # This method uses Anthropic. It will only be called if self.use_llm_internal is True
        # AND the Anthropic client was successfully initialized.
        if not self.use_llm_internal:
            return "Internal LLM analysis (Anthropic) was not enabled or client failed to init."
        # ... (rest of Anthropic get_llm_analysis method) ...
        logger.info("PadresTest: Calling internal Anthropic client for analysis.")
        try:
            from anthropic import Anthropic # Keep import local to this method if truly isolated
            # This assumes self.anthropic was initialized if self.use_llm_internal is True and key was present
            if not hasattr(self, 'anthropic') or not self.anthropic:
                 logger.warning("PadresTest: Anthropic client not available for get_llm_analysis even though use_llm_internal is True.")
                 return "Anthropic client not initialized."

            prompt = f"You are an AI research assistant... Simulation Data:\n{observation}\n..."
            response = self.anthropic.messages.create(
                model="claude-3-opus-20240229", # Or your preferred Claude model
                max_tokens=1000, temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content # Or response.completion, check Anthropic SDK
        except Exception as e:
            logger.error(f"PadresTest: Error getting internal LLM analysis (Anthropic): {e}", exc_info=True)
            return f"Error in internal Anthropic analysis: {str(e)}"

def main():
    # Parse command line arguments (could be added later)
    # use_llm = bool(os.getenv('ANTHROPIC_API_KEY')) # Original logic
    # For direct execution of this script, it will still rely on os.getenv inside PadresTest unless an override is passed.
    # This is fine as enhanced_research_without_mcp.py will now provide the override.
    
    # Create test instance
    tester = PadresTest(use_llm=True) # When run directly, PadresTest will use its original os.getenv logic for the key.
    
    # Run test
    results = tester.test_padres_api()
    
    # Save results (optional)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"padres_test_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main() 