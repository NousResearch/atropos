# enhanced_padres_perplexity.py
import json
import logging
import os
from datetime import datetime

import google.generativeai as genai
import requests
from dotenv import load_dotenv

# Assuming run_single_padres_test.py is in the same directory or accessible via PYTHONPATH
from run_single_padres_test import (  # This class uses Anthropic, will need changes if we fully remove Anthropic
    PadresTest,
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


class SimplePadresResearch:
    def __init__(self):
        logger.info("Initializing SimplePadresResearch with Gemini...")
        # PadresTest might still try to init Anthropic client based on ANTHROPIC_API_KEY.
        # This needs careful review if PadresTest itself calls Claude directly.
        # For now, assuming primary LLM calls are made from this class.
        self.padres = PadresTest(
            use_llm=False
        )  # Set use_llm=False if PadresTest only uses LLM for its own analysis
        # and we want this class to be the sole LLM interactor.
        # If PadresTest.test_padres_api() is expected to return 'llm_analysis'
        # from its own LLM call, this needs more thought.

        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.gemini_api_key = os.getenv(
            "GEMINI_API_KEY"
        )  # Expect GEMINI_API_KEY in .env or environment

        if not self.gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY not found in environment. Gemini calls will fail."
            )
            # Depending on strictness, could raise an error:
            # raise ValueError("GEMINI_API_KEY is required for SimplePadresResearch.")
        else:
            try:
                genai.configure(api_key=self.gemini_api_key)
                # Initialize a specific model - let's use gemini-1.5-flash by default for text generation
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
                logger.info(
                    "Gemini client configured and model initialized (gemini-1.5-flash-latest)."
                )
            except Exception as e:
                logger.error(
                    f"Failed to configure Gemini client or model: {e}", exc_info=True
                )
                self.gemini_model = None  # Ensure it's None if init fails

        if not self.perplexity_key:
            logger.warning(
                "PERPLEXITY_API_KEY not found. Perplexity searches will fail."
            )
        logger.info("SimplePadresResearch initialization complete.")

    def call_llm_for_text(self, prompt_text: str) -> str:
        """Calls the configured LLM (now Gemini) to generate text based on a prompt."""
        if not self.gemini_model:
            logger.error("Gemini model not initialized. Cannot generate text.")
            return "[Error: Gemini model not available]"

        logger.info(
            f"Sending prompt to Gemini (model: gemini-1.5-flash-latest). Prompt length: {len(prompt_text)}"
        )
        try:
            # For simple text prompts, generate_content is used.
            response = self.gemini_model.generate_content(prompt_text)
            # Accessing the text part of the response:
            # Depending on the Gemini API version and response structure, you might need to adjust this.
            # Typically, response.text or iterating through response.parts if it's a multi-part response.
            if response.parts:
                # Assuming the first part contains the text if it's a simple text response
                generated_text = "".join(
                    part.text for part in response.parts if hasattr(part, "text")
                )
            elif hasattr(response, "text") and response.text:
                generated_text = response.text
            else:  # Fallback for unexpected response structure
                logger.warning(
                    f"Gemini response structure not as expected. Full response: {response}"
                )
                # Attempt to find text in candidates if available
                if hasattr(response, "candidates") and response.candidates:
                    for candidate in response.candidates:
                        if (
                            hasattr(candidate, "content")
                            and hasattr(candidate.content, "parts")
                            and candidate.content.parts
                        ):
                            for part in candidate.content.parts:
                                if hasattr(part, "text"):
                                    generated_text += part.text
                            if generated_text:
                                break  # Take first available text
            if not generated_text:
                logger.warning(
                    "No text found in Gemini response parts or direct .text attribute."
                )
            logger.info(
                f"Received response from Gemini. Response length: {len(generated_text)}"
            )
            return generated_text
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}", exc_info=True)
            return f"[Error interacting with Gemini: {e}]"

    def search_perplexity(self, query):
        """Direct Perplexity API call - simple and reliable"""
        if not self.perplexity_key:
            logger.error("Perplexity API key not found.")
            return "Error: Perplexity API key not found"

        logger.info(f"Querying Perplexity for: {query[:100]}...")
        try:
            headers = {
                "Authorization": f"Bearer {self.perplexity_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": "sonar",  # Trying a generally available online model
                "messages": [{"role": "user", "content": query}],
                "max_tokens": 1000,
            }

            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data,
                timeout=60,  # Increased timeout slightly for Perplexity
            )
            response.raise_for_status()  # Raise an exception for HTTP errors

            generated_text = response.json()["choices"][0]["message"]["content"]
            logger.info("Received response from Perplexity.")
            return generated_text

        except requests.exceptions.HTTPError as http_err:
            error_message = f"Perplexity HTTP Error: {http_err.response.status_code}"
            if (
                http_err.response
                and hasattr(http_err.response, "text")
                and http_err.response.text
            ):
                error_message += f" - {http_err.response.text}"
            logger.error(error_message, exc_info=True)
            return error_message
        except Exception as e:
            logger.error(f"Perplexity API error: {str(e)}", exc_info=True)
            return f"Perplexity API error: {str(e)}"

    def run_research_experiment(self):
        """Runs a single experiment: Padres -> LLM (Gemini) for analysis."""
        logger.info("=== Running Single Research Experiment (Padres -> Gemini) ===")

        logger.info("1. Running Padres experiment...")
        padres_result_raw = self.padres.test_padres_api()

        observation_for_llm = padres_result_raw.get(
            "observation", "No observation provided by Padres API."
        )
        # Check if padres_result_raw indicates an error itself, and adjust observation if so
        if isinstance(padres_result_raw.get("status"), dict) and padres_result_raw[
            "status"
        ].get("api_status", "").endswith("ERROR"):
            logger.warning(
                f"Padres API call resulted in an error state: {padres_result_raw['status']}. Observation for LLM will be this error."
            )
            observation_for_llm = f"Padres API Error: {padres_result_raw['status'].get('error_message', json.dumps(padres_result_raw['status']))}"
        elif not observation_for_llm and isinstance(padres_result_raw, dict):
            observation_for_llm = json.dumps(
                padres_result_raw
            )  # Fallback if 'observation' key is missing

        llm_analysis_prompt = f"Analyze the following spatial simulation data and provide insights. Simulation Data:\n{observation_for_llm}"
        logger.info("2. Analyzing with LLM (Gemini)...")
        llm_generated_analysis = self.call_llm_for_text(llm_analysis_prompt)
        # logger.info(f"LLM (Gemini) analysis (first 200 chars): {str(llm_generated_analysis)[:200]}...") # Logged in call_llm_for_text

        # Determine padres_success based on the action part of padres_result_raw
        action_result = padres_result_raw.get("action", {})
        is_padres_success = False
        if isinstance(action_result, dict):
            # Assuming Padres API returns a clear success indicator in action['status'] or uses 'done' and high 'reward'
            if action_result.get("status") == "SUCCESS":  # Ideal case
                is_padres_success = True
            elif (
                action_result.get("done") == True
                and isinstance(action_result.get("reward"), (int, float))
                and action_result.get("reward", 0) > 0
            ):  # Fallback logic
                is_padres_success = True
            # Check if any top-level status indicates an error explicitly
            elif isinstance(
                padres_result_raw.get("status"), dict
            ) and padres_result_raw["status"].get("api_status", "").endswith("ERROR"):
                is_padres_success = False

        current_timestamp = datetime.utcnow().isoformat() + "Z"
        experiment_core_data = {
            "experiment_id": current_timestamp,
            "timestamp": current_timestamp,
            "padres_api_response": padres_result_raw,  # Contains full details from PadresTest, including its own status, setup, action objects
            "padres_success": is_padres_success,
            "score": (
                action_result.get("reward", 0) if isinstance(action_result, dict) else 0
            ),
            "distance": (
                action_result.get("distance", 0)
                if isinstance(action_result, dict)
                else 0
            ),
            "task_completed": (
                action_result.get("done", False)
                if isinstance(action_result, dict)
                else False
            ),
            "llm_analysis": llm_generated_analysis,
        }
        logger.info(
            f"Single experiment (Padres -> Gemini) processed. Experiment ID: {current_timestamp}, Padres Success: {is_padres_success}"
        )
        return experiment_core_data


# Main block for local testing (if needed)
if __name__ == "__main__":
    load_dotenv()  # Ensure .env is loaded
    logger.info("Running SimplePadresResearch with Gemini locally (direct test)...")
    if not os.getenv("GEMINI_API_KEY"):
        print(
            "Error: GEMINI_API_KEY not found in environment. Please set it in your .env file."
        )
    # elif not os.getenv("PERPLEXITY_API_KEY"): # Not needed for run_research_experiment now
    #     print("Error: PERPLEXITY_API_KEY not found in environment. Please set it in your .env file.")
    elif not os.getenv("PADRES_API_URL"):  # Crucial for PadresTest
        print("Error: PADRES_API_URL not found in environment. Please set it.")
    else:
        researcher = SimplePadresResearch()
        results = researcher.run_research_experiment()
        logger.info("--- Local Run of single experiment (Padres->Gemini) Complete ---")
        logger.info(json.dumps(results, indent=2))

        # Test perplexity search separately if needed
        if os.getenv("PERPLEXITY_API_KEY"):
            logger.info("\n--- Testing Perplexity Search Separately ---")
            pq_result = researcher.search_perplexity("Latest in AI for robotics")
            logger.info(f"Perplexity Result: {pq_result[:500]}...")
        else:
            logger.warning(
                "PERPLEXITY_API_KEY not set, skipping separate Perplexity test."
            )
