import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Optional imports for LLM integration
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class PadresTest:
    def __init__(self, base_url="http://localhost:8088", use_llm=True, anthropic_api_key_override=None):
        self.base_url = base_url
        
        # Use override if provided, otherwise try to get from env
        actual_anthropic_key = anthropic_api_key_override if anthropic_api_key_override else os.getenv('ANTHROPIC_API_KEY')
        
        print(f"DEBUG (PadresTest init): ANTHROPIC_API_KEY_OVERRIDE (first 5, last 5): {anthropic_api_key_override[:5] if anthropic_api_key_override else '[NOT PROVIDED]'}...{anthropic_api_key_override[-5:] if anthropic_api_key_override and len(anthropic_api_key_override) > 10 else '[NOT PROVIDED OR TOO SHORT]'}")
        print(f"DEBUG (PadresTest init): ACTUAL_ANTHROPIC_KEY being used (first 5, last 5): {actual_anthropic_key[:5] if actual_anthropic_key else '[KEY NOT FOUND]'}...{actual_anthropic_key[-5:] if actual_anthropic_key and len(actual_anthropic_key) > 10 else '[KEY INVALID OR TOO SHORT]'}")


        self.use_llm = use_llm and ANTHROPIC_AVAILABLE and actual_anthropic_key
        if self.use_llm:
            try:
                self.anthropic = Anthropic(api_key=actual_anthropic_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Anthropic client: {e}")
                self.use_llm = False

    def test_padres_api(self):
        """Run a complete test of the Padres API with optional LLM analysis"""
        results = {
            'status': None,
            'setup': None,
            'action': None,
            'observation': None,
            'llm_analysis': None
        }
        
        try:
            # 1. Check status
            print("\n=== Testing Padres API ===")
            status_response = requests.get(f"{self.base_url}/status")
            results['status'] = status_response.json()
            print(f"Status: {json.dumps(results['status'], indent=2)}")
            
            # 2. Setup environment
            setup_response = requests.post(f"{self.base_url}/setup_environment")
            results['setup'] = setup_response.json()
            print(f"\nEnvironment Setup: {json.dumps(results['setup'], indent=2)}")
            
            # 3. Execute action
            action_response = requests.post(f"{self.base_url}/execute_action")
            results['action'] = action_response.json()
            print(f"\nAction Result: {json.dumps(results['action'], indent=2)}")
            
            # 4. Format observation for LLM
            results['observation'] = self.format_observation_for_llm(results['action'])
            print(f"\nFormatted for LLM: {results['observation']}")
            
            # 5. Get LLM analysis if enabled
            if self.use_llm:
                results['llm_analysis'] = self.get_llm_analysis(results['observation'])
                print(f"\nLLM Analysis: {results['llm_analysis']}")
                # Ensure the LLM analysis is a string for JSON serialization
                if isinstance(results['llm_analysis'], list) and len(results['llm_analysis']) > 0:
                    results['llm_analysis'] = results['llm_analysis'][0].text
            else:
                print("\nSkipping LLM analysis (not enabled or not available)")
            
            return results
            
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to Padres API at {self.base_url}")
            print("Please ensure the API server is running")
            return results
        except Exception as e:
            print(f"Error during test execution: {str(e)}")
            return results

    def format_observation_for_llm(self, action_data):
        """Convert Padres API response into LLM-friendly text"""
        try:
            # Extract relevant information from the action data
            action_applied = action_data.get('action_applied', {})
            observation = action_data.get('observation', '')
            reward = action_data.get('reward', 0)
            done = action_data.get('done', False)
            
            # Extract object states from the full_outcome_debug
            full_outcome = action_data.get('full_outcome_debug', {})
            object_states = full_outcome.get('new_state_viz', [])
            
            # Format the observation in a more natural language way
            formatted_observation = f"""
Spatial Simulation Result:
- Action performed: {action_applied.get('action_type', 'unknown')} on {action_applied.get('object_id', 'unknown')}
- System observation: {observation}
- Reward achieved: {reward}
- Task completed: {done}
- Current object positions:
"""
            # Add object positions in a readable format
            for obj in object_states:
                pos = obj.get('position', [0, 0, 0])
                formatted_observation += f"  * {obj['id']}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}\n"
                
            return formatted_observation.strip()
        except Exception as e:
            return f"Error formatting observation: {str(e)}\nRaw data: {json.dumps(action_data, indent=2)}"

    def get_llm_analysis(self, observation):
        """Get LLM's analysis of the simulation result"""
        if not self.use_llm:
            return "LLM analysis not enabled"
            
        try:
            # Prepare the prompt
            prompt = f"""You are an AI research assistant analyzing spatial simulation results. 
Please analyze the following simulation outcome and provide insights about:
1. Whether the task was successful
2. The efficiency of the movement
3. Suggestions for improvement

Simulation Data:
{observation}

Please provide your analysis in a clear, structured format."""

            # Get LLM response
            response = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system="You are an AI research assistant specializing in analyzing spatial simulations and robotic movements.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return response.content
        except Exception as e:
            return f"Error getting LLM analysis: {str(e)}"

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