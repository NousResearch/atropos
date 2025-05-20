# enhanced_padres_perplexity.py
import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from run_single_padres_test import PadresTest

load_dotenv(override=True)

class SimplePadresResearch:
    def __init__(self):
        self.padres = PadresTest(use_llm=True)
        self.perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        
    def search_perplexity(self, query):
        """Direct Perplexity API call - simple and reliable"""
        if not self.perplexity_key:
            return "Error: Perplexity API key not found"
            
        try:
            headers = {
                "Authorization": f"Bearer {self.perplexity_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "sonar", # Trying a generally available online model
                "messages": [
                    {"role": "user", "content": query}
                ],
                "max_tokens": 1000
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status() # Raise an exception for HTTP errors
            
            return response.json()['choices'][0]['message']['content']
                
        except requests.exceptions.HTTPError as http_err:
            # It's good to include the response text for more detailed error info
            error_message = f"Error: {response.status_code}"
            if response.text:
                error_message += f" - {response.text}"
            return error_message
        except Exception as e:
            return f"Perplexity API error: {str(e)}"
    
    def run_research_experiment(self):
        """Complete research cycle: Padres -> Claude -> Perplexity -> Save"""
        
        print("=== Running Enhanced Research Experiment ===\n")
        
        # 1. Run spatial experiment
        print("1. Running Padres experiment...")
        padres_result = self.padres.test_padres_api()
        
        # 2. Extract Claude's analysis
        claude_analysis = padres_result.get('llm_analysis', 'Claude analysis not found.')
        print(f"2. Claude's analysis: {str(claude_analysis)[:200]}...\n")
        
        # 3. Research with Perplexity
        print("3. Researching with Perplexity...")
        research_query = f"Latest spatial reasoning AI research 2024 2025 LLM physics simulation"
        perplexity_result = self.search_perplexity(research_query)
        print(f"4. Perplexity research: {str(perplexity_result)[:200]}...\n")
        
        # 4. Combine results
        combined_result = {
            'timestamp': datetime.now().isoformat(),
            'padres_experiment': padres_result,
            'claude_analysis': claude_analysis,
            'perplexity_research': perplexity_result,
            'research_query': research_query
        }
        
        # 5. Save results
        filename = f"research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(combined_result, f, indent=2)
        
        print(f"5. Results saved to: {filename}")
        return combined_result

if __name__ == "__main__":
    researcher = SimplePadresResearch()
    results = researcher.run_research_experiment() 