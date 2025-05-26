#!/usr/bin/env python3
"""
REAL VR RESEARCH PIPELINE
Connects to actual Padres API for real physics simulation experiments
"""

import os
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class RealVRResearchPipeline:
    def __init__(self):
        # REAL API ENDPOINTS
        self.padres_url = "https://padres-api-service-312425595703.us-central1.run.app"
        
        # REAL AI SERVICES
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY required for real AI analysis")
            
        # Initialize Gemini
        genai.configure(api_key=self.gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        print(f"🚀 REAL VR Research Pipeline initialized")
        print(f"📡 Padres API: {self.padres_url}")
        print(f"🤖 Gemini AI: Configured")
        
    def run_real_vr_experiment(self) -> Dict[str, Any]:
        """Run actual VR experiment with real physics simulation"""
        experiment_id = datetime.utcnow().isoformat()
        
        try:
            print(f"🧪 Starting REAL VR experiment {experiment_id}")
            
            # 1. Setup real VR environment
            print("  📋 Setting up real VR environment...")
            setup_response = requests.post(f"{self.padres_url}/setup_environment", timeout=30)
            setup_response.raise_for_status()
            setup_data = setup_response.json()
            
            # 2. Execute real VR action
            print("  🎮 Executing real VR action...")
            action_response = requests.post(f"{self.padres_url}/execute_action", timeout=30)
            action_response.raise_for_status()
            action_data = action_response.json()
            
            # 3. Extract real physics data
            real_physics_data = {
                'task_id': action_data.get('task_id'),
                'action_applied': action_data.get('action_applied'),
                'observation': action_data.get('observation'),
                'reward': action_data.get('reward'),
                'done': action_data.get('done'),
                'object_positions': action_data.get('full_outcome_debug', {}).get('new_state_viz', [])
            }
            
            # 4. Real AI analysis
            print("  🧠 Analyzing with real AI...")
            ai_analysis = self.analyze_with_real_ai(real_physics_data)
            
            result = {
                'experiment_id': experiment_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data_source': 'REAL_VR_PHYSICS_SIMULATION',
                'setup_data': setup_data,
                'real_physics_data': real_physics_data,
                'ai_analysis': ai_analysis,
                'success': True
            }
            
            print(f"  ✅ REAL experiment completed successfully!")
            print(f"     Reward: {real_physics_data.get('reward')}")
            print(f"     Task completed: {real_physics_data.get('done')}")
            
            return result
            
        except Exception as e:
            print(f"  ❌ REAL experiment failed: {e}")
            return {
                'experiment_id': experiment_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data_source': 'REAL_VR_PHYSICS_SIMULATION',
                'error': str(e),
                'success': False
            }
    
    def analyze_with_real_ai(self, physics_data: Dict[str, Any]) -> str:
        """Analyze real physics data with real AI"""
        try:
            prompt = f"""
            You are analyzing REAL VR physics simulation data from PyBullet.
            
            REAL EXPERIMENTAL DATA:
            - Task ID: {physics_data.get('task_id')}
            - Action: {physics_data.get('action_applied')}
            - Observation: {physics_data.get('observation')}
            - Reward: {physics_data.get('reward')}
            - Task Completed: {physics_data.get('done')}
            - Object Positions: {physics_data.get('object_positions')}
            
            Provide a scientific analysis of:
            1. Physics accuracy and realism
            2. Spatial reasoning insights
            3. VR affordance discoveries
            4. Potential research implications
            
            Focus on what this REAL data tells us about VR interaction design.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            return f"AI analysis failed: {e}"
    
    def run_real_research_batch(self, num_experiments: int = 10) -> List[Dict[str, Any]]:
        """Run batch of real VR experiments"""
        print(f"\n🔬 Starting REAL research batch: {num_experiments} experiments")
        
        results = []
        successful_experiments = 0
        
        for i in range(num_experiments):
            print(f"\n--- Experiment {i+1}/{num_experiments} ---")
            
            # Run real experiment
            result = self.run_real_vr_experiment()
            results.append(result)
            
            if result.get('success'):
                successful_experiments += 1
            
            # Brief pause between experiments
            time.sleep(1)
        
        print(f"\n📊 REAL Research Batch Complete!")
        print(f"   Total experiments: {num_experiments}")
        print(f"   Successful: {successful_experiments}")
        print(f"   Success rate: {successful_experiments/num_experiments*100:.1f}%")
        
        return results
    
    def save_real_research_data(self, results: List[Dict[str, Any]], filename: str = None):
        """Save real research data to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_vr_research_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Real research data saved to: {filename}")
        return filename

def main():
    """Run real VR research experiments"""
    print("🚀 REAL VR RESEARCH PIPELINE")
    print("=" * 50)
    
    # Initialize real pipeline
    pipeline = RealVRResearchPipeline()
    
    # Run real experiments
    results = pipeline.run_real_research_batch(num_experiments=5)
    
    # Save real data
    filename = pipeline.save_real_research_data(results)
    
    # Summary
    successful = sum(1 for r in results if r.get('success'))
    print(f"\n🎯 REAL RESEARCH SUMMARY")
    print(f"   Experiments completed: {len(results)}")
    print(f"   Successful experiments: {successful}")
    print(f"   Data file: {filename}")
    print(f"   API endpoint: {pipeline.padres_url}")
    
    print("\n✨ REAL SCIENCE ACHIEVED! ✨")

if __name__ == "__main__":
    main() 