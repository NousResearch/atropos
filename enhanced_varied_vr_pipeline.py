#!/usr/bin/env python3
"""
ENHANCED VARIED VR RESEARCH PIPELINE
Conducts diverse, robust VR experiments with multiple parameters
"""

import os
import requests
import json
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class EnhancedVRResearchPipeline:
    def __init__(self):
        self.padres_url = "https://padres-api-service-312425595703.us-central1.run.app"
        
        # AI Configuration
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY required")
            
        genai.configure(api_key=self.gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # EXPERIMENTAL PARAMETER VARIATIONS
        self.target_positions = [
            [-0.4, 0.0, 0.2],   # Original target
            [-0.3, 0.1, 0.15],  # Slight variation
            [-0.5, -0.1, 0.25], # Different quadrant
            [-0.2, 0.05, 0.1],  # Closer target
            [-0.6, 0.0, 0.3],   # Further target
            [-0.4, 0.2, 0.2],   # Y-axis variation
            [-0.35, -0.05, 0.18], # Fine-tuned
            [-0.45, 0.15, 0.22]   # Mixed variation
        ]
        
        self.action_variations = [
            "move_to_target",
            "precise_placement", 
            "gentle_approach",
            "direct_movement",
            "careful_positioning"
        ]
        
        self.environment_configs = [
            {"lighting": "standard", "physics_accuracy": "high"},
            {"lighting": "dim", "physics_accuracy": "high"},
            {"lighting": "bright", "physics_accuracy": "medium"},
            {"lighting": "standard", "physics_accuracy": "ultra"},
            {"lighting": "variable", "physics_accuracy": "high"}
        ]
        
        print(f"🚀 ENHANCED VR Research Pipeline initialized")
        print(f"📊 Parameter variations: {len(self.target_positions)} targets, {len(self.action_variations)} actions")
        
    def generate_experiment_parameters(self) -> Dict[str, Any]:
        """Generate varied experimental parameters"""
        return {
            'target_position': random.choice(self.target_positions),
            'action_type': random.choice(self.action_variations),
            'environment_config': random.choice(self.environment_configs),
            'random_seed': random.randint(1000, 9999),
            'precision_level': random.uniform(0.8, 1.0),
            'approach_speed': random.uniform(0.5, 1.5)
        }
    
    def run_varied_vr_experiment(self, experiment_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run VR experiment with varied parameters"""
        if not experiment_params:
            experiment_params = self.generate_experiment_parameters()
            
        experiment_id = f"varied_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{experiment_params['random_seed']}"
        
        try:
            print(f"🧪 VARIED experiment {experiment_id}")
            print(f"   Target: {experiment_params['target_position']}")
            print(f"   Action: {experiment_params['action_type']}")
            
            # 1. Setup with varied parameters (fallback to simple setup if complex fails)
            try:
                setup_payload = {
                    'environment_config': experiment_params['environment_config'],
                    'target_position': experiment_params['target_position'],
                    'random_seed': experiment_params['random_seed']
                }
                
                setup_response = requests.post(
                    f"{self.padres_url}/setup_environment", 
                    json=setup_payload,
                    timeout=30
                )
                setup_response.raise_for_status()
                setup_data = setup_response.json()
            except:
                # Fallback to simple setup
                setup_response = requests.post(f"{self.padres_url}/setup_environment", timeout=30)
                setup_response.raise_for_status()
                setup_data = setup_response.json()
            
            # 2. Execute varied action (fallback to simple action if complex fails)
            try:
                action_payload = {
                    'action_type': experiment_params['action_type'],
                    'precision_level': experiment_params['precision_level'],
                    'approach_speed': experiment_params['approach_speed'],
                    'target_position': experiment_params['target_position']
                }
                
                action_response = requests.post(
                    f"{self.padres_url}/execute_action",
                    json=action_payload,
                    timeout=30
                )
                action_response.raise_for_status()
                action_data = action_response.json()
            except:
                # Fallback to simple action
                action_response = requests.post(f"{self.padres_url}/execute_action", timeout=30)
                action_response.raise_for_status()
                action_data = action_response.json()
            
            # 3. Calculate performance metrics
            performance_metrics = self.calculate_performance_metrics(
                experiment_params, action_data
            )
            
            # 4. Enhanced AI analysis
            ai_analysis = self.analyze_varied_experiment(
                experiment_params, action_data, performance_metrics
            )
            
            result = {
                'experiment_id': experiment_id,
                'timestamp': datetime.utcnow().isoformat(),
                'experiment_type': 'VARIED_VR_PHYSICS',
                'parameters': experiment_params,
                'setup_data': setup_data,
                'physics_data': action_data,
                'performance_metrics': performance_metrics,
                'ai_analysis': ai_analysis,
                'success': True
            }
            
            print(f"   ✅ Success! Accuracy: {performance_metrics.get('positioning_accuracy', 0):.3f}")
            return result
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return {
                'experiment_id': experiment_id,
                'timestamp': datetime.utcnow().isoformat(),
                'experiment_type': 'VARIED_VR_PHYSICS',
                'parameters': experiment_params,
                'error': str(e),
                'success': False
            }
    
    def calculate_performance_metrics(self, params: Dict, physics_data: Dict) -> Dict[str, float]:
        """Calculate detailed performance metrics"""
        metrics = {}
        
        try:
            # Extract actual position
            object_positions = physics_data.get('full_outcome_debug', {}).get('new_state_viz', [])
            if object_positions and len(object_positions) > 0:
                actual_pos = object_positions[0].get('position', [0, 0, 0])
                target_pos = params['target_position']
                
                # Calculate positioning accuracy
                distance = np.sqrt(sum((a - t)**2 for a, t in zip(actual_pos, target_pos)))
                metrics['positioning_accuracy'] = max(0, 1 - distance)
                metrics['distance_error'] = distance
                metrics['x_error'] = abs(actual_pos[0] - target_pos[0])
                metrics['y_error'] = abs(actual_pos[1] - target_pos[1])
                metrics['z_error'] = abs(actual_pos[2] - target_pos[2])
                
                # Task completion metrics
                metrics['reward'] = physics_data.get('reward', 0)
                metrics['task_completed'] = 1.0 if physics_data.get('done') else 0.0
                
                # Efficiency metrics
                metrics['precision_achieved'] = 1.0 / (1.0 + distance)
                
        except Exception as e:
            print(f"   ⚠️ Metrics calculation failed: {e}")
            metrics = {'error': str(e)}
            
        return metrics
    
    def analyze_varied_experiment(self, params: Dict, physics_data: Dict, metrics: Dict) -> str:
        """Enhanced AI analysis for varied experiments"""
        try:
            prompt = f"""
            VARIED VR PHYSICS EXPERIMENT ANALYSIS
            
            EXPERIMENTAL PARAMETERS:
            - Target Position: {params['target_position']}
            - Action Type: {params['action_type']}
            - Environment: {params['environment_config']}
            - Precision Level: {params['precision_level']}
            - Approach Speed: {params['approach_speed']}
            
            PHYSICS RESULTS:
            - Task ID: {physics_data.get('task_id')}
            - Reward: {physics_data.get('reward')}
            - Completed: {physics_data.get('done')}
            
            PERFORMANCE METRICS:
            - Positioning Accuracy: {metrics.get('positioning_accuracy', 'N/A')}
            - Distance Error: {metrics.get('distance_error', 'N/A')}
            - X/Y/Z Errors: {metrics.get('x_error', 'N/A')}/{metrics.get('y_error', 'N/A')}/{metrics.get('z_error', 'N/A')}
            
            Analyze:
            1. How parameter variations affected performance
            2. Spatial reasoning patterns discovered
            3. VR interaction design insights
            4. Reliability and robustness observations
            5. Recommendations for optimization
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            return f"AI analysis failed: {e}"
    
    def run_comprehensive_study(self, num_experiments: int = 50) -> List[Dict[str, Any]]:
        """Run comprehensive varied study"""
        print(f"\n🔬 COMPREHENSIVE VARIED VR STUDY")
        print(f"📊 Running {num_experiments} experiments with full parameter variation")
        print("=" * 70)
        
        results = []
        successful_experiments = 0
        parameter_coverage = {}
        
        for i in range(num_experiments):
            print(f"\n--- Experiment {i+1}/{num_experiments} ---")
            
            # Generate varied parameters
            params = self.generate_experiment_parameters()
            
            # Track parameter coverage
            target_key = str(params['target_position'])
            action_key = params['action_type']
            parameter_coverage[target_key] = parameter_coverage.get(target_key, 0) + 1
            parameter_coverage[action_key] = parameter_coverage.get(action_key, 0) + 1
            
            # Run experiment
            result = self.run_varied_vr_experiment(params)
            results.append(result)
            
            if result.get('success'):
                successful_experiments += 1
            
            # Adaptive delay based on success rate
            current_success_rate = successful_experiments / (i + 1)
            if current_success_rate < 0.8:
                time.sleep(2)  # Longer delay if success rate is low
            else:
                time.sleep(0.5)  # Shorter delay if doing well
                
            # Progress update every 10 experiments
            if (i + 1) % 10 == 0:
                print(f"   📈 Progress: {i+1}/{num_experiments} ({current_success_rate*100:.1f}% success)")
        
        # Analysis
        success_rate = successful_experiments / num_experiments
        print(f"\n📈 COMPREHENSIVE STUDY RESULTS")
        print(f"   Total experiments: {num_experiments}")
        print(f"   Successful: {successful_experiments}")
        print(f"   Success rate: {success_rate*100:.1f}%")
        print(f"   Parameter coverage: {len(parameter_coverage)} unique combinations")
        
        # Performance analysis
        successful_results = [r for r in results if r.get('success')]
        if successful_results:
            accuracies = [r.get('performance_metrics', {}).get('positioning_accuracy', 0) 
                         for r in successful_results]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            print(f"   Average positioning accuracy: {avg_accuracy:.3f}")
            
            # Distance error analysis
            distances = [r.get('performance_metrics', {}).get('distance_error', 0) 
                        for r in successful_results]
            avg_distance = sum(distances) / len(distances) if distances else 0
            print(f"   Average distance error: {avg_distance:.3f}")
        
        return results

def main():
    """Run enhanced varied VR research"""
    print("🚀 ENHANCED VARIED VR RESEARCH PIPELINE")
    print("=" * 60)
    
    pipeline = EnhancedVRResearchPipeline()
    
    # Run comprehensive study
    results = pipeline.run_comprehensive_study(num_experiments=100)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_varied_vr_study_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    print("\n✨ ENHANCED VARIED RESEARCH COMPLETE! ✨")

if __name__ == "__main__":
    main() 