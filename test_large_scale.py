#!/usr/bin/env python3
"""
Test script for large-scale VR study
"""

import requests
import json
import time
from datetime import datetime

def test_api():
    """Test API connectivity"""
    url = "https://padres-api-service-312425595703.us-central1.run.app"
    
    print("🧪 Testing API connectivity...")
    
    try:
        # Test status
        response = requests.get(f"{url}/status", timeout=10)
        print(f"Status check: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test setup
        setup_response = requests.post(f"{url}/setup_environment", timeout=10)
        print(f"Setup: {setup_response.status_code}")
        setup_data = setup_response.json()
        print(f"Setup data: {setup_data}")
        
        # Test action
        action_response = requests.post(f"{url}/execute_action", timeout=10)
        print(f"Action: {action_response.status_code}")
        action_data = action_response.json()
        print(f"Action data keys: {list(action_data.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def run_small_batch(num_experiments=10):
    """Run small batch of experiments"""
    print(f"\n🔬 Running {num_experiments} test experiments...")
    
    url = "https://padres-api-service-312425595703.us-central1.run.app"
    results = []
    
    for i in range(num_experiments):
        print(f"Experiment {i+1}/{num_experiments}")
        
        try:
            # Setup
            setup_response = requests.post(f"{url}/setup_environment", timeout=15)
            setup_response.raise_for_status()
            
            # Action
            action_response = requests.post(f"{url}/execute_action", timeout=15)
            action_response.raise_for_status()
            action_data = action_response.json()
            
            # Extract position for accuracy calculation
            object_positions = action_data.get('full_outcome_debug', {}).get('new_state_viz', [])
            if object_positions:
                actual_pos = object_positions[0].get('position', [0, 0, 0])
                target_pos = [-0.4, 0.0, 0.2]  # Default target
                
                distance = sum((a - t)**2 for a, t in zip(actual_pos, target_pos))**0.5
                accuracy = max(0, 1 - distance)
                
                result = {
                    'experiment_id': f"test_{i+1}",
                    'success': True,
                    'accuracy': accuracy,
                    'distance_error': distance,
                    'actual_position': actual_pos,
                    'target_position': target_pos
                }
                
                print(f"  ✅ Success! Accuracy: {accuracy:.3f}")
            else:
                result = {'experiment_id': f"test_{i+1}", 'success': False, 'error': 'No position data'}
                print(f"  ❌ No position data")
            
            results.append(result)
            time.sleep(0.5)  # Small delay
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({'experiment_id': f"test_{i+1}", 'success': False, 'error': str(e)})
    
    # Summary
    successful = [r for r in results if r.get('success')]
    success_rate = len(successful) / len(results)
    
    print(f"\n📊 Test Results:")
    print(f"  Success rate: {success_rate*100:.1f}%")
    
    if successful:
        avg_accuracy = sum(r.get('accuracy', 0) for r in successful) / len(successful)
        print(f"  Average accuracy: {avg_accuracy:.3f}")
    
    return results

def main():
    print("🚀 LARGE-SCALE VR STUDY - TEST MODE")
    print("=" * 50)
    
    # Test API first
    if not test_api():
        print("❌ API test failed - cannot proceed")
        return
    
    print("\n✅ API test passed!")
    
    # Run small batch
    results = run_small_batch(10)
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Test results saved to: {filename}")
    print("✨ Test complete!")

if __name__ == "__main__":
    main() 