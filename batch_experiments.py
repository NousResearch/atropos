import time
from run_single_padres_test import PadresTest


def run_experiment_batch(num_experiments=5):
    """Run multiple experiments and collect results"""
    tester = PadresTest(use_llm=True)
    
    all_results = []
    
    for i in range(num_experiments):
        print(f"\n{'='*50}")
        print(f"Running Experiment {i+1}/{num_experiments}")
        print(f"{'='*50}")
        
        results = tester.test_padres_api()
        results['experiment_id'] = i + 1
        all_results.append(results)
        
        # Brief pause between experiments
        time.sleep(2)
    
    # Save batch results
    import json
    from datetime import datetime
    
    batch_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(batch_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Batch results saved to: {batch_file}")
    return all_results


if __name__ == "__main__":
    results = run_experiment_batch(5) 