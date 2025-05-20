import json
import glob
from datetime import datetime


def analyze_valid_results():
    """Analyze only valid, complete experiment results"""
    
    # Focus on batch results which are more reliable
    batch_files = glob.glob("batch_results_*.json")
    
    if not batch_files:
        print("No batch result files found. Run batch_experiments.py first!")
        return
    
    print(f"Found {len(batch_files)} batch result files")
    
    all_experiments = []
    for file in batch_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                all_experiments.extend(data)
                print(f"✅ Loaded {len(data)} experiments from {file}")
        except json.JSONDecodeError as e:
            print(f"❌ Skipping corrupted file {file}: {e}")
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
    
    if not all_experiments:
        print("No valid experiments found!")
        return
    
    print(f"\n📊 Analyzing {len(all_experiments)} total experiments...\n")
    
    # Analyze results
    successful_experiments = 0
    total_reward = 0
    completed_tasks = 0
    claude_analyses = []
    
    for i, exp in enumerate(all_experiments):
        action_data = exp.get('action', {})
        
        # Check for successful actions
        reward = action_data.get('reward', 0)
        done = action_data.get('done', False)
        
        if reward > 0:
            successful_experiments += 1
            total_reward += reward
        
        if done:
            completed_tasks += 1
        
        # Collect Claude analyses
        llm_analysis = exp.get('llm_analysis')
        if llm_analysis:
            # Convert TextBlock to string if needed
            if hasattr(llm_analysis, 'text'):
                claude_analyses.append(llm_analysis.text)
            elif isinstance(llm_analysis, str):
                claude_analyses.append(llm_analysis)
            elif isinstance(llm_analysis, list) and len(llm_analysis) > 0:
                if hasattr(llm_analysis[0], 'text'):
                    claude_analyses.append(llm_analysis[0].text)
    
    # Print summary statistics
    print("="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"📈 Total Experiments: {len(all_experiments)}")
    print(f"✅ Successful Actions: {successful_experiments}/{len(all_experiments)} ({successful_experiments/len(all_experiments)*100:.1f}%)")
    print(f"🏆 Completed Tasks: {completed_tasks}/{len(all_experiments)} ({completed_tasks/len(all_experiments)*100:.1f}%)")
    print(f"⭐ Average Reward: {total_reward/len(all_experiments):.3f}")
    print(f"🤖 Claude Analyses: {len(claude_analyses)}")
    
    # Show sample Claude insights
    if claude_analyses:
        print("\n" + "="*60)
        print("SAMPLE CLAUDE INSIGHTS")
        print("="*60)
        for i, analysis in enumerate(claude_analyses[:3]):
            print(f"\n--- Experiment {i+1} Analysis ---")
            # Show first 300 characters of analysis
            preview = analysis[:300].replace('\n', ' ')
            print(f"{preview}...")
    
    # Save clean summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(all_experiments),
        'successful_experiments': successful_experiments,
        'completed_tasks': completed_tasks,
        'average_reward': total_reward/len(all_experiments),
        'claude_analyses_count': len(claude_analyses),
        'sample_analyses': claude_analyses[:3] if claude_analyses else []
    }
    
    summary_file = f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    return summary


if __name__ == "__main__":
    analyze_valid_results() 