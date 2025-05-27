#!/usr/bin/env python3
"""
Real Data Analysis for AMIEN
Analyzes actual Padres API experimental data to find genuine patterns
"""

import json
import glob
from typing import Dict, List, Any
from datetime import datetime

def load_real_experiments() -> List[Dict]:
    """Load all real experimental data from Padres API"""
    experiments = []
    
    # Load all research results files
    for filename in glob.glob("research_results_*.json"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if 'padres_experiment' in data:
                    experiments.append({
                        'filename': filename,
                        'timestamp': data.get('timestamp'),
                        'experiment': data['padres_experiment']
                    })
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return experiments

def extract_real_metrics(experiments: List[Dict]) -> Dict[str, Any]:
    """Extract real performance metrics from experiments"""
    metrics = {
        'total_experiments': len(experiments),
        'successful_tasks': 0,
        'rewards': [],
        'distances': [],
        'task_types': [],
        'object_interactions': [],
        'spatial_patterns': []
    }
    
    for exp in experiments:
        exp_data = exp['experiment']
        
        # Extract action data
        if 'action' in exp_data:
            action = exp_data['action']
            
            # Success rate
            if action.get('done', False):
                metrics['successful_tasks'] += 1
            
            # Rewards
            reward = action.get('reward', 0)
            if reward is not None:
                metrics['rewards'].append(reward)
            
            # Extract distance information
            if 'full_outcome_debug' in action:
                debug = action['full_outcome_debug']
                if 'observation' in debug:
                    obs = debug['observation']
                    # Parse distance from observation text
                    if 'Distance to ref:' in obs:
                        try:
                            dist_str = obs.split('Distance to ref:')[1].split('.')[0] + '.' + obs.split('Distance to ref:')[1].split('.')[1].split()[0]
                            distance = float(dist_str)
                            metrics['distances'].append(distance)
                        except:
                            pass
            
            # Extract object interaction patterns
            if 'action_applied' in action:
                applied = action['action_applied']
                metrics['object_interactions'].append({
                    'action_type': applied.get('action_type'),
                    'object_id': applied.get('object_id'),
                    'target_position': applied.get('target_position'),
                    'reward': reward
                })
    
    return metrics

def analyze_spatial_patterns(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze spatial reasoning patterns from real data"""
    analysis = {
        'success_rate': 0,
        'average_reward': 0,
        'average_distance': 0,
        'object_preferences': {},
        'action_effectiveness': {},
        'spatial_insights': []
    }
    
    if metrics['total_experiments'] > 0:
        analysis['success_rate'] = metrics['successful_tasks'] / metrics['total_experiments']
    
    if metrics['rewards']:
        analysis['average_reward'] = sum(metrics['rewards']) / len(metrics['rewards'])
    
    if metrics['distances']:
        analysis['average_distance'] = sum(metrics['distances']) / len(metrics['distances'])
    
    # Analyze object interaction patterns
    object_counts = {}
    action_rewards = {}
    
    for interaction in metrics['object_interactions']:
        obj_id = interaction.get('object_id', 'unknown')
        action_type = interaction.get('action_type', 'unknown')
        reward = interaction.get('reward', 0)
        
        # Count object interactions
        object_counts[obj_id] = object_counts.get(obj_id, 0) + 1
        
        # Track action effectiveness
        if action_type not in action_rewards:
            action_rewards[action_type] = []
        action_rewards[action_type].append(reward)
    
    analysis['object_preferences'] = object_counts
    
    # Calculate action effectiveness
    for action, rewards in action_rewards.items():
        if rewards:
            analysis['action_effectiveness'][action] = {
                'count': len(rewards),
                'average_reward': sum(rewards) / len(rewards),
                'success_rate': sum(1 for r in rewards if r > 0) / len(rewards)
            }
    
    # Generate insights
    if analysis['success_rate'] > 0.8:
        analysis['spatial_insights'].append("High task success rate indicates effective spatial reasoning")
    
    if analysis['average_distance'] < 0.3:
        analysis['spatial_insights'].append("Precise object positioning achieved")
    
    if 'red_cube' in object_counts and 'blue_sphere' in object_counts:
        analysis['spatial_insights'].append("Multi-object spatial reasoning demonstrated")
    
    return analysis

def generate_real_discovery_paper(metrics: Dict, analysis: Dict) -> str:
    """Generate a research paper from REAL experimental data"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    paper = f"""
# Real Spatial Reasoning Performance Analysis

**Generated from Actual AMIEN/Padres Experimental Data**  
**Date:** {timestamp}  
**Status:** Real Data Analysis

## Abstract

This paper presents an analysis of {metrics['total_experiments']} real spatial reasoning experiments conducted using the AMIEN/Padres API. Our analysis reveals actual performance patterns in VR spatial manipulation tasks, with a {analysis['success_rate']:.1%} task success rate and average positioning accuracy of {analysis['average_distance']:.3f} units.

## Real Experimental Data

### Dataset Overview
- **Total Experiments**: {metrics['total_experiments']}
- **Successful Tasks**: {metrics['successful_tasks']}
- **Success Rate**: {analysis['success_rate']:.1%}
- **Average Reward**: {analysis['average_reward']:.3f}
- **Average Distance Error**: {analysis['average_distance']:.3f} units

### Object Interaction Patterns
"""
    
    for obj, count in analysis['object_preferences'].items():
        paper += f"- **{obj}**: {count} interactions\n"
    
    paper += f"""

### Action Effectiveness Analysis
"""
    
    for action, stats in analysis['action_effectiveness'].items():
        paper += f"""
**{action}**:
- Attempts: {stats['count']}
- Average Reward: {stats['average_reward']:.3f}
- Success Rate: {stats['success_rate']:.1%}
"""
    
    paper += f"""

## Key Findings from Real Data

"""
    
    for i, insight in enumerate(analysis['spatial_insights'], 1):
        paper += f"{i}. {insight}\n"
    
    paper += f"""

## Implications

This analysis of real experimental data provides genuine insights into spatial reasoning performance, unlike synthetic simulations. The {analysis['success_rate']:.1%} success rate and {analysis['average_distance']:.3f} average distance error represent actual system capabilities.

## Limitations

- Limited dataset size ({metrics['total_experiments']} experiments)
- Single task type analyzed
- No demographic segmentation in current data

## Future Work

- Scale to 100+ real experiments
- Implement demographic tracking
- Connect evolutionary optimization to real API
- Validate synthetic discoveries against real performance

---

*This analysis is based on actual experimental data from the AMIEN/Padres spatial reasoning system.*
"""
    
    return paper

def main():
    """Analyze real experimental data and generate insights"""
    
    print("🔍 Loading real experimental data...")
    experiments = load_real_experiments()
    
    if not experiments:
        print("❌ No real experimental data found")
        return
    
    print(f"📊 Found {len(experiments)} real experiments")
    
    print("🧮 Extracting real metrics...")
    metrics = extract_real_metrics(experiments)
    
    print("🔬 Analyzing spatial patterns...")
    analysis = analyze_spatial_patterns(metrics)
    
    print("📝 Generating real discovery paper...")
    paper = generate_real_discovery_paper(metrics, analysis)
    
    # Save paper
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"real_spatial_analysis_{timestamp}.md"
    
    with open(filename, 'w') as f:
        f.write(paper)
    
    print(f"✅ Real analysis complete: {filename}")
    print(f"\n=== REAL DISCOVERIES ===")
    print(f"📊 Experiments Analyzed: {metrics['total_experiments']}")
    print(f"🎯 Success Rate: {analysis['success_rate']:.1%}")
    print(f"🎖️ Average Reward: {analysis['average_reward']:.3f}")
    print(f"📏 Average Distance: {analysis['average_distance']:.3f}")
    print(f"\n🔍 Real Insights:")
    for insight in analysis['spatial_insights']:
        print(f"• {insight}")

if __name__ == "__main__":
    main() 