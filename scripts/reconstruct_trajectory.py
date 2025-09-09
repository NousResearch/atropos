#!/usr/bin/env python3
"""
Reconstruct the full Diplomacy trajectory from individual samples.

Takes the 1,178 samples and combines them into a single trajectory file
with alternating user/assistant messages in chronological order.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def extract_interaction_number(system_prompt: str) -> int:
    """Extract the interaction number from system prompt."""
    # Look for "This is interaction X of 1178"
    if "interaction" in system_prompt and "of 1178" in system_prompt:
        parts = system_prompt.split("interaction")[1].split("of")[0].strip()
        try:
            return int(parts)
        except:
            return -1
    return -1


def reconstruct_trajectory(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Reconstruct the full trajectory from individual samples.
    
    Returns stats about the reconstruction.
    """
    stats = {
        'total_samples': 0,
        'total_messages': 0,
        'phases': set(),
        'interaction_range': {'min': float('inf'), 'max': 0}
    }
    
    # First, load all samples and sort by interaction number
    samples = []
    
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Extract interaction number from system prompt
            system_prompt = data['messages'][0]['content']
            interaction_num = extract_interaction_number(system_prompt)
            
            # Extract phase info from user message
            user_content = data['messages'][1]['content'] if len(data['messages']) > 1 else ''
            phase = 'unknown'
            year = 'unknown'
            
            if 'Year:' in user_content:
                year = user_content.split('Year:')[1].split('\n')[0].strip()
            if 'Phase:' in user_content:
                phase = user_content.split('Phase:')[1].split('\n')[0].strip()
            
            samples.append({
                'interaction': interaction_num,
                'year': year,
                'phase': phase,
                'system': data['messages'][0]['content'],
                'user': data['messages'][1]['content'] if len(data['messages']) > 1 else '',
                'assistant': data['messages'][2]['content'] if len(data['messages']) > 2 else ''
            })
            
            stats['total_samples'] += 1
            stats['phases'].add(f"{year}_{phase}")
            
            if interaction_num > 0:
                stats['interaction_range']['min'] = min(stats['interaction_range']['min'], interaction_num)
                stats['interaction_range']['max'] = max(stats['interaction_range']['max'], interaction_num)
    
    # Sort by interaction number
    samples.sort(key=lambda x: x['interaction'])
    
    # Build the full trajectory
    full_trajectory = {
        'metadata': {
            'game': 'Diplomacy',
            'player': 'ITALY',
            'outcome': 'VICTORY',
            'total_interactions': len(samples),
            'years_covered': f"1901-1953",
            'total_phases': len(stats['phases'])
        },
        'system_prompt': samples[0]['system'] if samples else '',
        'messages': []
    }
    
    # Add all user-assistant pairs in order
    for sample in samples:
        # User message
        full_trajectory['messages'].append({
            'role': 'user',
            'content': sample['user'],
            'metadata': {
                'interaction': sample['interaction'],
                'year': sample['year'],
                'phase': sample['phase']
            }
        })
        
        # Assistant message
        full_trajectory['messages'].append({
            'role': 'assistant',
            'content': sample['assistant'],
            'metadata': {
                'interaction': sample['interaction'],
                'year': sample['year'],
                'phase': sample['phase']
            }
        })
        
        stats['total_messages'] += 2
    
    # Write the full trajectory
    with open(output_path, 'w') as f:
        json.dump(full_trajectory, f, ensure_ascii=False, indent=2)
    
    # Also create a simplified version for easier processing
    simple_path = output_path.with_name(output_path.stem + '_simple.jsonl')
    with open(simple_path, 'w') as f:
        for i in range(0, len(full_trajectory['messages']), 2):
            if i + 1 < len(full_trajectory['messages']):
                simple_entry = {
                    'interaction': full_trajectory['messages'][i]['metadata']['interaction'],
                    'year': full_trajectory['messages'][i]['metadata']['year'],
                    'phase': full_trajectory['messages'][i]['metadata']['phase'],
                    'user': full_trajectory['messages'][i]['content'],
                    'assistant': full_trajectory['messages'][i + 1]['content']
                }
                f.write(json.dumps(simple_entry, ensure_ascii=False) + '\n')
    
    stats['phases'] = len(stats['phases'])  # Convert set to count
    
    return stats


def verify_trajectory(trajectory_path: Path):
    """Verify the reconstructed trajectory is complete and ordered."""
    with open(trajectory_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nTrajectory Verification:")
    print(f"Total messages: {len(data['messages'])}")
    print(f"User messages: {sum(1 for m in data['messages'] if m['role'] == 'user')}")
    print(f"Assistant messages: {sum(1 for m in data['messages'] if m['role'] == 'assistant')}")
    
    # Check interaction ordering
    interactions = []
    for msg in data['messages']:
        if msg['role'] == 'user' and 'metadata' in msg:
            interactions.append(msg['metadata']['interaction'])
    
    # Check if sorted
    is_sorted = all(interactions[i] <= interactions[i+1] for i in range(len(interactions)-1))
    print(f"Interactions in order: {is_sorted}")
    
    if interactions:
        print(f"Interaction range: {min(interactions)} to {max(interactions)}")
        
        # Check for gaps
        expected = set(range(min(interactions), max(interactions) + 1))
        actual = set(interactions)
        missing = expected - actual
        
        if missing:
            print(f"Missing interactions: {sorted(missing)[:10]}...")
        else:
            print("No missing interactions!")
    
    # Sample first and last phases
    phases = [(msg['metadata']['year'], msg['metadata']['phase']) 
              for msg in data['messages'] 
              if msg['role'] == 'user' and 'metadata' in msg]
    
    if phases:
        print(f"\nFirst phase: {phases[0]}")
        print(f"Last phase: {phases[-1]}")
        
        # Count unique phases
        unique_phases = set(phases)
        print(f"Total unique phases: {len(unique_phases)}")


def main():
    """Main function."""
    input_file = Path('/home/maxpaperclips/atropos/data/diplomacy_with_tool_calls.jsonl')
    output_file = Path('/home/maxpaperclips/atropos/data/diplomacy_full_trajectory.json')
    
    print(f"Reconstructing trajectory from {input_file}...")
    print(f"Output will be saved to {output_file}")
    
    stats = reconstruct_trajectory(input_file, output_file)
    
    print(f"\nReconstruction complete!")
    print(f"Total samples processed: {stats['total_samples']}")
    print(f"Total messages in trajectory: {stats['total_messages']}")
    print(f"Unique phases: {stats['phases']}")
    print(f"Interaction range: {stats['interaction_range']['min']} to {stats['interaction_range']['max']}")
    
    # Verify the output
    verify_trajectory(output_file)
    
    print(f"\nAlso created simplified version: {output_file.with_name(output_file.stem + '_simple.jsonl')}")


if __name__ == '__main__':
    main()