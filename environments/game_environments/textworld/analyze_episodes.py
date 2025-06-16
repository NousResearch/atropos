#!/usr/bin/env python3
"""Analyze TextWorld episodes from generated data."""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def group_by_episode(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group scored data groups by episode ID."""
    episodes = defaultdict(list)
    for item in data:
        metadata = item.get('metadata', {})
        episode_id = metadata.get('episode_id', 'unknown')
        episodes[episode_id].append(item)
    
    # Sort each episode by turn number
    for episode_id in episodes:
        episodes[episode_id].sort(key=lambda x: x.get('metadata', {}).get('turn_number', 0))
    
    return dict(episodes)


def analyze_episode(episode_data: List[Dict[str, Any]], episode_id: str) -> None:
    """Analyze a single episode."""
    print(f"\n{'='*80}")
    print(f"Episode: {episode_id}")
    print(f"Total turns: {len(episode_data)}")
    print(f"{'='*80}")
    
    for turn_data in episode_data:
        metadata = turn_data.get('metadata', {})
        turn_num = metadata.get('turn_number', -1)
        chosen_idx = metadata.get('chosen_alternative_index', -1)
        vrcli_scores = metadata.get('vrcli_scores', [])
        env_rewards = metadata.get('env_rewards', [])
        final_scores = turn_data.get('scores', [])
        
        print(f"\nTurn {turn_num}:")
        print(f"  Chosen alternative: {chosen_idx}")
        print(f"  Number of alternatives: {len(vrcli_scores)}")
        
        # Find best alternatives by different metrics
        if vrcli_scores:
            best_vrcli_idx = max(range(len(vrcli_scores)), key=lambda i: vrcli_scores[i])
            print(f"  Best VR-CLI score: {vrcli_scores[best_vrcli_idx]:.4f} (alternative {best_vrcli_idx})")
        
        if env_rewards:
            best_env_idx = max(range(len(env_rewards)), key=lambda i: env_rewards[i])
            print(f"  Best env reward: {env_rewards[best_env_idx]:.4f} (alternative {best_env_idx})")
        
        if final_scores:
            best_final_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
            print(f"  Best final score: {final_scores[best_final_idx]:.4f} (alternative {best_final_idx})")
        
        # Check for zero VR-CLI scores
        zero_vrcli = sum(1 for s in vrcli_scores if s == 0.0)
        print(f"  Zero VR-CLI scores: {zero_vrcli}/{len(vrcli_scores)}")
        
        # Show memory if present
        messages = turn_data.get('messages', [[]])[0]  # First alternative's messages
        for msg in messages:
            if msg.get('role') == 'assistant' and '<memory>' in msg.get('content', ''):
                import re
                memory_match = re.search(r'<memory>(.*?)</memory>', msg['content'], re.DOTALL)
                if memory_match:
                    print(f"  Memory: {memory_match.group(1).strip()}")
                    break


def main():
    parser = argparse.ArgumentParser(description='Analyze TextWorld episode data')
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('--episode', help='Analyze specific episode ID')
    parser.add_argument('--stats-only', action='store_true', help='Show statistics only')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} scored data groups")
    
    # Group by episode
    episodes = group_by_episode(data)
    print(f"Found {len(episodes)} unique episodes")
    
    if args.stats_only:
        # Show overall statistics
        total_turns = sum(len(ep_data) for ep_data in episodes.values())
        avg_turns = total_turns / len(episodes) if episodes else 0
        print(f"Average turns per episode: {avg_turns:.2f}")
        
        # VR-CLI score statistics
        all_vrcli_scores = []
        for ep_data in episodes.values():
            for turn_data in ep_data:
                all_vrcli_scores.extend(turn_data.get('metadata', {}).get('vrcli_scores', []))
        
        if all_vrcli_scores:
            zero_scores = sum(1 for s in all_vrcli_scores if s == 0.0)
            non_zero_scores = [s for s in all_vrcli_scores if s > 0.0]
            print(f"\nVR-CLI Score Statistics:")
            print(f"  Total scores: {len(all_vrcli_scores)}")
            print(f"  Zero scores: {zero_scores} ({zero_scores/len(all_vrcli_scores)*100:.1f}%)")
            if non_zero_scores:
                print(f"  Non-zero score range: {min(non_zero_scores):.6f} - {max(non_zero_scores):.6f}")
                print(f"  Average non-zero score: {sum(non_zero_scores)/len(non_zero_scores):.6f}")
        
        # Show episode length distribution
        print(f"\nEpisode length distribution:")
        length_counts = defaultdict(int)
        for ep_data in episodes.values():
            length_counts[len(ep_data)] += 1
        for length in sorted(length_counts.keys()):
            print(f"  {length} turns: {length_counts[length]} episodes")
    
    elif args.episode:
        # Analyze specific episode
        if args.episode in episodes:
            analyze_episode(episodes[args.episode], args.episode)
        else:
            print(f"Episode {args.episode} not found")
            print("Available episodes:")
            for ep_id in sorted(episodes.keys())[:10]:
                print(f"  {ep_id}")
    else:
        # Analyze first few episodes
        for i, (ep_id, ep_data) in enumerate(episodes.items()):
            if i >= 3:  # Only show first 3 episodes
                break
            analyze_episode(ep_data, ep_id)


if __name__ == '__main__':
    main()