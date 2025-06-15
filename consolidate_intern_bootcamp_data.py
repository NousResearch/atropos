#!/usr/bin/env python3
"""
Consolidate multiple intern_bootcamp data files, filter by score, and convert to ShareGPT format.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse


def consolidate_and_filter_datasets(
    input_files: List[str],
    output_file: str,
    min_score: float = 0.8,
    format_type: str = "sft",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Consolidate multiple JSONL files, filter by score, and convert to ShareGPT format.
    
    Args:
        input_files: List of input JSONL files to consolidate
        output_file: Output file path
        min_score: Minimum score threshold (default: 0.8)
        format_type: Output format - "sft" for ShareGPT or "groups" for filtered groups
        verbose: Whether to print detailed statistics
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "total_groups": 0,
        "total_conversations": 0,
        "kept_conversations": 0,
        "avg_score": 0.0,
        "score_range": {"min": float("inf"), "max": float("-inf")},
        "task_distribution": {},
        "file_stats": {}
    }
    
    kept_scores = []
    
    with open(output_file, 'w') as outfile:
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"Warning: File {input_file} does not exist, skipping...")
                continue
            
            file_stats = {
                "groups": 0,
                "conversations": 0,
                "kept": 0
            }
            
            if verbose:
                print(f"\nProcessing: {input_file}")
            
            with open(input_file, 'r') as infile:
                for line_num, line in enumerate(infile, 1):
                    try:
                        group = json.loads(line.strip())
                        stats["total_groups"] += 1
                        file_stats["groups"] += 1
                        
                        scores = group.get("scores", [])
                        messages = group.get("messages", [])
                        bootcamp_names = group.get("bootcamp_names", [])
                        
                        if not messages or len(scores) != len(messages):
                            continue
                        
                        for i, (score, conversation) in enumerate(zip(scores, messages)):
                            stats["total_conversations"] += 1
                            file_stats["conversations"] += 1
                            
                            if score >= min_score:
                                kept_scores.append(score)
                                stats["kept_conversations"] += 1
                                file_stats["kept"] += 1
                                
                                # Update score range
                                stats["score_range"]["min"] = min(stats["score_range"]["min"], score)
                                stats["score_range"]["max"] = max(stats["score_range"]["max"], score)
                                
                                # Extract task name
                                task_name = "unknown_task"
                                if bootcamp_names and i < len(bootcamp_names):
                                    task_name = bootcamp_names[i]
                                else:
                                    # Try to extract from messages (fallback)
                                    task_name = extract_task_name_from_messages(conversation)
                                
                                stats["task_distribution"][task_name] = stats["task_distribution"].get(task_name, 0) + 1
                                
                                if format_type == "sft":
                                    # Convert to ShareGPT format
                                    sharegpt_conversations = []
                                    
                                    for msg in conversation:
                                        role = msg.get("role", "")
                                        content = msg.get("content", "")
                                        
                                        if role == "system":
                                            sharegpt_conversations.append({
                                                "from": "system",
                                                "value": content
                                            })
                                        elif role == "user":
                                            sharegpt_conversations.append({
                                                "from": "human",
                                                "value": content
                                            })
                                        elif role == "assistant":
                                            sharegpt_conversations.append({
                                                "from": "gpt",
                                                "value": content
                                            })
                                    
                                    sharegpt_entry = {
                                        "conversations": sharegpt_conversations,
                                        "task_name": task_name,
                                        "score": score,
                                        "source": "intern_bootcamp"
                                    }
                                    
                                    outfile.write(json.dumps(sharegpt_entry) + "\n")
                                
                                else:  # groups format
                                    # Keep the original group format but only with high-scoring entries
                                    filtered_group = {
                                        "tokens": [group["tokens"][i]] if "tokens" in group else [],
                                        "masks": [group["masks"][i]] if "masks" in group else [],
                                        "scores": [score],
                                        "messages": [conversation],
                                        "bootcamp_names": [task_name]
                                    }
                                    outfile.write(json.dumps(filtered_group) + "\n")
                    
                    except json.JSONDecodeError as e:
                        if verbose:
                            print(f"Error parsing line {line_num} in {input_file}: {e}")
                    except Exception as e:
                        if verbose:
                            print(f"Error processing line {line_num} in {input_file}: {e}")
            
            stats["file_stats"][input_file] = file_stats
            if verbose:
                print(f"  Groups: {file_stats['groups']}, Conversations: {file_stats['conversations']}, Kept: {file_stats['kept']}")
    
    # Calculate average score
    if kept_scores:
        stats["avg_score"] = sum(kept_scores) / len(kept_scores)
    
    return stats


def extract_task_name_from_messages(messages: List[Dict]) -> str:
    """Extract task name from the conversation messages."""
    try:
        # Look for bootcamp names in the content
        content_str = str(messages).lower()
        
        # Check for specific bootcamp patterns
        bootcamp_patterns = {
            "aquariumbootcamp": "aquarium",
            "sudokubootcamp": "sudoku",
            "mazebootcamp": "maze",
            "puzzlebootcamp": "puzzle",
            "cryptomathbootcamp": "cryptomath",
            "kakurasubootcamp": "kakurasu",
            "starbattlebootcamp": "starbattle",
            "slitherlinkbootcamp": "slitherlink",
            "roadconstructionbootcamp": "road_construction",
            "digitoperationsbootcamp": "digit_operations",
            "subsetoperationsbootcamp": "subset_operations",
            "palindromebootcamp": "palindrome_transformation",
            "galacticbootcamp": "galactic_fractions",
            "threestatesbootcamp": "three_states_puzzle"
        }
        
        for pattern, name in bootcamp_patterns.items():
            if pattern in content_str:
                return name + "_bootcamp"
        
        # Fallback to generic patterns
        user_message = next((msg for msg in messages if msg.get("role") == "user"), None)
        if user_message:
            content = user_message.get("content", "").lower()
            
            if "slitherlink" in content:
                return "slitherlink"
            elif "road construction" in content or "connectivity analysis" in content:
                return "road_construction"
            elif "digit" in content and "operation" in content:
                return "digit_operations"
            elif "subset" in content and "indices" in content:
                return "subset_operations"
            elif "palindrome" in content and "arrow" in content:
                return "palindrome_transformation"
            elif "galactic empire" in content and "fraction" in content:
                return "galactic_fractions"
            elif "three states" in content or "grid" in content and ("1" in content and "2" in content and "3" in content):
                return "three_states_puzzle"
        
        return "reasoning_task"
    except Exception:
        return "unknown_task"


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate and filter InternBootcamp datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input-pattern",
        default="data/intern_bootcamp_deephermes24b_parallel_sglang_dataset_*.jsonl",
        help="Glob pattern for input files (default: data/intern_bootcamp_deephermes24b_parallel_sglang_dataset_*.jsonl)"
    )
    
    parser.add_argument(
        "--output",
        default="data/intern_bootcamp_consolidated_score_0.8_sft.jsonl",
        help="Output file path"
    )
    
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.8,
        help="Minimum score threshold (default: 0.8)"
    )
    
    parser.add_argument(
        "--format",
        choices=["groups", "sft"],
        default="sft",
        help="Output format: 'groups' or 'sft' for ShareGPT format (default: sft)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed statistics"
    )
    
    args = parser.parse_args()
    
    # Find all matching files
    from glob import glob
    input_files = sorted(glob(args.input_pattern))
    
    if not input_files:
        print(f"Error: No files found matching pattern '{args.input_pattern}'")
        return 1
    
    print(f"Found {len(input_files)} files to process:")
    for f in input_files:
        print(f"  - {f}")
    
    print(f"\nConsolidating with min score: {args.min_score}")
    print(f"Output format: {args.format}")
    print(f"Output file: {args.output}")
    
    # Process files
    stats = consolidate_and_filter_datasets(
        input_files,
        args.output,
        args.min_score,
        args.format,
        args.verbose
    )
    
    # Print summary
    print("\n=== Consolidation Results ===")
    print(f"Total groups processed: {stats['total_groups']}")
    print(f"Total conversations: {stats['total_conversations']}")
    print(f"Kept conversations: {stats['kept_conversations']}")
    print(f"Retention rate: {stats['kept_conversations']/max(1, stats['total_conversations'])*100:.1f}%")
    
    if stats['kept_conversations'] > 0:
        print(f"Average score: {stats['avg_score']:.3f}")
        print(f"Score range: {stats['score_range']['min']:.3f} to {stats['score_range']['max']:.3f}")
        
        print("\n=== Task Distribution ===")
        for task, count in sorted(stats['task_distribution'].items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"{task}: {count} conversations")
        
        if len(stats['task_distribution']) > 20:
            print(f"... and {len(stats['task_distribution']) - 20} more task types")
    
    print(f"\nOutput written to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())