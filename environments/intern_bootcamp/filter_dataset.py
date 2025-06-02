#!/usr/bin/env python3
"""
Dataset Filtering Script for InternBootcamp

This script filters the generated dataset to keep only high-scoring responses
suitable for SFT training. It processes JSONL files from the process mode output.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any


def filter_scored_groups(input_file: str, output_file: str, min_score: float = 0.0, verbose: bool = False) -> Dict[str, Any]:
    """
    Filter scored groups to keep only responses above the minimum score threshold.
    
    Args:
        input_file: Path to the input JSONL file with scored groups
        output_file: Path to the output JSONL file for filtered data
        min_score: Minimum score threshold (default: 0.0)
        verbose: Whether to print detailed statistics
        
    Returns:
        Dictionary with filtering statistics
    """
    stats = {
        "total_groups": 0,
        "total_responses": 0,
        "filtered_responses": 0,
        "kept_responses": 0,
        "score_distribution": {},
        "tasks_with_good_responses": set(),
        "avg_score_kept": 0.0
    }
    
    kept_scores = []
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                group = json.loads(line.strip())
                stats["total_groups"] += 1
                
                # Extract information from the group
                tokens = group.get("tokens", [])
                scores = group.get("scores", [])
                messages = group.get("messages", [])
                
                if len(tokens) != len(scores) or len(scores) != len(messages):
                    if verbose:
                        print(f"Warning: Mismatched lengths in group {line_num}")
                    continue
                
                stats["total_responses"] += len(scores)
                
                # Filter responses above threshold
                filtered_indices = []
                for i, score in enumerate(scores):
                    # Count score distribution
                    score_bucket = f"{score:.1f}"
                    stats["score_distribution"][score_bucket] = stats["score_distribution"].get(score_bucket, 0) + 1
                    
                    if score > min_score:
                        filtered_indices.append(i)
                        kept_scores.append(score)
                        
                        # Track which tasks have good responses
                        if messages and len(messages[i]) > 1:
                            task_content = messages[i][1].get("content", "")
                            # Try to extract task type from content
                            if "bootcamp" in str(group.get("messages", [])):
                                stats["tasks_with_good_responses"].add("bootcamp_task")
                
                stats["filtered_responses"] += len(filtered_indices)
                
                # Create filtered group if we have any good responses
                if filtered_indices:
                    filtered_group = {
                        "tokens": [tokens[i] for i in filtered_indices],
                        "masks": [group["masks"][i] for i in filtered_indices],
                        "scores": [scores[i] for i in filtered_indices],
                        "messages": [messages[i] for i in filtered_indices] if messages else None
                    }
                    
                    # Preserve other fields if they exist
                    for key in ["advantages", "ref_logprobs", "group_overrides", "overrides"]:
                        if key in group and group[key] is not None:
                            if isinstance(group[key], list):
                                filtered_group[key] = [group[key][i] for i in filtered_indices]
                            else:
                                filtered_group[key] = group[key]
                    
                    outfile.write(json.dumps(filtered_group) + '\n')
                    stats["kept_responses"] += len(filtered_indices)
                
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                if verbose:
                    print(f"Error processing line {line_num}: {e}")
                continue
    
    # Calculate average score of kept responses
    if kept_scores:
        stats["avg_score_kept"] = sum(kept_scores) / len(kept_scores)
    
    # Convert set to list for JSON serialization
    stats["tasks_with_good_responses"] = list(stats["tasks_with_good_responses"])
    
    return stats


def extract_task_name_from_messages(messages: List[Dict]) -> str:
    """Extract task name from the conversation messages."""
    try:
        # Look for task name in the user message content
        user_message = next((msg for msg in messages if msg.get("role") == "user"), None)
        if not user_message:
            return "unknown_task"
        
        content = user_message.get("content", "").lower()
        
        # Check for specific task patterns
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
        elif "bootcamp" in str(messages).lower():
            # Fallback: try to extract from any bootcamp reference
            bootcamp_indicators = ["aquarium", "sudoku", "maze", "puzzle", "cryptomath", "kakurasu", "starbattle"]
            for indicator in bootcamp_indicators:
                if indicator in content:
                    return f"{indicator}_puzzle"
        
        return "reasoning_task"
    except Exception:
        return "unknown_task"


def extract_conversations_for_sft(input_file: str, output_file: str, min_score: float = 0.0, verbose: bool = False) -> Dict[str, Any]:
    """
    Extract conversations in ShareGPT format suitable for SFT training.
    
    Args:
        input_file: Path to the input JSONL file with scored groups
        output_file: Path to the output JSONL file for SFT conversations in ShareGPT format
        min_score: Minimum score threshold (default: 0.0)
        verbose: Whether to print detailed statistics
        
    Returns:
        Dictionary with extraction statistics
    """
    stats = {
        "total_conversations": 0,
        "kept_conversations": 0,
        "avg_score": 0.0,
        "score_range": {"min": float("inf"), "max": float("-inf")},
        "task_distribution": {}
    }
    
    kept_scores = []
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                group = json.loads(line.strip())
                
                scores = group.get("scores", [])
                messages = group.get("messages", [])
                
                if not messages or len(scores) != len(messages):
                    continue
                
                for i, (score, conversation) in enumerate(zip(scores, messages)):
                    stats["total_conversations"] += 1
                    
                    if score > min_score:
                        # Extract task name for labeling
                        task_name = extract_task_name_from_messages(conversation)
                        stats["task_distribution"][task_name] = stats["task_distribution"].get(task_name, 0) + 1
                        
                        # Convert to ShareGPT format
                        sharegpt_conversations = []
                        
                        for msg in conversation:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            
                            # Map roles to ShareGPT format
                            if role == "system":
                                from_role = "system"
                            elif role == "user":
                                from_role = "human"
                            elif role == "assistant":
                                from_role = "gpt"
                            else:
                                continue  # Skip unknown roles
                            
                            sharegpt_conversations.append({
                                "from": from_role,
                                "value": content
                            })
                        
                        # Create ShareGPT format item
                        sharegpt_item = {
                            "conversations": sharegpt_conversations,
                            "task_name": task_name,
                            "score": score,
                            "source": "intern_bootcamp"
                        }
                        
                        outfile.write(json.dumps(sharegpt_item) + '\n')
                        stats["kept_conversations"] += 1
                        kept_scores.append(score)
                        
                        # Update score range
                        stats["score_range"]["min"] = min(stats["score_range"]["min"], score)
                        stats["score_range"]["max"] = max(stats["score_range"]["max"], score)
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                if verbose:
                    print(f"Error processing conversation: {e}")
                continue
    
    if kept_scores:
        stats["avg_score"] = sum(kept_scores) / len(kept_scores)
    else:
        stats["score_range"] = {"min": 0, "max": 0}
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter InternBootcamp dataset by scores for SFT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter keeping scores > 0 (default)
  python filter_dataset.py data/intern_bootcamp_deephermes24b_dataset.jsonl

  # Filter with custom threshold
  python filter_dataset.py data/intern_bootcamp_deephermes24b_dataset.jsonl --min-score 0.5

  # Extract conversations for SFT training
  python filter_dataset.py data/intern_bootcamp_deephermes24b_dataset.jsonl --format sft

  # Custom output file
  python filter_dataset.py input.jsonl --output filtered.jsonl --verbose
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input JSONL file with scored groups from process mode"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: auto-generated based on input)"
    )
    
    parser.add_argument(
        "--min-score", "-s",
        type=float,
        default=0.0,
        help="Minimum score threshold (default: 0.0)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["groups", "sft"],
        default="groups",
        help="Output format: 'groups' (default) or 'sft' for training"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed statistics and warnings"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.input_file)
        suffix = "sft" if args.format == "sft" else f"filtered_score_{args.min_score:.1f}"
        args.output = str(input_path.with_stem(f"{input_path.stem}_{suffix}"))
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the file
    print(f"Processing: {args.input_file}")
    print(f"Min score threshold: {args.min_score}")
    print(f"Output format: {args.format}")
    print(f"Output file: {args.output}")
    print()
    
    if args.format == "sft":
        stats = extract_conversations_for_sft(args.input_file, args.output, args.min_score, args.verbose)
        
        print("=== SFT Extraction Results ===")
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Kept conversations: {stats['kept_conversations']}")
        print(f"Retention rate: {stats['kept_conversations']/max(1, stats['total_conversations'])*100:.1f}%")
        if stats['kept_conversations'] > 0:
            print(f"Average score: {stats['avg_score']:.3f}")
            print(f"Score range: {stats['score_range']['min']:.3f} to {stats['score_range']['max']:.3f}")
            
            if args.verbose and stats['task_distribution']:
                print("\n=== Task Distribution ===")
                for task, count in sorted(stats['task_distribution'].items()):
                    print(f"{task}: {count} conversations")
    
    else:
        stats = filter_scored_groups(args.input_file, args.output, args.min_score, args.verbose)
        
        print("=== Filtering Results ===")
        print(f"Total groups processed: {stats['total_groups']}")
        print(f"Total responses: {stats['total_responses']}")
        print(f"Responses kept: {stats['kept_responses']}")
        print(f"Retention rate: {stats['kept_responses']/max(1, stats['total_responses'])*100:.1f}%")
        
        if stats['kept_responses'] > 0:
            print(f"Average score of kept responses: {stats['avg_score_kept']:.3f}")
        
        if args.verbose and stats['score_distribution']:
            print("\n=== Score Distribution ===")
            for score, count in sorted(stats['score_distribution'].items()):
                print(f"Score {score}: {count} responses")
    
    print(f"\nFiltered data saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())