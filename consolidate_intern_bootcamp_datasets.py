#!/usr/bin/env python3
"""
Consolidate InternBootcamp datasets, filter for top-scoring samples, and convert to ShareGPT format.

This script:
1. Reads all intern_bootcamp dataset files
2. Filters for top N scoring samples per group (weighted by word count)
3. Converts to ShareGPT format (JSONL)
"""

import json
import argparse
import os
import glob
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import re


def extract_bootcamp_name(messages: List[Dict[str, str]]) -> str:
    """Extract the bootcamp task name from the user message."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Look for bootcamp task patterns
            patterns = [
                r'Problem Name:\s*([A-Za-z0-9_]+)',
                r'Task:\s*([A-Za-z0-9_]+)',
                r'Bootcamp:\s*([A-Za-z0-9_]+)',
                r'\[([A-Za-z0-9_]+)bootcamp\]',
                r'([A-Za-z0-9_]+)bootcamp',
            ]
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1).lower() + "bootcamp"
            # Fallback: try to extract from the content itself
            if "bootcamp" in content.lower():
                # Extract the word before "bootcamp"
                match = re.search(r'(\w+)\s*bootcamp', content, re.IGNORECASE)
                if match:
                    return match.group(1).lower() + "bootcamp"
    return "unknown_bootcamp"


def load_dataset_file(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset file."""
    groups = []
    print(f"Loading {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    group = json.loads(line)
                    groups.append(group)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse line {line_num}: {e}")
    except Exception as e:
        print(f"  Error reading file: {e}")
    
    print(f"  Loaded {len(groups)} groups")
    return groups


def filter_all_positive_samples(groups: List[Dict[str, Any]]) -> List[Tuple[List[Dict], float, str]]:
    """
    Get ALL samples with positive scores (> 0).
    
    Returns list of (messages, score, bootcamp_name) tuples.
    """
    all_selected_samples = []
    total_positive_samples = 0
    groups_with_positive = 0
    
    for group_idx, group in enumerate(groups):
        scores = group.get("scores", [])
        messages_list = group.get("messages", [])
        
        if not scores or not messages_list:
            continue
        
        positive_in_group = 0
        
        for idx, (score, messages) in enumerate(zip(scores, messages_list)):
            if score <= 0:  # Skip negative or zero scores
                continue
            
            positive_in_group += 1
            
            # Extract bootcamp name
            bootcamp_name = extract_bootcamp_name(messages)
            
            all_selected_samples.append((messages, score, bootcamp_name))
        
        if positive_in_group > 0:
            groups_with_positive += 1
            total_positive_samples += positive_in_group
    
    print(f"\nTotal positive samples found: {total_positive_samples} across {groups_with_positive} groups")
    print(f"Selected ALL {len(all_selected_samples)} positive samples")
    
    return all_selected_samples


def convert_to_sharegpt(messages: List[Dict[str, str]], score: float, bootcamp_name: str) -> Dict[str, Any]:
    """Convert a conversation to ShareGPT format."""
    conversations = []
    
    for msg in messages:
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
        
        conversations.append({
            "from": from_role,
            "value": content
        })
    
    # Create the ShareGPT entry with metadata
    return {
        "conversations": conversations,
        "task_name": bootcamp_name,
        "score": score,
        "source": "intern_bootcamp"
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Consolidate InternBootcamp datasets and convert to ShareGPT format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='intern_bootcamp_deephermes24b_parallel_sglang_dataset_*.jsonl',
        help='File pattern to match dataset files (default: intern_bootcamp_deephermes24b_parallel_sglang_dataset_*.jsonl)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/maxpaperclips/atropos/data',
        help='Directory containing dataset files (default: /home/maxpaperclips/atropos/data)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='intern_bootcamp_all_positive_sharegpt.jsonl',
        help='Output JSONL file path (default: intern_bootcamp_all_positive_sharegpt.jsonl)'
    )
    
    parser.add_argument(
        '--include-all-datasets',
        action='store_true',
        help='Include all dataset types (not just parallel_sglang)'
    )
    
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.0,
        help='Minimum score threshold (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Find matching files
    if args.include_all_datasets:
        patterns = [
            'intern_bootcamp_deephermes24b_parallel_sglang_dataset_*.jsonl',
            'intern_bootcamp_deephermes24b_dataset_*.jsonl',
            'intern_bootcamp_deephermes24b_sglang_dataset_*.jsonl',
        ]
    else:
        patterns = [args.pattern]
    
    all_files = []
    for pattern in patterns:
        file_pattern = os.path.join(args.data_dir, pattern)
        matching_files = glob.glob(file_pattern)
        all_files.extend(matching_files)
    
    # Remove duplicates and sort
    all_files = sorted(set(all_files))
    
    if not all_files:
        print(f"No files found matching patterns in {args.data_dir}")
        return 1
    
    print(f"Found {len(all_files)} dataset files")
    
    # Load all datasets
    all_groups = []
    total_lines = 0
    for filepath in all_files:
        groups = load_dataset_file(filepath)
        all_groups.extend(groups)
        total_lines += len(groups)
    
    print(f"\nTotal groups loaded: {total_lines}")
    
    # Filter samples
    print(f"\nSelecting ALL positive samples...")
    selected_samples = filter_all_positive_samples(all_groups)
    
    print(f"\nSelected {len(selected_samples)} samples")
    
    # Group by bootcamp name for statistics
    bootcamp_counts = defaultdict(int)
    for _, _, bootcamp_name in selected_samples:
        bootcamp_counts[bootcamp_name] += 1
    
    print("\nBootcamp distribution:")
    for bootcamp, count in sorted(bootcamp_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {bootcamp}: {count}")
    
    # Convert to ShareGPT format and write output
    print(f"\nWriting to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for messages, score, bootcamp_name in selected_samples:
            if score >= args.min_score:
                sharegpt_entry = convert_to_sharegpt(messages, score, bootcamp_name)
                f.write(json.dumps(sharegpt_entry, ensure_ascii=False) + '\n')
    
    print(f"Conversion complete! Output saved to {args.output}")
    
    # Print summary statistics
    output_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
    print(f"\nOutput file size: {output_size:.2f} MB")
    
    return 0


if __name__ == "__main__":
    exit(main())