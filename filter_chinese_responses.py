#!/usr/bin/env python3
"""
Filter out dataset examples where the model responded in Chinese despite being 
instructed to respond in English.

This script:
1. Loads a JSONL dataset file
2. Detects Chinese language in the final "gpt" role response
3. Removes records where the gpt response is in Chinese
4. Preserves records where input instructions are in Chinese (as intended for training)
5. Outputs filtered results to a new file

Usage:
    python filter_chinese_responses.py input.jsonl output.jsonl [--dry-run] [--verbose]
"""

import json
import argparse
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("Error: langdetect library not found. Please install with: pip install langdetect")
    sys.exit(1)


def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of the given text.
    
    Args:
        text: The text to analyze
        
    Returns:
        The detected language code (e.g., 'zh', 'en') or None if detection fails
    """
    if not text or not text.strip():
        return None
        
    try:
        # Clean text by removing extra whitespace and special tokens
        cleaned_text = text.strip()
        
        # Skip very short texts as they're unreliable for detection
        if len(cleaned_text) < 10:
            return None
            
        return detect(cleaned_text)
    except LangDetectException:
        return None


def is_chinese(text: str) -> bool:
    """
    Check if the text is primarily in Chinese.
    
    Args:
        text: The text to check
        
    Returns:
        True if the text is detected as Chinese, False otherwise
    """
    detected_lang = detect_language(text)
    return detected_lang == 'zh-cn' or detected_lang == 'zh'


def extract_gpt_response(conversations: List[Dict]) -> Optional[str]:
    """
    Extract the final "gpt" role response from conversations.
    
    Args:
        conversations: List of conversation messages
        
    Returns:
        The content of the final gpt response, or None if not found
    """
    # Find the last message with "from": "gpt"
    gpt_responses = [msg for msg in conversations if msg.get("from") == "gpt"]
    
    if not gpt_responses:
        return None
        
    # Return the content of the last gpt response
    return gpt_responses[-1].get("value", "")


def should_filter_record(record: Dict, verbose: bool = False) -> Tuple[bool, str]:
    """
    Determine if a record should be filtered out.
    
    Args:
        record: The dataset record to check
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (should_filter, reason)
    """
    conversations = record.get("conversations", [])
    if not conversations:
        return True, "No conversations found"
    
    # Extract the final gpt response
    gpt_response = extract_gpt_response(conversations)
    if not gpt_response:
        return True, "No gpt response found"
    
    # Check if the gpt response is in Chinese
    if is_chinese(gpt_response):
        if verbose:
            print(f"Chinese detected in gpt response: {gpt_response[:100]}...")
        return True, "Gpt response is in Chinese"
    
    return False, "Gpt response is not in Chinese"


def filter_dataset(
    input_file: Path, 
    output_file: Path, 
    dry_run: bool = False, 
    verbose: bool = False
) -> Dict[str, int]:
    """
    Filter the dataset to remove records with Chinese gpt responses.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        dry_run: If True, don't write output, just report statistics
        verbose: If True, print detailed information
        
    Returns:
        Dictionary with filtering statistics
    """
    stats = {
        "total_records": 0,
        "filtered_out": 0,
        "kept": 0,
        "errors": 0
    }
    
    print(f"Processing {input_file}...")
    
    if not dry_run and output_file.exists():
        response = input("Output file exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return stats
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            if dry_run:
                outfile = None
            else:
                outfile = open(output_file, 'w', encoding='utf-8')
            
            try:
                for line_num, line in enumerate(infile, 1):
                    stats["total_records"] += 1
                    
                    try:
                        # Parse the JSON record
                        record = json.loads(line.strip())
                        
                        # Check if this record should be filtered
                        should_filter, reason = should_filter_record(record, verbose)
                        
                        if should_filter:
                            stats["filtered_out"] += 1
                            if verbose:
                                print(f"Line {line_num}: FILTERED - {reason}")
                        else:
                            stats["kept"] += 1
                            if verbose:
                                print(f"Line {line_num}: KEPT - {reason}")
                            
                            # Write to output file if not dry run
                            if not dry_run:
                                json.dump(record, outfile, ensure_ascii=False)
                                outfile.write('\n')
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        stats["errors"] += 1
                        if verbose:
                            print(f"Line {line_num}: ERROR - {e}")
                        continue
                    
                    # Progress reporting
                    if line_num % 1000 == 0:
                        print(f"Processed {line_num} records...")
                        
            finally:
                if outfile and not dry_run:
                    outfile.close()
                    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return stats
    except Exception as e:
        print(f"Error processing file: {e}")
        return stats
    
    # Print summary
    print("\n" + "="*50)
    print("FILTERING SUMMARY")
    print("="*50)
    print(f"Total records processed: {stats['total_records']:,}")
    print(f"Records kept: {stats['kept']:,}")
    print(f"Records filtered out: {stats['filtered_out']:,}")
    print(f"Errors encountered: {stats['errors']:,}")
    
    if stats['total_records'] > 0:
        kept_pct = (stats['kept'] / stats['total_records']) * 100
        filtered_pct = (stats['filtered_out'] / stats['total_records']) * 100
        print(f"Kept: {kept_pct:.1f}%")
        print(f"Filtered: {filtered_pct:.1f}%")
    
    if not dry_run and stats['kept'] > 0:
        print(f"\nFiltered dataset saved to: {output_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset to remove records with Chinese gpt responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Filter a dataset file
    python filter_chinese_responses.py data/input.jsonl data/output_filtered.jsonl
    
    # Dry run to see what would be filtered
    python filter_chinese_responses.py data/input.jsonl data/output.jsonl --dry-run
    
    # Verbose output to see details
    python filter_chinese_responses.py data/input.jsonl data/output.jsonl --verbose
        """
    )
    
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input JSONL dataset file"
    )
    
    parser.add_argument(
        "output_file", 
        type=Path,
        help="Output JSONL file for filtered dataset"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be filtered without creating output file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Print detailed information about each record"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input_file.exists():
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)
    
    # Run the filtering
    stats = filter_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Exit with error code if no records were kept
    if stats['kept'] == 0 and stats['total_records'] > 0:
        print("Warning: No records were kept after filtering")
        sys.exit(1)


if __name__ == "__main__":
    main() 