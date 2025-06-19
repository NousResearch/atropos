#!/usr/bin/env python3
"""
Dataset Filter and Merger Script

This script filters the intern_bootcamp_all_positive_sharegpt.jsonl dataset by a score
threshold of 0.8 and merges it with the already-filtered intern_bootcamp_consolidated_score_0.8_sft.jsonl
dataset into a single output file.

Requirements:
1. Filter all_positive dataset: Keep only entries with score >= 0.8
2. Merge with the consolidated dataset (already filtered)
3. Output to a single JSONL file

Usage:
    python filter_and_merge_datasets.py
"""

import json
import os
from pathlib import Path


def filter_jsonl_by_score(input_file, output_file, threshold=0.8):
    """
    Filter a JSONL file by score threshold.

    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output filtered JSONL file
        threshold (float): Minimum score threshold (inclusive)

    Returns:
        int: Number of entries that passed the filter
    """
    filtered_count = 0
    total_count = 0

    print(f"Filtering {input_file} with score threshold >= {threshold}")

    try:
        with (
            open(input_file, "r", encoding="utf-8") as infile,
            open(output_file, "w", encoding="utf-8") as outfile,
        ):

            for line_num, line in enumerate(infile, 1):
                total_count += 1

                # Progress indicator for large files
                if total_count % 10000 == 0:
                    print(f"  Processed {total_count:,} lines...")

                try:
                    # Parse JSON line
                    data = json.loads(line.strip())

                    # Check if score field exists and meets threshold
                    if "score" in data:
                        score = data["score"]
                        if isinstance(score, (int, float)) and score >= threshold:
                            # Write filtered entry to output file
                            outfile.write(line)
                            filtered_count += 1
                    else:
                        print(f"  Warning: Line {line_num} missing 'score' field")

                except json.JSONDecodeError as e:
                    print(f"  Error parsing JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"  Unexpected error on line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return 0
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return 0

    print(f"  Filtered {filtered_count:,} entries out of {total_count:,} total entries")
    print(f"  Filter rate: {filtered_count/total_count*100:.2f}%")

    return filtered_count


def merge_jsonl_files(file1, file2, output_file):
    """
    Merge two JSONL files into a single output file.

    Args:
        file1 (str): Path to first JSONL file
        file2 (str): Path to second JSONL file
        output_file (str): Path to merged output file

    Returns:
        int: Total number of entries in merged file
    """
    total_entries = 0

    print(f"Merging {file1} and {file2} into {output_file}")

    try:
        with open(output_file, "w", encoding="utf-8") as outfile:

            # Process first file
            print(f"  Adding entries from {file1}...")
            with open(file1, "r", encoding="utf-8") as infile1:
                for line_num, line in enumerate(infile1, 1):
                    outfile.write(line)
                    total_entries += 1

                    if total_entries % 10000 == 0:
                        print(f"    Written {total_entries:,} entries...")

            file1_count = total_entries
            print(f"    Added {file1_count:,} entries from {file1}")

            # Process second file
            print(f"  Adding entries from {file2}...")
            with open(file2, "r", encoding="utf-8") as infile2:
                for line_num, line in enumerate(infile2, 1):
                    outfile.write(line)
                    total_entries += 1

                    if total_entries % 10000 == 0:
                        print(f"    Written {total_entries:,} entries...")

            file2_count = total_entries - file1_count
            print(f"    Added {file2_count:,} entries from {file2}")

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
        return 0
    except Exception as e:
        print(f"Error merging files: {e}")
        return 0

    print(f"  Total merged entries: {total_entries:,}")
    return total_entries


def get_file_stats(file_path):
    """
    Get basic statistics about a JSONL file.

    Args:
        file_path (str): Path to JSONL file

    Returns:
        dict: Statistics including line count, file size, etc.
    """
    try:
        file_size = os.path.getsize(file_path)
        line_count = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1

        return {
            "line_count": line_count,
            "file_size_mb": file_size / (1024 * 1024),
            "exists": True,
        }
    except Exception as e:
        return {"line_count": 0, "file_size_mb": 0, "exists": False, "error": str(e)}


def main():
    """
    Main function to orchestrate the filtering and merging process.
    """
    # Define file paths
    data_dir = Path("data")
    all_positive_file = data_dir / "intern_bootcamp_all_positive_sharegpt.jsonl"
    consolidated_file = data_dir / "intern_bootcamp_consolidated_score_0.8_sft.jsonl"

    # Temporary file for filtered all_positive data
    filtered_temp_file = data_dir / "intern_bootcamp_all_positive_filtered_temp.jsonl"

    # Final merged output file
    merged_output_file = data_dir / "intern_bootcamp_merged_filtered.jsonl"

    # Score threshold for filtering
    score_threshold = 0.8

    print("=" * 80)
    print("INTERN BOOTCAMP DATASET FILTER AND MERGER")
    print("=" * 80)

    # Check if input files exist
    print("\n1. CHECKING INPUT FILES")
    print("-" * 40)

    all_positive_stats = get_file_stats(all_positive_file)
    consolidated_stats = get_file_stats(consolidated_file)

    if not all_positive_stats["exists"]:
        print(f"Error: {all_positive_file} not found")
        return 1

    if not consolidated_stats["exists"]:
        print(f"Error: {consolidated_file} not found")
        return 1

    print(f"✓ {all_positive_file.name}")
    print(f"  Lines: {all_positive_stats['line_count']:,}")
    print(f"  Size: {all_positive_stats['file_size_mb']:.1f} MB")

    print(f"✓ {consolidated_file.name}")
    print(f"  Lines: {consolidated_stats['line_count']:,}")
    print(f"  Size: {consolidated_stats['file_size_mb']:.1f} MB")

    # Step 1: Filter the all_positive dataset by score >= 0.8
    print(f"\n2. FILTERING ALL_POSITIVE DATASET (score >= {score_threshold})")
    print("-" * 40)

    filtered_count = filter_jsonl_by_score(
        input_file=all_positive_file,
        output_file=filtered_temp_file,
        threshold=score_threshold,
    )

    if filtered_count == 0:
        print("Error: No entries passed the filter or filtering failed")
        return 1

    # Step 2: Merge the filtered all_positive with the consolidated dataset
    print(f"\n3. MERGING DATASETS")
    print("-" * 40)

    total_merged = merge_jsonl_files(
        file1=filtered_temp_file,
        file2=consolidated_file,
        output_file=merged_output_file,
    )

    if total_merged == 0:
        print("Error: Merging failed")
        return 1

    # Step 3: Clean up temporary file
    print(f"\n4. CLEANUP")
    print("-" * 40)

    try:
        filtered_temp_file.unlink()
        print(f"✓ Removed temporary file: {filtered_temp_file.name}")
    except Exception as e:
        print(
            f"Warning: Could not remove temporary file {filtered_temp_file.name}: {e}"
        )

    # Step 4: Final summary
    print(f"\n5. SUMMARY")
    print("-" * 40)

    merged_stats = get_file_stats(merged_output_file)

    print(f"✓ Successfully created merged dataset: {merged_output_file.name}")
    print(f"  Total entries: {merged_stats['line_count']:,}")
    print(f"  File size: {merged_stats['file_size_mb']:.1f} MB")
    print(f"  ")
    print(f"  Breakdown:")
    print(f"    - Filtered all_positive entries: {filtered_count:,}")
    print(f"    - Consolidated entries: {consolidated_stats['line_count']:,}")
    print(f"    - Total: {filtered_count + consolidated_stats['line_count']:,}")

    print(f"\n✓ Process completed successfully!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
