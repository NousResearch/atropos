#!/usr/bin/env python3
"""
Script to setup all environments and generate train/test indices.

This script loops over all the environments (letter counting, instruction following,
tool calling, format following, pydantic to json, and reasoning gym) and calls their 
setup functions to generate train/test splits and save the indices to files.

For reasoning gym, it saves the full test data samples since it doesn't use traditional
train/test splits.

The script runs each environment twice to check for determinism and warns if any
differences are found between runs.
"""

import asyncio
import sys
import os
import shutil
import filecmp
import json
import hashlib

# Add the environments directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from letter_counting_environment import LetterCountingEnv
from instruction_following_algorithm_environment import InstructionFollowingEnv
from tool_calling_server import SingleToolCallingEnv
from answer_format_environment.answer_format_environment import AnswerFormatEnv
from pydantic_schema_following_environment.pydantic_schema_following_environment import PydanticSchemaFollowingEnv
from reasoning_gym_environment.reasoning_gym_environment import ReasoningGymEnv


async def setup_environment(env_class, env_name, run_number=1):
    """
    Setup a single environment and handle any errors.
    
    Args:
        env_class: The environment class to instantiate
        env_name: Name of the environment for logging
        run_number: Which run this is (1 or 2) for determinism testing
    """
    try:
        print(f"\n{'='*60}")
        print(f"Setting up {env_name} (Run {run_number})")
        print(f"{'='*60}")
        
        # Get configuration from the environment class
        config, server_configs = env_class.config_init()
        
        # Create environment instance
        env = env_class(config, server_configs, slurm=False, testing=True)
        
        # Run setup to generate train/test splits and save indices
        await env.setup()
        
        print(f"✅ Successfully set up {env_name} (Run {run_number})")
        
        # Clean up if needed
        if hasattr(env, 'close'):
            await env.close()
        
    except Exception as e:
        print(f"❌ Error setting up {env_name} (Run {run_number}): {str(e)}")
        import traceback
        traceback.print_exc()


def backup_files(run_number):
    """
    Backup all generated files for comparison.
    
    Args:
        run_number: Which run to backup (1 or 2)
    """
    backup_dir = f"backup_run_{run_number}"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    os.makedirs(backup_dir)
    
    # Define all directories that contain generated files
    dirs_to_backup = [
        "atropos_train_test_data"
    ]
    
    for dir_path in dirs_to_backup:
        if os.path.exists(dir_path):
            backup_path = os.path.join(backup_dir, dir_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copytree(dir_path, backup_path)
            print(f"📁 Backed up {dir_path} to {backup_path}")


def compute_file_hash(file_path):
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"❌ Error computing hash for {file_path}: {e}")
        return None


def compare_files(file1, file2):
    """
    Compare two files and return True if they are identical.
    
    Args:
        file1: Path to first file
        file2: Path to second file
        
    Returns:
        True if files are identical, False otherwise
    """
    if not os.path.exists(file1) or not os.path.exists(file2):
        return False
    
    # First check if files are identical using filecmp
    if filecmp.cmp(file1, file2, shallow=False):
        return True
    
    # If basic comparison fails, check hashes
    hash1 = compute_file_hash(file1)
    hash2 = compute_file_hash(file2)
    
    if hash1 is None or hash2 is None:
        return False
    
    return hash1 == hash2


def compare_directories(dir1, dir2):
    """
    Compare two directories recursively and return differences.
    
    Args:
        dir1: Path to first directory
        dir2: Path to second directory
        
    Returns:
        List of differences found
    """
    differences = []
    
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        if os.path.exists(dir1) != os.path.exists(dir2):
            differences.append(f"Directory existence mismatch: {dir1} vs {dir2}")
        return differences
    
    # Get all files in both directories
    files1 = set()
    files2 = set()
    
    for root, dirs, files in os.walk(dir1):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), dir1)
            files1.add(rel_path)
    
    for root, dirs, files in os.walk(dir2):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), dir2)
            files2.add(rel_path)
    
    # Check for missing files
    missing_in_dir2 = files1 - files2
    missing_in_dir1 = files2 - files1
    
    for file in missing_in_dir2:
        differences.append(f"File missing in run 2: {file}")
    
    for file in missing_in_dir1:
        differences.append(f"File missing in run 1: {file}")
    
    # Compare common files
    common_files = files1 & files2
    for file in common_files:
        file1_path = os.path.join(dir1, file)
        file2_path = os.path.join(dir2, file)
        
        if not compare_files(file1_path, file2_path):
            differences.append(f"File content differs: {file}")
    
    return differences


def check_determinism():
    """
    Compare the results of both runs to check for determinism.
    
    Returns:
        True if all results are identical, False otherwise
    """
    print(f"\n{'='*60}")
    print("🔍 Checking determinism between runs...")
    print(f"{'='*60}")
    
    # Directories to compare
    dirs_to_compare = [
        "atropos_train_test_data"
    ]
    
    all_identical = True
    
    for dir_path in dirs_to_compare:
        run1_dir = os.path.join("backup_run_1", dir_path)
        run2_dir = os.path.join("backup_run_2", dir_path)
        
        print(f"\n📂 Comparing {dir_path}...")
        
        differences = compare_directories(run1_dir, run2_dir)
        
        if differences:
            all_identical = False
            print(f"⚠️  NON-DETERMINISTIC BEHAVIOR DETECTED in {dir_path}:")
            for diff in differences:
                print(f"   - {diff}")
        else:
            print(f"✅ Identical results for {dir_path}")
    
    if all_identical:
        print(f"\n🎉 ALL ENVIRONMENTS SHOW DETERMINISTIC BEHAVIOR!")
        print("All generated files are identical between runs.")
    else:
        print(f"\n⚠️  WARNING: NON-DETERMINISTIC BEHAVIOR DETECTED!")
        print("Some environments produced different results between runs.")
        print("This may indicate issues with random seeding or other non-deterministic factors.")
    
    return all_identical


async def setup_all_environments():
    """
    Setup all environments and generate train/test indices.
    
    This function runs each environment twice to check for determinism and warns
    if any differences are found between runs.
    """
    print("🚀 Starting determinism check for all environments...")
    print("This will run each environment twice and compare results.")
    
    # Define environments to setup
    environments = [
        (LetterCountingEnv, "Letter Counting Environment"),
        (InstructionFollowingEnv, "Instruction Following Environment"),
        (SingleToolCallingEnv, "Tool Calling Environment"),
        (AnswerFormatEnv, "Answer Format Environment"),
        (PydanticSchemaFollowingEnv, "Pydantic Schema Following Environment"),
        (ReasoningGymEnv, "Reasoning Gym Environment"),
    ]
    
    # Clean up any existing backup directories
    for i in [1, 2]:
        backup_dir = f"backup_run_{i}"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
    
    # First run
    print(f"\n{'='*80}")
    print("🔄 FIRST RUN - Setting up all environments...")
    print(f"{'='*80}")
    
    for env_class, env_name in environments:
        await setup_environment(env_class, env_name, run_number=1)
    
    print(f"\n📦 Backing up results from first run...")
    backup_files(1)
    
    # Clean up generated files for second run
    print(f"\n🧹 Cleaning up for second run...")
    dirs_to_clean = [
        "atropos_train_test_data"
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"🗑️  Removed {dir_path}")
    
    # Second run
    print(f"\n{'='*80}")
    print("🔄 SECOND RUN - Setting up all environments...")
    print(f"{'='*80}")
    
    for env_class, env_name in environments:
        await setup_environment(env_class, env_name, run_number=2)
    
    print(f"\n📦 Backing up results from second run...")
    backup_files(2)
    
    # Compare results and check for determinism
    deterministic = check_determinism()
    
    print(f"\n{'='*60}")
    print("🎉 Environment setup and determinism check complete!")
    print(f"{'='*60}")
    print("\nAll train/test data has been saved to a unified directory:")
    print("- environments/atropos_train_test_data/")
    
    print("\nFiles created:")
    print("- letter_counting_train_indices.txt / letter_counting_test_indices.txt")
    print("- instruction_following_train_indices.txt / instruction_following_test_indices.txt")
    print("- tool_calling_train_indices.txt / tool_calling_test_indices.txt")
    print("- answer_format_train_indices.txt / answer_format_test_indices.txt")
    print("- pydantic_schema_following_train_indices.txt / pydantic_schema_following_test_indices.txt")
    print("- reasoning_gym_test_data.json / reasoning_gym_test_task_names.txt")
    
    print("\nBackup directories for comparison:")
    print("- backup_run_1/ (first run results)")
    print("- backup_run_2/ (second run results)")
    
    if not deterministic:
        print("\n⚠️  WARNING: Non-deterministic behavior detected!")
        print("Check the comparison results above for details.")
        return False
    
    return True


def main():
    """Main entry point."""
    print("Environment Setup Script with Determinism Check")
    print("=" * 60)
    
    # Run the async setup function
    deterministic = asyncio.run(setup_all_environments())
    
    # Exit with appropriate code
    if deterministic:
        print("\n✅ All environments are deterministic!")
        sys.exit(0)
    else:
        print("\n❌ Some environments show non-deterministic behavior!")
        sys.exit(1)


if __name__ == "__main__":
    main()