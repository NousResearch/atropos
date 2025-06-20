#!/usr/bin/env python3
"""
Test script to verify parallel processing implementation with a small test case
"""
import asyncio
import os
import sys
from pathlib import Path

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables only.")

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


async def test_parallel_processing():
    """Test the parallel processing with a small number of groups"""
    # Import the environment
    from atroposlib.envs.server_handling.server_manager import APIServerConfig
    from environments.intern_bootcamp.intern_bootcamp_env import InternBootcampEnv

    # Create a test config
    config = InternBootcampEnv.env_config_cls(
        task_name="RandomTask",
        group_size=2,  # Small group size for testing
        total_steps=5,  # Just 5 groups to test
        max_num_workers=3,  # Test with 3 parallel workers
        use_parallel_processing=True,
        data_path_to_save_groups="test_parallel_output.jsonl",
        ensure_scores_are_not_same=False,
        include_messages=True,
        use_wandb=False,
        temperature=0.7,
        top_p=0.9,
        max_token_length=2048,
    )

    # Use OpenAI API for testing (replace with your actual API details)
    server_config = APIServerConfig(
        model_name="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )

    # Create the environment
    env = InternBootcampEnv(
        config=config,
        server_configs=server_config,
        slurm=False,
        testing=True,
    )

    # Set up process mode parameters
    env.process_mode = True
    env.n_groups_to_process = config.total_steps
    env.group_size_to_process = config.group_size

    print("Starting parallel processing test...")
    print(
        f"Processing {config.total_steps} groups with up to {config.max_num_workers} parallel workers"
    )
    print(f"Each group will generate {config.group_size} responses")

    # Run the parallel process manager
    await env.parallel_process_manager()

    print("\nTest completed!")

    # Check output
    output_path = Path(config.data_path_to_save_groups)
    if output_path.exists():
        with open(output_path, "r") as f:
            lines = f.readlines()
        print(f"Generated {len(lines)} groups")
        print(f"Output file: {output_path}")

        # Clean up test file
        os.remove(output_path)
        print("Test file cleaned up")
    else:
        print("ERROR: No output file generated!")


if __name__ == "__main__":
    print("Testing InternBootcamp parallel processing implementation")
    print("=" * 60)

    # Check if we have an API key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: No OPENAI_API_KEY found. Set it to run the test:")
        print("export OPENAI_API_KEY='your-api-key'")
        print(
            "\nAlternatively, you can modify this script to use a different API endpoint"
        )
        sys.exit(1)

    asyncio.run(test_parallel_processing())
