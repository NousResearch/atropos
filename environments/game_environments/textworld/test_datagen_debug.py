"""Debug script to test TextWorld data generation and find where it hangs."""
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from atroposlib.envs.base import APIServerConfig
from environments.game_environments.textworld.textworld_env import TextWorldEnv, TextWorldEnvConfig
from environments.game_environments.textworld.agents.atropos_agent import AtroposAgentConfig

# Configure logging to be very verbose
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set debug logging for all relevant modules
for logger_name in [
    "environments.game_environments.textworld",
    "atroposlib.envs.base",
    "environments.game_environments.textworld.textworld_env",
    "environments.game_environments.textworld.agents.atropos_agent",
]:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)


async def test_environment_initialization():
    """Test just the environment initialization."""
    print("\n=== Testing Environment Initialization ===")
    
    server_config = APIServerConfig(
        model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        base_url="http://localhost:30000/v1",
        api_key="dummy",
        num_requests_for_eval=0,
        server_type="openai",
    )
    
    env_config = TextWorldEnvConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        max_token_length=16384,
        total_steps=1,  # Just one step
        group_size=2,   # Just 2 alternatives
        max_steps=5,    # Short episodes
        use_registry=True,
        registry_mode="challenge",  # Use pre-built to avoid generation issues
        registry_difficulty="easy",
        debug_mode=True,
        default_server_config=server_config,
        atropos_agent_config=AtroposAgentConfig(
            enable_memory=True,
            temperature=0.7,
            max_tokens_per_completion=1024,
        ),
        vrcli_enabled=True,
        vrcli_weight=0.3,
        data_path_to_save_groups="test_debug_output.jsonl",
        use_parallel_processing=False,  # Sequential for debugging
        use_wandb=False,
    )
    
    print("Creating TextWorldEnv...")
    try:
        env = TextWorldEnv(
            config=env_config, 
            server_configs=[server_config],
            slurm=False
        )
        print("✓ TextWorldEnv created successfully")
    except Exception as e:
        print(f"✗ Failed to create TextWorldEnv: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nCalling env.setup()...")
    try:
        await env.setup()
        print("✓ env.setup() completed successfully")
    except Exception as e:
        print(f"✗ env.setup() failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nCalling env.get_next_item()...")
    try:
        item = await env.get_next_item()
        if item:
            print(f"✓ Got item with keys: {list(item.keys())}")
        else:
            print("✗ get_next_item() returned None")
            return False
    except Exception as e:
        print(f"✗ get_next_item() failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nCalling env.collect_trajectories()...")
    try:
        trajectories, _ = await env.collect_trajectories(item)
        print(f"✓ collect_trajectories() returned {len(trajectories) if trajectories else 0} trajectories")
    except Exception as e:
        print(f"✗ collect_trajectories() failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nCleaning up...")
    try:
        await env.cleanup()
        print("✓ Cleanup completed")
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
    
    return True


async def test_process_mode():
    """Test the process mode specifically."""
    print("\n=== Testing Process Mode ===")
    
    # Use the same config as the SLURM script
    args = [
        "process",
        "--config", "environments/game_environments/textworld/config_process.yaml",
        "--env.total_steps", "1",
        "--env.group_size", "2",
        "--env.max_num_workers", "1",
        "--env.data_path_to_save_groups", "test_debug_process.jsonl",
        "--env.use_parallel_processing", "false",
        "--env.debug_mode", "true",
        "--slurm", "false",
        "--openai.model_name", "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        "--openai.base_url", "http://localhost:30000/v1",
        "--openai.api_key", "dummy",
    ]
    
    # Import and patch sys.argv
    original_argv = sys.argv
    sys.argv = ["textworld_env.py"] + args
    
    try:
        print("Running TextWorldEnv.cli()...")
        TextWorldEnv.cli()
        print("✓ Process mode completed")
    except Exception as e:
        print(f"✗ Process mode failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv


async def main():
    """Run all tests."""
    print("Starting TextWorld data generation debugging")
    print("=" * 50)
    
    # First test basic initialization
    success = await test_environment_initialization()
    
    if success:
        print("\n" + "=" * 50)
        print("Basic initialization successful, testing process mode...")
        await test_process_mode()
    else:
        print("\nBasic initialization failed, skipping process mode test")


if __name__ == "__main__":
    asyncio.run(main())