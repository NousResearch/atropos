import sys
import os
from pathlib import Path

# 1. Add hermes-agent and atropos to sys.path
repo_root = Path("/home/ruffy-369/NousResearch/hermes-agent")
atropos_root = Path("/home/ruffy-369/NousResearch/atropos")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(atropos_root) not in sys.path:
    sys.path.insert(0, str(atropos_root))

try:
    from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
    from atroposlib.envs.server_handling.server_manager import APIServerConfig
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_init():
    print("Testing HermesAgentBaseEnv initialization...")
    try:
        # Create a config
        config = HermesAgentEnvConfig(
            tokenizer_name="gpt2",
            group_size=4
        )
        
        # Verify inheritance of new Atropos fields
        print(f"Checking inherited fields: reward_mode={config.reward_mode}, track_api_perf={config.track_api_perf}")
        
        # Mock server configs
        server_configs = [APIServerConfig(model_name="mock", base_url="http://localhost:8000", api_key="test")]
        
        # Initialize (with testing=True to use ServerHarness)
        env = HermesAgentBaseEnv(config, server_configs, testing=True)
        print("✅ Initialization successful")
        
        # Verify wandb_log signature compatibility
        print("Checking wandb_log signature compat...")
        import asyncio
        asyncio.run(env.wandb_log({}))
        print("✅ wandb_log call successful")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_init()
    print("🚀 hermes-agent Compatibility Check PASSED!")
