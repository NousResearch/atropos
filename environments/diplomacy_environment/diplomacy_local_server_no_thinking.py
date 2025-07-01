#!/usr/bin/env python3
"""
Local test script for Diplomacy Environment (No Thinking version)

This script demonstrates how to run a single Diplomacy game with:
- One Atropos policy agent being trained (France)
- Six strong LLM opponents
- Optional web interface for watching the game
"""

import asyncio
import logging
import os
import subprocess
import sys
from typing import Optional
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum, ScoredDataItem
from environments.diplomacy_environment.diplomacy_env_no_thinking import (
    DiplomacyEnvNoThinking,
    DiplomacyEnvNoThinkingConfig,
    PowerConfig,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main(max_turns=10, start_mock_server=True):
    """Run a single Diplomacy game with mixed agents."""
    logger.info("Starting Diplomacy (No Thinking) environment local debug runner")
    
    # Configure the environment
    env_config = DiplomacyEnvNoThinkingConfig(
        # Tokenizer settings
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        max_token_length=4096,
        
        # Training settings
        group_size=1,  # One game at a time
        use_wandb=False,  # Disable wandb for local testing
        wandb_name="diplomacy_no_thinking_local_debug",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        
        # Game settings
        max_game_turns=max_turns,  # Configurable for testing
        game_deadline_seconds=120,  # 2 minutes per phase for testing
        
        # Power configurations - France is our training agent
        # Using only OpenAI models to avoid multiple API key requirements
        powers_config={
            "FRANCE": PowerConfig(
                type="atropos",
                model="training-policy",
                is_training=True
            ),
            "ENGLAND": PowerConfig(
                type="llm",
                model="gpt-4o",
                is_training=False
            ),
            "GERMANY": PowerConfig(
                type="llm",
                model="gpt-4o-mini",
                is_training=False
            ),
            "ITALY": PowerConfig(
                type="llm",
                model="gpt-4o",
                is_training=False
            ),
            "AUSTRIA": PowerConfig(
                type="llm",
                model="gpt-4o-mini",
                is_training=False
            ),
            "RUSSIA": PowerConfig(
                type="llm",
                model="gpt-4o",
                is_training=False
            ),
            "TURKEY": PowerConfig(
                type="llm",
                model="gpt-4o-mini",
                is_training=False
            )
        },
        
        # Server settings
        launch_web_server=True,  # Enable web interface
        web_server_port=8432,
        atropos_server_url="http://localhost:8000",
        
        # Scoring settings
        score_by_supply_centers=True,
        survival_bonus=0.1,
        win_bonus=10.0,
        
        # Logging
        save_game_logs=True,
        game_logs_dir="./game_logs_debug",
        include_messages=True,  # Include messages in output for debugging
        
        # No evaluation in debug mode
        eval_episodes=0,
    )
    
    # Configure API servers
    # For local testing, we'll use GPT-4o-mini as our test policy
    server_configs = [
        APIServerConfig(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        )
    ]
    
    # You may want to add API keys for the LLM opponents if testing with real APIs
    # For now, AI_Diplomacy will use environment variables for these
    
    logger.info("Configuration:")
    logger.info(f"  Training power: FRANCE (Atropos policy)")
    logger.info(f"  Opponents: OpenAI models (GPT-4o and GPT-4o-mini)")
    logger.info(f"  Max turns: {env_config.max_game_turns}")
    logger.info(f"  Web interface: http://localhost:{env_config.web_server_port}")
    logger.info(f"  Game logs: {env_config.game_logs_dir}")
    
    # Start mock Atropos server if requested
    mock_server_process = None
    if start_mock_server:
        logger.info("Starting mock Atropos server...")
        mock_server_process = subprocess.Popen(
            [sys.executable, "mock_atropos_server.py"],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        await asyncio.sleep(2)  # Give server time to start
        logger.info("Mock Atropos server started at http://localhost:8000")
    
    try:
        # Initialize environment
        env = DiplomacyEnvNoThinking(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=False,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize DiplomacyEnvNoThinking: {e}")
        return
    
    logger.info("\n" + "="*60)
    logger.info("IMPORTANT: Make sure you have set API keys in environment variables:")
    logger.info("   - OPENAI_API_KEY (for GPT-4o and GPT-4o-mini)")
    logger.info("="*60 + "\n")
    
    await asyncio.sleep(2)  # Give user time to read
    
    try:
        # Set up environment (starts web server)
        await env.setup()
        
        # Generate a game configuration
        item_for_env = await env.get_next_item()
        logger.info(f"Starting game: {item_for_env}")
        
        if env_config.launch_web_server:
            logger.info(f"\n🎮 Watch the game at: http://localhost:{env_config.web_server_port}")
            logger.info("   Create a new game and use the same game ID to spectate\n")
        
        # Run a single game
        logger.info("Running game (this may take several minutes)...")
        result_tuple = await env.collect_trajectory(item_for_env)
        
        # Process results
        scored_data_item: Optional[ScoredDataItem] = None
        if result_tuple and result_tuple[0]:
            scored_data_item = result_tuple[0]
            score = scored_data_item.get('scores', 0)
            
            logger.info(f"\n✅ Game completed! France (training agent) score: {score:.2f}")
            
            # Show collected messages if available
            if env_config.include_messages and scored_data_item.get("messages"):
                logger.info("\nCollected Messages:")
                for i, msg in enumerate(scored_data_item["messages"][:5]):  # First 5 messages
                    content = str(msg['content'])[:150]
                    logger.info(f"  {i}. Role: {msg['role']}, Content: '{content}...'")
                if len(scored_data_item["messages"]) > 5:
                    logger.info(f"  ... and {len(scored_data_item['messages']) - 5} more messages")
            
            # Show token info
            logger.info(f"\nTokenization:")
            logger.info(f"  Tokens: {len(scored_data_item.get('tokens', []))}")
            logger.info(f"  Masks: {len(scored_data_item.get('masks', []))}")
        else:
            logger.error("❌ Game failed to produce trajectory data")
        
        # Show game summary from buffer
        if env.game_outcomes_buffer:
            outcome = env.game_outcomes_buffer[-1]
            logger.info("\n" + "="*60)
            logger.info("GAME SUMMARY")
            logger.info("="*60)
            logger.info(f"Game ID: {outcome['game_id']}")
            logger.info(f"Winner: {outcome.get('winner', 'No winner')}")
            logger.info(f"Turns played: {outcome['turns']}")
            logger.info(f"\nFinal supply centers:")
            for power, centers in outcome['final_centers'].items():
                is_training = " 🎯" if power in outcome['training_powers'] else ""
                logger.info(f"  {power}: {centers} centers{is_training}")
            logger.info(f"\nTraining agent scores:")
            for power, score in outcome['scores'].items():
                logger.info(f"  {power}: {score:.2f}")
            logger.info("="*60)
            
            if env_config.save_game_logs:
                logger.info(f"\n📁 Full game log saved to: {env_config.game_logs_dir}/{outcome['game_id']}.json")
        
    except Exception as e:
        logger.exception(f"An error occurred during game execution: {e}")
    
    finally:
        # Clean up
        if hasattr(env, 'web_server_process') and env.web_server_process:
            logger.info("\nShutting down web server...")
            env.web_server_process.terminate()
            env.web_server_process.wait()
        
        if mock_server_process:
            logger.info("Shutting down mock Atropos server...")
            mock_server_process.terminate()
            mock_server_process.wait()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a local Diplomacy test game")
    parser.add_argument("--max-turns", type=int, default=10, 
                       help="Maximum number of game turns (default: 10)")
    parser.add_argument("--no-mock-server", action="store_true",
                       help="Don't start the mock Atropos server")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode - only 2 turns")
    
    args = parser.parse_args()
    
    max_turns = 2 if args.quick else args.max_turns
    start_mock_server = not args.no_mock_server
    
    print("\n🎮 DIPLOMACY LOCAL TEST RUNNER 🎮\n")
    print("This will run a single Diplomacy game with:")
    print("- France: Atropos training policy")
    print("- Other powers: OpenAI models (GPT-4o and GPT-4o-mini)")
    print("- Web interface for watching the game")
    print(f"- Maximum turns: {max_turns}")
    print(f"- Mock server: {'Yes' if start_mock_server else 'No'}")
    print("\nPress Ctrl+C to cancel, or wait to continue...\n")
    
    try:
        asyncio.run(main(max_turns=max_turns, start_mock_server=start_mock_server))
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")