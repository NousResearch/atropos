import asyncio
import logging
import os
import random

from dotenv import load_dotenv

from atroposlib.envs.base import OpenaiConfig
from environments.game_environments.textarena.tictactoe_env import (
    UltimateTicTacToeEnv,
    UltimateTicTacToeEnvConfig,
)

load_dotenv() # Load environment variables from .env file

# Configure logging to show debug messages from the environment
logging.basicConfig(level=logging.DEBUG)
# Reduce verbosity of very noisy loggers if needed
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.INFO) 


logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting Ultimate Tic Tac Toe environment local debug runner")

    env_config = UltimateTicTacToeEnvConfig(
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct", # Using a Llama-3 based tokenizer
        group_size=2,  # Number of game rollouts per initial seed for GRPO-style data
        use_wandb=False,
        wandb_name="utt_local_debug",
        max_num_workers=1, # Not strictly used in this direct call script
        rollout_server_url="http://localhost:9999", # Placeholder, not used
        total_steps=1, # Not strictly used
        batch_size=1, # Not strictly used
        steps_per_eval=0, # No evaluation in this script
        max_token_length=4096,
        inference_weight=1.0, # Not strictly used
        data_path_to_save_groups=None, # Don't save groups for this test
        # eval_handling=EvalHandlingEnum.NONE, # BaseEnvConfig has this, not UTTT directly
        # eval_limit_ratio=0.0,
        max_episode_actions=40,  # Limit total actions in a game to prevent very long games
        temperature=0.7, # As set by user
        eval_episodes=0, # No evaluation
        include_messages=True # So we can see the dialogue
    )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set. Please set it to run this test.")
        return

    server_configs = [
        OpenaiConfig(
            model_name="gpt-4.1-nano", 
            base_url="https://api.openai.com/v1",
            api_key=openai_api_key,
            num_requests_for_eval=0,
        )
    ]
    logger.info(f"Using model: {server_configs[0].model_name}")
    # logger.debug(f"Env Config: {env_config.model_dump_json(indent=2)}")
    # logger.debug(f"Server Configs: {server_configs[0].model_dump_json(indent=2)}")

    try:
        env = UltimateTicTacToeEnv(
            config=env_config,
            server_configs=server_configs,
            slurm=False, # Not using Slurm for local test
            testing=True # Can enable testing mode if it has specific behavior
        )
    except Exception as e:
        logger.exception(f"Failed to initialize UltimateTicTacToeEnv: {e}")
        return

    logger.info("Running collect_trajectories directly for one item (seed)")
    try:
        await env.setup() # Sets up tokenizer etc.
        seed = random.randint(0, 10000)
        item_to_collect = {"seed": seed}
        logger.info(f"Using initial seed: {seed} for the group of {env_config.group_size} games.")

        # This will run `env_config.group_size` games
        result_groups_list, _ = await env.collect_trajectories(item_to_collect)

        logger.info(
            f"Trajectory collection complete. Received {len(result_groups_list)} ScoredDataGroup(s)."
        )

        if result_groups_list and len(result_groups_list) == 2: # Expecting one for P0, one for P1
            group_p0 = result_groups_list[0]
            group_p1 = result_groups_list[1]

            if group_p0 and group_p1:
                logger.info("\n========== ScoredDataGroup for Player 0 Perspective ==========")
                if group_p0.get("scores"):
                    logger.info(f"  Scores (P0): {group_p0['scores']}")
                if group_p0.get("messages"):
                    logger.info(f"  Number of game dialogues in group: {len(group_p0['messages'])}")
                    for i, game_dialogue in enumerate(group_p0['messages']):
                        logger.info(f"    --- Game Dialogue {i+1}/{len(group_p0['messages'])} (P0 Perspective) ---")
                        for msg_idx, msg in enumerate(game_dialogue):
                            logger.info(f"      Msg {msg_idx}: Role: {msg.get('role')}, Content (first 100 chars): {str(msg.get('content'))[:100]}...")
                else:
                    logger.info("  No messages included in P0 group (set include_messages=True in config)")

                logger.info("\n========== ScoredDataGroup for Player 1 Perspective ==========")
                if group_p1.get("scores"):
                    logger.info(f"  Scores (P1): {group_p1['scores']}")
                if group_p1.get("messages"):
                    logger.info(f"  Number of game dialogues in group: {len(group_p1['messages'])}")
                    for i, game_dialogue in enumerate(group_p1['messages']):
                        logger.info(f"    --- Game Dialogue {i+1}/{len(group_p1['messages'])} (P1 Perspective) ---")
                        # Messages are the same, only scores and masks differ per perspective
                        # For brevity, we can just confirm the messages are present
                        # for msg_idx, msg in enumerate(game_dialogue):
                        #     logger.info(f"      Msg {msg_idx}: Role: {msg.get('role')}, Content (first 100 chars): {str(msg.get('content'))[:100]}...") 
                else:
                    logger.info("  No messages included in P1 group (set include_messages=True in config)")
            else:
                logger.error("One or both ScoredDataGroups are None.")

        elif result_groups_list and len(result_groups_list) == 1 and result_groups_list[0] is None:
             logger.error("Trajectory collection returned [[None]], indicating a group failure.")
        else:
            logger.error(f"Trajectory collection returned an unexpected result: {result_groups_list}")

        logger.info("\n========== Episode Outcomes Buffer (P0_reward, P1_reward from each game) ==========")
        if env.episode_outcomes_buffer:
            for i, outcome_pair in enumerate(env.episode_outcomes_buffer):
                logger.info(f"  Game {i+1}: P0 Reward: {outcome_pair[0]}, P1 Reward: {outcome_pair[1]}")
        else:
            logger.info("  Episode outcomes buffer is empty.")
        logger.info("=======================================================================================")

    except Exception as e:
        logger.exception(
            f"An error occurred during trajectory collection or summary: {e}"
        )


if __name__ == "__main__":
    asyncio.run(main()) 