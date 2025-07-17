import asyncio
import logging
import os

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig

from .agents.atropos_agent import AtroposAgentConfig
from .textworld_env import TextWorldEnv, TextWorldEnvConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run a complete TextWorld episode for testing."""
    logger.info("Starting TextWorld Environment Test")

    # Set debug logging for more info
    logging.getLogger("environments.game_environments.textworld").setLevel(
        logging.DEBUG
    )
    logging.getLogger("atroposlib.utils.tool_call_parser").setLevel(logging.DEBUG)

    # Using local SGLang server with DeepHermes-3-Llama-3-8B
    server_config = APIServerConfig(
        model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        base_url="http://localhost:30000/v1",
        api_key="dummy",  # SGLang doesn't need a real API key
        num_requests_for_eval=0,
        server_type="openai",  # SGLang is OpenAI-compatible
    )

    env_config = TextWorldEnvConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        max_token_length=16384,  # Increased for long thinking
        max_steps=20,  # Reduced for quicker testing
        # Use the new registry system for game selection
        use_registry=True,
        registry_mode="random",  # Randomly select between challenges and generated games
        registry_generation_ratio=0.7,  # 70% generated, 30% pre-built challenges
        registry_difficulty="random",  # Random difficulty
        # Old challenge system - will be ignored when use_registry=True
        challenge_name="tw-simple",
        challenge_rewards="sparse",  # Using sparse rewards with VR-CLI
        debug_mode=True,
        default_server_config=server_config,
        atropos_agent_config=AtroposAgentConfig(enable_memory=True),
        group_size=3,  # Test with 3 alternatives
        vrcli_enabled=True,
        vrcli_weight=0.7,  # 70% VR-CLI, 30% environment reward
        vrcli_discount_factor=0.99,
        enable_policy_thinking_summarization=True,
        max_policy_thinking_summary_tokens=128,
    )

    try:
        env = TextWorldEnv(
            config=env_config, server_configs=[server_config], slurm=False
        )
    except Exception as e:
        logger.error(f"Failed to initialize TextWorldEnv: {e}")
        return

    try:
        await env.setup()
        logger.info("Environment setup complete")

        item = await env.get_next_item()
        if not item or "episode_state" not in item:
            logger.error("Failed to create new episode")
            await env.cleanup()
            return

        episode_state = item["episode_state"]
        episode_id = item["episode_id"]

        logger.info(f"Starting Episode: {episode_id}")
        logger.info(f"Objective: {episode_state.initial_infos.get('objective', 'N/A')}")

        policy_sdgs_for_episode, _ = await env.collect_trajectories(item)
        final_policy_data = await env.postprocess_histories(policy_sdgs_for_episode)

        logger.info(f"\nEpisode Summary: {episode_id}")
        logger.info(f"  Status: Won={episode_state.won}, Lost={episode_state.lost}")
        logger.info(
            f"  Score: {episode_state.last_score}, Moves: {episode_state.moves}"
        )
        logger.info(
            f"  Turns: {len(policy_sdgs_for_episode) if policy_sdgs_for_episode else 0}"
        )

        if final_policy_data:
            logger.info("\nPolicy Agent Results with VR-CLI:")
            for turn_num, sdg in enumerate(final_policy_data):
                if sdg:
                    metadata = sdg.get("metadata", {})
                    chosen_idx = metadata.get("chosen_alternative_index", -1)
                    scores = sdg.get("scores", [])
                    messages = sdg.get("messages", [])
                    vrcli_scores = metadata.get("vrcli_scores", [])
                    env_rewards = metadata.get("env_rewards", [])

                    logger.info(f"\n  Turn {turn_num + 1}:")
                    logger.info(f"    Chosen alternative: {chosen_idx}")
                    logger.info(f"    Combined scores: {[f'{s:.3f}' for s in scores]}")
                    logger.info(
                        f"    VR-CLI scores: {[f'{s:.3f}' for s in vrcli_scores]}"
                    )
                    logger.info(f"    Env rewards: {[f'{s:.3f}' for s in env_rewards]}")

                    # Display all alternatives with their predictions
                    for alt_idx, alt_messages in enumerate(messages):
                        if alt_messages and len(alt_messages) >= 1:
                            last_msg = alt_messages[-1].get("content", "")
                            # Extract action and prediction from the tool call
                            action_start = last_msg.find('"command":')
                            prediction_start = last_msg.find('"expected_outcome":')

                            chosen_marker = " [CHOSEN]" if alt_idx == chosen_idx else ""
                            logger.info(f"\n    Alternative {alt_idx}{chosen_marker}:")

                            if action_start != -1 and prediction_start != -1:
                                # Parse the action and prediction
                                import json

                                try:
                                    tool_call_start = last_msg.find('{"name"')
                                    tool_call_end = last_msg.rfind("}")
                                    if tool_call_start != -1 and tool_call_end != -1:
                                        tool_call_str = last_msg[
                                            tool_call_start : tool_call_end + 1
                                        ]
                                        tool_call = json.loads(tool_call_str)
                                        action = tool_call.get("arguments", {}).get(
                                            "command", "N/A"
                                        )
                                        prediction = tool_call.get("arguments", {}).get(
                                            "expected_outcome", "N/A"
                                        )
                                        logger.info(f"      Action: {action}")
                                        logger.info(
                                            f"      Prediction: {prediction[:100]}..."
                                        )
                                except:
                                    logger.info(f"      Raw: {last_msg[:200]}...")

    except Exception as e:
        logger.error(f"Error during episode: {e}")
    finally:
        await env.cleanup()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
