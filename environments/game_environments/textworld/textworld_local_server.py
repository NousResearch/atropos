import asyncio
import logging
import os
from dotenv import load_dotenv
from atroposlib.envs.base import APIServerConfig
from .textworld_env import TextWorldEnv, TextWorldEnvConfig
from .agents.atropos_agent import AtroposAgentConfig
from .agents.atropos_rm import AtroposRMConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run a complete TextWorld episode for testing."""
    logger.info("Starting TextWorld Environment Test")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return

    server_config = APIServerConfig(
        model_name="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        num_requests_for_eval=0,
        server_type="openai"
    )

    env_config = TextWorldEnvConfig(
        tokenizer_name="NousResearch/Hermes-2-Pro-Llama-3-8B",
        max_token_length=4096,
        max_steps=100,
        challenge_name="tw-simple",
        debug_mode=True,
        default_server_config=server_config,
        atropos_agent_config=AtroposAgentConfig(enable_memory=True),
        atropos_rm_config=AtroposRMConfig(thinking=False),
        group_size=2,
        enable_policy_thinking_summarization=True,
        max_policy_thinking_summary_tokens=128,
    )

    try:
        env = TextWorldEnv(
            config=env_config,
            server_configs=[server_config],
            slurm=False
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
        logger.info(f"  Score: {episode_state.last_score}, Moves: {episode_state.moves}")
        logger.info(f"  Turns: {len(policy_sdgs_for_episode) if policy_sdgs_for_episode else 0}")

        if final_policy_data:
            logger.info("\nPolicy Agent Results:")
            for turn_num, sdg in enumerate(final_policy_data):
                if sdg:
                    metadata = sdg.get("metadata", {})
                    chosen_idx = metadata.get("chosen_alternative_index", -1)
                    scores = sdg.get("scores", [])
                    messages = sdg.get("messages", [])

                    logger.info(f"  Turn {turn_num + 1}:")
                    logger.info(f"    Chosen alternative: {chosen_idx}")
                    logger.info(f"    Scores: {scores}")

                    if chosen_idx != -1 and 0 <= chosen_idx < len(messages):
                        chosen_alt_messages = messages[chosen_idx]
                        if chosen_alt_messages and len(chosen_alt_messages) >= 2:
                            logger.info(f"    Input: '{chosen_alt_messages[-2].get('content', '')[:200]}...'")
                            logger.info(f"    Output: '{chosen_alt_messages[-1].get('content', '')[:200]}...'")

        logger.info(f"\nRM Judgements: {len(episode_state.rm_judgement_history)}")
        for i, judgement in enumerate(episode_state.rm_judgement_history):
            logger.info(f"  Judgement {i+1}:")
            logger.info(f"    Q-value: {judgement.get('parsed_q_value', 'N/A')}")
            if judgement.get('api_error'):
                logger.info("    API Error: True")
            if judgement.get('q_value_parse_error'):
                logger.info("    Parse Error: True")

    except Exception as e:
        logger.error(f"Error during episode: {e}")
    finally:
        await env.cleanup()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())