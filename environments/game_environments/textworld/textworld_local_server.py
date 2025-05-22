import asyncio
import logging
import os
import random
from dotenv import load_dotenv

import sys
# Calculate the project root directory (three levels up from the script's directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from atroposlib.envs.base import APIServerConfig
from environments.game_environments.textworld.textworld_env import TextWorldEnv, TextWorldEnvConfig
# Use absolute imports now that project root is in sys.path
from environments.game_environments.textworld.agents.atropos_agent import AtroposAgentConfig 
from environments.game_environments.textworld.agents.atropos_rm import AtroposRMConfig, RMJudgementLog

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("--- Starting TextWorld Environment Local Test Runner ---")

    # API Server Configuration (using gpt-4.1-mini as requested)
    # Ensure OPENAI_API_KEY and optionally OPENAI_API_BASE_URL are in your .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
        return

    gpt4_mini_server_config = APIServerConfig(
        model_name="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        num_requests_for_eval=0,
        server_type="openai"
    )

    # Environment Configuration
    env_config = TextWorldEnvConfig(
        # Tokenizer and LLM settings from TextWorldEnv.config_init or suitable defaults
        tokenizer_name="NousResearch/Hermes-2-Pro-Llama-3-8B", # Consistent with TextWorldEnv.config_init
        max_token_length=4096, # Max tokens for an LLM call (agent/RM), from TextWorldEnv.config_init
        
        # Test-specific settings
        max_steps=100,  # Increased for more complete episodes
        challenge_name="tw-simple", # A simple, quick challenge
        debug_mode=True,

        # Atropos Agent and RM settings
        default_server_config=gpt4_mini_server_config, # Agent and RM will use this by default
        # policy_agent_server_config=None, # Explicitly None to use default
        # rm_agent_server_config=None,     # Explicitly None to use default
        
        atropos_agent_config=AtroposAgentConfig(enable_memory=False), # Use default agent config, ensure memory is off
        atropos_rm_config=AtroposRMConfig(thinking=False), # RM thinking off for cleaner logs initially
        
        group_size=2, # Explicitly set group_size for policy alternatives, matching old G_policy_alternatives for this test script
        
        # Data processing configuration for testing
        enable_policy_thinking_summarization=True,  # Enable to test LLM-based thinking summarization
        max_policy_thinking_summary_tokens=128,
        
        # Other relevant fields from TextWorldEnvConfig defaults if needed
        # game_seed=None, # Use random seed by default for variety unless specific seed needed
        # max_trajectory_tokens = 24576, # Default from TextWorldEnvConfig
        # rm_reward_discount_factor = 0.99, # Default
    )

    logger.info(f"Using Environment Configuration: {env_config.model_dump_json(indent=2)}")
    logger.info(f"Policy Agent will use model: {gpt4_mini_server_config.model_name}")
    logger.info(f"RM Agent will use model: {gpt4_mini_server_config.model_name}")


    # Initialize TextWorld Environment
    # The server_configs list should contain all unique APIServerConfig instances the env might use.
    # Since policy and RM are configured to use the default_server_config, we pass that one.
    try:
        env = TextWorldEnv(
            config=env_config,
            server_configs=[gpt4_mini_server_config], 
            slurm=False # Explicitly set for local testing
        )
    except Exception as e:
        logger.exception(f"Failed to initialize TextWorldEnv: {e}")
        return

    try:
        # Setup environment (e.g., agent model loading, TextWorld checks)
        await env.setup()
        logger.info("TextWorldEnv setup complete.")

        # Get a new game/episode
        item = await env.get_next_item() 

        if not item or "episode_state" not in item:
            logger.error("Failed to get or create a new TextWorld episode.")
            await env.cleanup()
            return

        episode_state = item["episode_state"]
        episode_id = item["episode_id"]

        logger.info(f"--- Starting Episode: {episode_id} ---")
        logger.info(f"Game File: {episode_state.game_file}")
        game_objective = episode_state.initial_infos.get('objective', 'N/A')
        logger.info(f"Objective: {game_objective.strip() if game_objective else 'N/A'}")
        
        system_prompt_to_log = env.agent.system_prompt_content
        logger.info(f"""Policy Agent System Prompt:
{system_prompt_to_log}""")
        
        logger.info(f"""Initial Observation (Turn 1):
{episode_state.initial_formatted_obs}""")

        # Run the full episode using collect_trajectories
        logger.info(f"--- Running Full Episode Trajectory Collection for Episode: {episode_id} ---")
        policy_sdgs_for_episode, _ = await env.collect_trajectories(item)
        logger.info(f"--- Trajectory Collection Finished for Episode: {episode_id} ---")

        # Post-process the collected trajectories (for policy agent data)
        logger.info(f"--- Post-processing Policy Trajectories for Episode: {episode_id} ---")
        final_policy_data = await env.postprocess_histories(policy_sdgs_for_episode)
        logger.info(f"--- Post-processing Finished for Episode: {episode_id} ---")

        # Log results from final_policy_data
        if final_policy_data:
            logger.info("\n--- Processed Policy Agent Data (from postprocess_histories) ---")
            for turn_num, sdg in enumerate(final_policy_data):
                if sdg: # sdg is a ScoredDataGroup (TypedDict)
                    metadata = sdg.get("metadata")
                    chosen_idx = metadata.get("chosen_alternative_index", -1) if metadata else -1
                    scores = sdg.get("scores", [])
                    messages = sdg.get("messages", [])
                    
                    logger.info(f"  Turn {turn_num + 1} ScoredDataGroup:")
                    logger.info(f"    Chosen Alternative Index (from _next_step metadata): {chosen_idx}")
                    logger.info(f"    Number of alternatives: {len(scores)}")
                    logger.info(f"    Scores: {scores}")
                    if chosen_idx != -1 and 0 <= chosen_idx < len(scores):
                         logger.info(f"    Score of Chosen Alternative (after collect_trajectories): {scores[chosen_idx]:.4f}")
                    
                    if chosen_idx != -1 and 0 <= chosen_idx < len(messages):
                        chosen_alt_messages = messages[chosen_idx]
                        if chosen_alt_messages and len(chosen_alt_messages) >= 2: # Need at least obs and action
                             logger.info(f"    Chosen Alternative Input (last user message): '{chosen_alt_messages[-2].get('content', '')[:200]}...'")
                             logger.info(f"    Chosen Alternative Output (last assistant message): '{chosen_alt_messages[-1].get('content', '')[:200]}...'")
                        elif chosen_alt_messages: # Only one message?
                             logger.info(f"    Chosen Alternative Message (single): '{chosen_alt_messages[0].get('content', '')[:200]}...'")
                else:
                    logger.info(f"  Turn {turn_num + 1}: No ScoredDataGroup found (None).")
        else:
            logger.info("  No policy data returned from postprocess_histories.")


        # Episode summary (already present in the script, good to keep)
        logger.info(f"\n<<< --- Episode Summary: {episode_id} --- >>>")
        logger.info(f"  Final Status: Won={episode_state.won}, Lost={episode_state.lost}")
        # The number of turns is now len(policy_sdgs_for_episode) or len(final_policy_data)
        # ep_state.policy_step_data is populated by _next_step, which is called by collect_trajectories
        logger.info(f"  Total Policy SDGs generated: {len(policy_sdgs_for_episode) if policy_sdgs_for_episode else 0}") 
        logger.info(f"  Total Game Moves in Env: {episode_state.moves}")
        logger.info(f"  Final Game Score: {episode_state.last_score}")
        
        
        logger.info("\n--- Full RM Judgement History for Episode (from episode_state.rm_judgement_history) ---")
        if episode_state.rm_judgement_history:
            for i, judgement_log_dict in enumerate(episode_state.rm_judgement_history):
                # Assuming judgement_log_dict is a dict. RMJudgementLog is a TypedDict.
                judgement = judgement_log_dict 

                logger.info(f"  Judgement {i+1}:")
                input_msgs = judgement.get('rm_input_messages', []) # Corrected field name
                last_user_msg_content = "N/A"
                # The RM input messages are [system_prompt, user_prompt_content_str]
                # user_prompt_content_str contains the policy agent's proposed move
                if input_msgs and isinstance(input_msgs, list) and len(input_msgs) > 1 and isinstance(input_msgs[1], dict):
                    # The actual content with policy action is in the 'user' message to RM
                    rm_user_prompt_str = input_msgs[1].get('content', 'N/A')
                    # Extracting the policy action part for logging
                    action_marker = "--- Policy Agent's Proposed Move (to be evaluated) ---"
                    end_action_marker = "--- End of Proposed Move ---"
                    start_idx = rm_user_prompt_str.find(action_marker)
                    end_idx = rm_user_prompt_str.find(end_action_marker)
                    if start_idx != -1 and end_idx != -1:
                        last_user_msg_content = rm_user_prompt_str[start_idx + len(action_marker) : end_idx].strip()
                    else:
                        last_user_msg_content = rm_user_prompt_str[:300] # Fallback
                
                logger.info(f"    Policy Action Evaluated (part of RM user prompt): {last_user_msg_content[:300].strip()}...")
                logger.info(f"    Raw LLM Output (RM): {str(judgement.get('raw_rm_response_content', 'N/A'))[:300].strip()}...") # Corrected
                logger.info(f"    Parsed Q-value: {judgement.get('parsed_q_value', 'N/A')}")
                logger.info(f"    Parsed Thinking: {str(judgement.get('parsed_thinking_block', 'N/A'))[:300].strip()}...") # Corrected
                if judgement.get('api_error'): logger.info("    API Error: True")
                if judgement.get('q_value_parse_error'): logger.info("    Q-Value Parse Error: True")
        else:
            logger.info("  No RM judgements recorded in episode state.")
        
        logger.info("Note: RM training ScoredDataGroups were generated by collect_trajectories. "
                    "Their successful processing by the trainer API depends on BaseEnv's 'handle_send_to_api' validation (e.g., group_size checks).")

    except Exception as e:
        logger.exception(f"An error occurred during the episode run: {e}")
    finally:
        logger.info("Cleaning up TextWorld environment...")
        await env.cleanup()
        logger.info("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(main()) 