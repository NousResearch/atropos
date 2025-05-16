import asyncio
import logging
import os
import random
from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig
from environments.game_environments.textworld.textworld_env import TextWorldEnv, TextWorldEnvConfig
from environments.agents.atropos_agent import AtroposAgentConfig
from environments.agents.atropos_rm import AtroposRMConfig, RMJudgementLog

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
    )

    # Environment Configuration
    env_config = TextWorldEnvConfig(
        # Tokenizer and LLM settings from TextWorldEnv.config_init or suitable defaults
        tokenizer_name="NousResearch/Hermes-2-Pro-Llama-3-8B", # Consistent with TextWorldEnv.config_init
        max_token_length=4096, # Max tokens for an LLM call (agent/RM), from TextWorldEnv.config_init
        
        # Test-specific settings
        max_steps=10,  # Keep episodes short for testing
        challenge_name="tw-simple", # A simple, quick challenge
        debug_mode=True,

        # Atropos Agent and RM settings
        default_server_config=gpt4_mini_server_config, # Agent and RM will use this by default
        # policy_agent_server_config=None, # Explicitly None to use default
        # rm_agent_server_config=None,     # Explicitly None to use default
        
        atropos_agent_config=AtroposAgentConfig(), # Use default agent config
        atropos_rm_config=AtroposRMConfig(thinking=False), # RM thinking off for cleaner logs initially
        
        G_policy_alternatives=2, # Number of actions agent generates
        G_rm_judgements=1,       # Number of RM scores per action
        
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
        # episode_seed = 12345 # Optionally set a seed for reproducibility
        # episode_state = await env._get_or_create_episode(episode_seed=episode_seed)
        item = await env.get_next_item() # Uses env_config.game_seed or random

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
        
        # The initial message history in episode_state already contains:
        # 1. System Prompt (from policy_agent_system_prompt_content)
        # 2. User message with the initial observation
        logger.info(f"""Policy Agent System Prompt:
{episode_state.message_history[0]['content']}""")
        logger.info(f"""Initial Observation (Turn {episode_state.current_turn + 1}):
{episode_state.message_history[1]['content']}""")

        # Main interaction loop
        while not episode_state.done:
            current_turn_for_log = episode_state.current_turn + 1
            logger.info(f"\n<<< --- Turn: {current_turn_for_log}/{episode_state.max_turns} --- >>>")
            
            scored_data_group, episode_done = await env._next_step(
                ep_state=episode_state, 
                current_turn_num=episode_state.current_turn # _next_step expects 0-indexed current_turn
            )
            
            if scored_data_group:
                logger.info(f"--- Turn {current_turn_for_log} Policy Agent Output & RM Evaluation ---")
                
                # Access metadata safely, handling both attribute and dictionary access
                chosen_alternative_idx = -1
                if hasattr(scored_data_group, 'metadata'):
                    chosen_alternative_idx = scored_data_group.metadata.get("chosen_alternative_index", -1)
                elif isinstance(scored_data_group, dict) and "metadata" in scored_data_group:
                    chosen_alternative_idx = scored_data_group["metadata"].get("chosen_alternative_index", -1)
                
                # Access messages safely, handling both attribute and dictionary access
                messages = []
                if hasattr(scored_data_group, 'messages'):
                    messages = scored_data_group.messages
                elif isinstance(scored_data_group, dict) and "messages" in scored_data_group:
                    messages = scored_data_group["messages"]
                
                # Access scores safely
                scores = []
                if hasattr(scored_data_group, 'scores'):
                    scores = scored_data_group.scores
                elif isinstance(scored_data_group, dict) and "scores" in scored_data_group:
                    scores = scored_data_group["scores"]
                
                if chosen_alternative_idx != -1 and chosen_alternative_idx < len(messages):
                    # The chosen alternative's history is in messages[chosen_alternative_idx]
                    # The last message is the agent's raw response.
                    chosen_agent_raw_response = messages[chosen_alternative_idx][-1]['content']
                    # Re-parse the command from the raw response for logging
                    parsed_command_executed = env._parse_action(chosen_agent_raw_response)
                    
                    logger.info(f"  Chosen Alternative Index: {chosen_alternative_idx}")
                    logger.info(f"  Agent's Raw Response (Chosen): '{chosen_agent_raw_response.strip()}'")
                    logger.info(f"  Parsed Command Executed: '{parsed_command_executed if parsed_command_executed else 'None/Error'}'")
                    logger.info(f"  RM Score for Chosen: {scores[chosen_alternative_idx]:.4f}")
                else:
                    logger.warning("  Could not determine chosen action details from ScoredDataGroup metadata.")

                logger.info("  --- All Generated Policy Alternatives & RM Scores ---")
                for i, alt_messages in enumerate(messages):
                    alt_raw_response = alt_messages[-1]['content'] # Last message is agent's output for this alternative
                    alt_parsed_cmd = env._parse_action(alt_raw_response)
                    alt_score = scores[i]
                    is_chosen = "(CHOSEN)" if i == chosen_alternative_idx else ""
                    logger.info(f"    Alt {i} {is_chosen}: Command='{alt_parsed_cmd}', Raw='{alt_raw_response.strip()}', Score={alt_score:.4f}")
            else:
                logger.warning(f"  _next_step for turn {current_turn_for_log} did not return a ScoredDataGroup (episode might have ended due to error).")


            logger.info(f"--- Turn {current_turn_for_log} Environment Response ---")
            if not episode_state.done and episode_state.message_history[-1]['role'] == 'user':
                # The last message in ep_state.message_history is now the new user observation
                logger.info(f"""Observation for Next Turn ({episode_state.current_turn + 1}):
{episode_state.message_history[-1]['content']}""")
            elif episode_state.done:
                 logger.info("Episode is now DONE.")

            logger.info(f"  State: Score={episode_state.last_score}, Moves={episode_state.moves}, Won={episode_state.won}, Lost={episode_state.lost}, Done={episode_state.done}")

            if episode_done and not episode_state.done: # Should be consistent
                logger.warning(f"  _next_step reported episode_done={episode_done} but ep_state.done={episode_state.done}. Syncing.")
                episode_state.done = True


        # End of episode
        logger.info(f"\n<<< --- Episode Finished: {episode_id} --- >>>")
        logger.info(f"  Final Status: Won={episode_state.won}, Lost={episode_state.lost}")
        logger.info(f"  Total Turns Taken: {episode_state.current_turn}")
        logger.info(f"  Final Score: {episode_state.last_score}")
        logger.info(f"  Total Moves Reported by Env: {episode_state.moves}")
        
        logger.info("\n--- Full RM Judgement History for Episode ---")
        if episode_state.rm_judgement_history:
            for i, judgement_log_dict in enumerate(episode_state.rm_judgement_history):
                # Convert dict to RMJudgementLog TypedDict for easier access if needed, or access by key
                # For now, direct dict access
                judgement = RMJudgementLog(**judgement_log_dict) if isinstance(judgement_log_dict, dict) else judgement_log_dict

                logger.info(f"  Judgement {i+1}:")
                # Assuming judgement is now a dict-like object or RMJudgementLog instance
                if hasattr(judgement, 'get') or isinstance(judgement, dict): # Check if it's dict-like
                    input_msgs = judgement.get('llm_input_messages', [])
                    last_user_msg_content = "N/A"
                    if input_msgs and isinstance(input_msgs, list) and len(input_msgs) > 0 and isinstance(input_msgs[-1], dict):
                        last_user_msg_content = input_msgs[-1].get('content', 'N/A')
                    
                    logger.info(f"    Policy Action Evaluated (from history): {last_user_msg_content[:300].strip()}...")
                    logger.info(f"    Raw LLM Output (RM): {str(judgement.get('raw_llm_output_content', 'N/A'))[:300].strip()}...")
                    logger.info(f"    Parsed Q-value: {judgement.get('parsed_q_value', 'N/A')}")
                    logger.info(f"    Parsed Thinking: {str(judgement.get('parsed_thinking_content', 'N/A'))[:300].strip()}...")
                    if judgement.get('api_error'): logger.info("    API Error: True")
                    if judgement.get('q_value_parse_error'): logger.info("    Q-Value Parse Error: True")
                else: # Fallback if it's not a dict (e.g. already an RMJudgementLog object, though unlikely here)
                    logger.info(f"    Log Entry (type {type(judgement)}): {str(judgement)[:300]}...")
        else:
            logger.info("  No RM judgements recorded in episode state.")

    except Exception as e:
        logger.exception(f"An error occurred during the episode run: {e}")
    finally:
        logger.info("Cleaning up TextWorld environment...")
        await env.cleanup()
        logger.info("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(main()) 