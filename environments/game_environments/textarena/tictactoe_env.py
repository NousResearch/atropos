import json
import logging
import random
from typing import Dict, List, Optional, Tuple, Any

import textarena as ta
from textarena.wrappers import LLMObservationWrapper

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, OpenaiConfig, ScoredDataItem, ScoredDataGroup
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)

# Llama 3 style Jinja chat template adapted for custom game roles
# This will be used by tokenizer.apply_chat_template
LLAMA3_CUSTOM_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

class UltimateTicTacToeEnvConfig(BaseEnvConfig):
    """
    Configuration for the UltimateTicTacToeEnv environment.
    """

    env_id: str = "UltimateTicTacToe-v0"
    max_episode_actions: int = 81 + 5
    eval_episodes: int = 100
    num_players: int = 2
    group_size: int = 16
    # Add temperature if you want to configure it via config
    # temperature: float = 0.5


class UltimateTicTacToeEnv(BaseEnv):
    name = "ultimate_tictactoe"
    env_config_cls = UltimateTicTacToeEnvConfig

    def __init__(
        self,
        config: UltimateTicTacToeEnvConfig,
        server_configs: List[OpenaiConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: UltimateTicTacToeEnvConfig = config
        # Stores tuples of (p0_reward, p1_reward) for each game in training batch
        self.episode_outcomes_buffer: List[Tuple[float, float]] = [] 
        self.eval_metrics_custom: List[Tuple[str, float]] = []

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_move",
                    "description": "Submit your move in Ultimate Tic Tac Toe.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "micro_board": {
                                "type": "integer",
                                "description": "Index of the micro board (0-8)",
                            },
                            "row": {
                                "type": "integer",
                                "description": "Row index in the micro board (0-2)",
                            },
                            "col": {
                                "type": "integer",
                                "description": "Column index in the micro board (0-2)",
                            },
                        },
                        "required": ["micro_board", "row", "col"],
                    },
                },
            }
        ]

        tools_json = json.dumps(self.tools, indent=2)
        self.system_prompt_base = (
            "You are playing Ultimate Tic Tac Toe. "
            "You will receive observations from the game environment which include rules and the current board state. "
            "Your goal is to win the game by strategically placing your marks. "
            "To submit your move, you MUST use the 'submit_move' tool call.\\n\\n"
            f"<tools>\\n{tools_json}\\n</tools>\\n\\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema (example for playing in the center of the top-left micro board (index 0)):\\n"
            '<tool_call>\\n{"arguments": {"micro_board": 0, "row": 1, "col": 1}, "name": "submit_move"}\\n</tool_call>\\n\\n'
            "Your full answer format should be (NO THINKING BLOCK, just the tool call):\\n"
            '<tool_call>\\n{"arguments": {"micro_board": 0, "row": 1, "col": 1}, "name": "submit_move"}\\n</tool_call>\\n'
        )

    @classmethod
    def config_init(cls) -> Tuple[UltimateTicTacToeEnvConfig, List[OpenaiConfig]]:
        env_config = UltimateTicTacToeEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=4096, 
            wandb_name=cls.name,
            steps_per_eval=50,
            max_episode_actions=81 + 10,
            eval_episodes=100,
            num_players=2,
        )
        server_configs = [
            OpenaiConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256, 
            ),
        ]
        return env_config, server_configs

    async def _sample_llm_response(self, sft_history_for_prompt: List[Message], server) -> str:
        if not sft_history_for_prompt: # Renamed from messages_for_template for clarity
            logger.error("No messages to send to LLM.")
            return ""

        # Apply the chat template. If this fails, it's a critical error.
        prompt_str = self.tokenizer.apply_chat_template(
            sft_history_for_prompt,
            tokenize=False,
            chat_template=LLAMA3_CUSTOM_CHAT_TEMPLATE,
            add_generation_prompt=True
        )

        max_tokens_for_llm_output = 512
        try:
            # Assuming server.completion handles default temperature if not provided
            # or uses a fixed/configurable one internally. 
            # If specific temperature control from this env config is needed, ensure 'temperature' is in UltimateTicTacToeEnvConfig.
            completions = await server.completion(
                prompt=prompt_str, 
                n=1, 
                max_tokens=max_tokens_for_llm_output,
                temperature=getattr(self.config, 'temperature', 0.5) # Use 0.5 if not in config
            )
            return completions.choices[0].text.strip() if completions.choices else ""
        except Exception as e:
            logger.error(f"LLM API error during completion: {e}. Prompt (start): {prompt_str[:200]}")
            return ""

    def _parse_action_from_llm(self, llm_response: str) -> Optional[str]:
        """Parses the 'submit_move' tool call and returns action string "[mb r c]"""
        if not llm_response:
            logger.warning("Attempted to parse an empty LLM response.")
            return None
        parsed_name, parsed_args, is_error = parse_tool_call(
            llm_response, self.tools, ["tool_call"]
        )

        if is_error:
            error_detail = (
                str(parsed_name)
                if parsed_name
                else "Parser indicated error, but no specific message was returned."
            )
            logger.warning(
                f"Failed to parse tool call. Full response: '{llm_response}'. Error: {error_detail}"
            )
            return None

        if parsed_name != "submit_move":
            logger.warning(
                f"Expected tool call name 'submit_move', but got '{parsed_name}'. Response: '{llm_response}'"
            )
            return None

        try:
            micro_board = int(parsed_args.get("micro_board"))
            row = int(parsed_args.get("row"))
            col = int(parsed_args.get("col"))

            if not (0 <= micro_board <= 8 and 0 <= row <= 2 and 0 <= col <= 2):
                raise ValueError("Parsed arguments out of valid range.")

            return f"[{micro_board} {row} {col}]"
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Successfully parsed tool call '{parsed_name}', "
                f"but action arguments are invalid or missing. Error: {e}. "
                f"Parsed args: {parsed_args}. Full response: '{llm_response}'."
            )
            return None

    async def collect_trajectories(self, item: Item) -> Tuple[List[Optional[ScoredDataGroup]], List[Item]]:
        initial_seed = item["seed"]
        
        all_game_messages: List[List[Message]] = []
        all_game_rewards_p0: List[float] = []
        all_game_rewards_p1: List[float] = []

        # GRPO: For each prompt (item/initial_seed), generate K (group_size) responses (game rollouts)
        for i in range(self.config.group_size):
            current_game_seed = initial_seed + i # Vary seed slightly for each game in the group, or rely on model stochasticity
            game_sft_messages: List[Message] = []
            current_game_rewards: Dict[int, float] = {0: 0.0, 1: 0.0}
            
            try:
                current_ta_env = ta.make(env_id=self.config.env_id)
                current_ta_env = LLMObservationWrapper(env=current_ta_env)
                # Use current_game_seed for this specific rollout
                _ = current_ta_env.reset(num_players=self.config.num_players, seed=current_game_seed) 
            except Exception as e:
                logger.error(f"TextArena env make/reset failed (seed {current_game_seed}, group_idx {i}): {e}")
                # If one game in the group fails, we might skip it or fill with placeholders.
                # For now, let's skip. This means the resulting ScoredDataGroup might have fewer than group_size entries.
                # Alternatively, fill with error markers / neutral scores if the trainer expects fixed size.
                # For simplicity, we'll allow variable actual group sizes if errors occur.
                continue # Skip this iteration of the group

            done = False
            num_actions_taken = 0
            async with self.server.dedicated_server() as server:
                while not done and num_actions_taken < self.config.max_episode_actions:
                    current_sft_len = len(self.tokenizer.apply_chat_template(game_sft_messages, tokenize=False, add_generation_prompt=False))
                    if current_sft_len > self.config.max_token_length - 700:
                        logger.warning(f"[Seed: {current_game_seed}, GroupIdx: {i}] SFT data approaching max token length. Truncating game.")
                        current_game_rewards = {0: 0.0, 1: 0.0}
                        done = True; break

                    current_player_id, current_player_obs_str = current_ta_env.get_observation()
                    player_mark = 'O' if current_player_id == 0 else 'X'
                    player_system_content = f"You are Player {current_player_id} ('{player_mark}'). {self.system_prompt_base}"

                    turn_messages_for_llm = game_sft_messages + [
                        {"role": "system", "content": player_system_content},
                        {"role": "environment", "content": current_player_obs_str}
                    ]
                    
                    llm_action_response = await self._sample_llm_response(turn_messages_for_llm, server)
                    num_actions_taken +=1

                    game_sft_messages.append({"role": "system", "content": player_system_content})
                    game_sft_messages.append({"role": "environment", "content": current_player_obs_str})

                    if not llm_action_response:
                        logger.error(f"[Seed: {current_game_seed}, P{current_player_id}, GroupIdx: {i}] LLM failed. Player forfeits.")
                        current_game_rewards[current_player_id] = -1.0
                        current_game_rewards[1 - current_player_id] = 1.0
                        game_sft_messages.append({"role": f"player_{current_player_id}", "content": "<LLM_ERROR_NO_RESPONSE>"})
                        done = True; break

                    game_sft_messages.append({"role": f"player_{current_player_id}", "content": llm_action_response})
                    parsed_action_str = self._parse_action_from_llm(llm_action_response)
                    
                    try:
                        ta_player_rewards, truncated, terminated, info = current_ta_env.step(action=parsed_action_str or "[INVALID_PARSE]")
                        current_game_rewards = {pid: float(r) for pid, r in ta_player_rewards.items()}
                        done = terminated or truncated
                        if done: logger.debug(f"[Seed {current_game_seed}, GroupIdx: {i}] Game ended. Reason: {info.get('reason', 'N/A')}. P0: {current_game_rewards.get(0,0)}, P1: {current_game_rewards.get(1,0)}")
                    except Exception as e:
                        logger.error(f"[Seed: {current_game_seed}, P{current_player_id}, GroupIdx: {i}] TextArena step error: {e}")
                        current_game_rewards[current_player_id] = -1.0; current_game_rewards[1 - current_player_id] = 0.0
                        done = True; break
                
                if not done and num_actions_taken >= self.config.max_episode_actions:
                    logger.warning(f"[Seed {current_game_seed}, GroupIdx: {i}] Max actions ({self.config.max_episode_actions}) reached. Draw.")
                    current_game_rewards = {0: 0.0, 1: 0.0}
            
            current_ta_env.close()
            # Store results for this game in the group
            if game_sft_messages: # Only if game produced messages
                all_game_messages.append(game_sft_messages)
                all_game_rewards_p0.append(current_game_rewards.get(0, 0.0))
                all_game_rewards_p1.append(current_game_rewards.get(1, 0.0))
                # Log to buffer for overall training stats
                self.episode_outcomes_buffer.append((current_game_rewards.get(0, 0.0), current_game_rewards.get(1, 0.0)))

        if not all_game_messages: # All games in the group failed or produced no messages
            logger.warning(f"[InitialSeed: {initial_seed}] No games in the group completed successfully.")
            return [[None],[None]], [] # Return list of two None groups

        # Tokenize for Player 0's perspective
        p0_tokens_list: List[List[int]] = []
        p0_masks_list: List[List[int]] = []
        for game_msg_idx, game_dialogue in enumerate(all_game_messages):
            tokenization_result_p0 = tokenize_for_trainer(
                tokenizer=self.tokenizer, 
                chat=game_dialogue, 
                train_on_all_assistant_turns=True, 
                unmasked_role="player_0" # Updated parameter name
            )
            p0_tokens_list.append(tokenization_result_p0["tokens"])
            p0_masks_list.append(tokenization_result_p0["masks"])

        group_p0 = ScoredDataGroup(
            tokens=p0_tokens_list,
            masks=p0_masks_list,
            scores=all_game_rewards_p0,
            messages=all_game_messages if self.config.include_messages else None,
            group_overrides={},
            overrides=[{}] * len(all_game_messages) # one override dict per item in group
        )

        # Tokenize for Player 1's perspective
        p1_tokens_list: List[List[int]] = []
        p1_masks_list: List[List[int]] = []
        for game_msg_idx, game_dialogue in enumerate(all_game_messages):
            tokenization_result_p1 = tokenize_for_trainer(
                tokenizer=self.tokenizer, 
                chat=game_dialogue, 
                train_on_all_assistant_turns=True, 
                unmasked_role="player_1" # Updated parameter name
            )
            p1_tokens_list.append(tokenization_result_p1["tokens"])
            p1_masks_list.append(tokenization_result_p1["masks"])
        
        group_p1 = ScoredDataGroup(
            tokens=p1_tokens_list,
            masks=p1_masks_list,
            scores=all_game_rewards_p1,
            messages=all_game_messages if self.config.include_messages else None,
            group_overrides={},
            overrides=[{}] * len(all_game_messages) # one override dict per item in group
        )
            
        return [group_p0, group_p1], [] # Return a list of two ScoredDataGroup items

    async def get_next_item(self) -> Item:
        return {"seed": random.randint(0, 1_000_000_000)}

    async def setup(self):
        logger.info(f"Setting up {self.name} env. Tokenizer: {self.config.tokenizer_name}")

    async def evaluate(self, *args, **kwargs):
        logger.info(f"Starting eval for {self.name} with {self.config.eval_episodes} episodes.")
        p0_wins, p1_wins, draws = 0, 0, 0
        total_p0_reward, total_p1_reward = 0.0, 0.0
        num_completed_games = 0

        for i in range(self.config.eval_episodes):
            seed = random.randint(1_000_000_001, 2_000_000_000)
            eval_messages: List[Message] = []
            current_eval_rewards: Dict[int, float] = {0:0.0, 1:0.0}
            
            try:
                eval_ta_env = ta.make(env_id=self.config.env_id)
                eval_ta_env = LLMObservationWrapper(env=eval_ta_env)
                _ = eval_ta_env.reset(num_players=self.config.num_players, seed=seed)
            except Exception as e: logger.error(f"Eval ep {i+1} (seed {seed}) env setup err: {e}"); continue

            eval_done = False; eval_actions_taken = 0
            async with self.server.dedicated_server() as server:
                while not eval_done and eval_actions_taken < self.config.max_episode_actions:
                    current_sft_len = len(self.tokenizer.apply_chat_template(eval_messages, tokenize=False, add_generation_prompt=False))
                    if current_sft_len > self.config.max_token_length - 700:
                        current_eval_rewards = {0:0.0, 1:0.0}; eval_done = True; break
                    
                    current_player_id, obs_str = eval_ta_env.get_observation()
                    player_mark = 'O' if current_player_id == 0 else 'X'
                    sys_content = f"You are Player {current_player_id} ('{player_mark}'). {self.system_prompt_base}"
                    
                    turn_msgs_for_llm = eval_messages + [
                        {"role": "system", "content": sys_content},
                        {"role": "environment", "content": obs_str},
                        {"role": f"player_{current_player_id}", "content": ""}
                    ]
                    llm_resp = await self._sample_llm_response(turn_msgs_for_llm, server)
                    eval_actions_taken += 1

                    eval_messages.append({"role": "system", "content": sys_content})
                    eval_messages.append({"role": "environment", "content": obs_str})

                    if not llm_resp:
                        current_eval_rewards[current_player_id] = -1.0; current_eval_rewards[1-current_player_id] = 1.0
                        eval_messages.append({"role": f"player_{current_player_id}", "content": "<LLM_ERR>"})
                        eval_done = True; break
                    eval_messages.append({"role": f"player_{current_player_id}", "content": llm_resp})
                    parsed_action = self._parse_action_from_llm(llm_resp)
                    try:
                        rewards_dict, trunc, term, _ = eval_ta_env.step(action=parsed_action or "[INV_PARSE]")
                        current_eval_rewards = {pid: float(r) for pid, r in rewards_dict.items()}
                        eval_done = term or trunc
                    except Exception:
                        current_eval_rewards[current_player_id] = -1.0; current_eval_rewards[1-current_player_id] = 0.0
                        eval_done = True; break
                if not eval_done and eval_actions_taken >= self.config.max_episode_actions:
                    current_eval_rewards = {0:0.0, 1:0.0}
            
            eval_ta_env.close()
            num_completed_games +=1
            p0_r, p1_r = current_eval_rewards.get(0,0.0), current_eval_rewards.get(1,0.0)
            total_p0_reward += p0_r; total_p1_reward += p1_r
            if p0_r > 0 and p0_r > p1_r : p0_wins +=1
            elif p1_r > 0 and p1_r > p0_r : p1_wins +=1
            else: draws +=1
        
        if not num_completed_games: logger.warning("No eval episodes completed."); self.eval_metrics_custom = []; return
        self.eval_metrics_custom = [
            (f"{self.name}_eval/p0_win_rate", p0_wins/num_completed_games),
            (f"{self.name}_eval/p1_win_rate", p1_wins/num_completed_games),
            (f"{self.name}_eval/draw_rate", draws/num_completed_games),
            (f"{self.name}_eval/avg_p0_reward", total_p0_reward/num_completed_games),
            (f"{self.name}_eval/avg_p1_reward", total_p1_reward/num_completed_games),
            (f"{self.name}_eval/num_completed_episodes", num_completed_games),
        ]
        logger.info(f"Evaluation done. Metrics: {self.eval_metrics_custom}")

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        if wandb_metrics is None: wandb_metrics = {}
        
        if self.episode_outcomes_buffer:
            num_train_games = len(self.episode_outcomes_buffer)
            train_p0_wins, train_p1_wins, train_draws = 0,0,0
            total_train_p0_reward, total_train_p1_reward = 0.0,0.0

            for p0_r, p1_r in self.episode_outcomes_buffer:
                total_train_p0_reward += p0_r
                total_train_p1_reward += p1_r
                if p0_r > 0 and p0_r > p1_r : train_p0_wins +=1
                elif p1_r > 0 and p1_r > p0_r : train_p1_wins +=1
                else: train_draws +=1
            
            wandb_metrics[f"{self.name}_train/p0_win_rate"] = train_p0_wins/num_train_games if num_train_games else 0
            wandb_metrics[f"{self.name}_train/p1_win_rate"] = train_p1_wins/num_train_games if num_train_games else 0
            wandb_metrics[f"{self.name}_train/draw_rate"] = train_draws/num_train_games if num_train_games else 0
            wandb_metrics[f"{self.name}_train/avg_p0_episode_reward"] = total_train_p0_reward/num_train_games if num_train_games else 0
            wandb_metrics[f"{self.name}_train/avg_p1_episode_reward"] = total_train_p1_reward/num_train_games if num_train_games else 0
            wandb_metrics[f"{self.name}_train/num_games_in_batch"] = num_train_games
        self.episode_outcomes_buffer = []

        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []
        await super().wandb_log(wandb_metrics)

if __name__ == "__main__":
    UltimateTicTacToeEnv.cli()
