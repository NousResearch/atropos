import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataItem,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Import our game implementations
from env_2048 import Environment2048


class Atropos2048Env(BaseEnv):
    """
    Atropos environment for the 2048 game.
    The environment uses an LLM to play the 2048 game and rewards it based on
    game performance.
    """

    name = "atropos_2048"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.max_tile_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []
        self.print_this_env = False

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="Qwen/Qwen3-4B",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=4096,
            wandb_name="atropos_2048",
            max_batches_offpolicy=5,
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen3-4B",
                base_url="http://localhost:9001",
                api_key="x",
                num_requests_for_eval=64,
                server_type="trl",
            ),
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log game performance metrics
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass
        try:
            wandb_metrics["train/avg_max_tile"] = sum(
                self.max_tile_buffer
            ) / len(self.max_tile_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        self.max_tile_buffer = list()

        # Log evaluation metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
            
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.iter = 0

    async def evaluate(self, *args, **kwargs):
        """
        Run evaluation games with the current model.
        """
        win_count = 0
        max_tile_values = []
        total_scores = []
        
        # Run a fixed number of evaluation games
        num_eval_games = 32
        winning_value = 2048
        
        for _ in range(num_eval_games):
            env = Environment2048(winning_value=winning_value)
            observation, prompt = env.reset()
            
            done = False
            steps = 0
            max_steps = 1000
            
            system_message = env.system_message
            
            while not done and steps < max_steps:
                # Get action from LLM
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Current game state:\n{prompt}"}
                ]
                
                chat_completion = await self.server.chat_completion(
                    messages=messages,
                    n=1,
                    max_tokens=128,
                    temperature=0.6, # TODO: change between [0.2, 0.6, 1.0]
                    split="eval",
                )
                
                action = chat_completion.choices[0].message.content
                
                # Take step in environment
                observation, prompt, done, reward, info = env.step(action)
                steps += 1
                
                # Check if we've won
                if done and observation["max_tile"] >= winning_value:
                    win_count += 1
            
            max_tile_values.append(observation["max_tile"])
            total_scores.append(observation["score"])
        
        # Log metrics
        win_rate = win_count / num_eval_games
        avg_max_tile = sum(max_tile_values) / len(max_tile_values)
        avg_score = sum(total_scores) / len(total_scores)
        
        self.eval_metrics.append(("eval/win_rate", win_rate))
        self.eval_metrics.append(("eval/avg_max_tile", avg_max_tile))
        self.eval_metrics.append(("eval/avg_score", avg_score))

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Collect a trajectory by having the LLM play a full game of 2048.
        """
        # Create game environment with random seed from item
        winning_value = 2048
        
        # Set lower winning value during training for faster convergence
        if self.iter < 1000:
            winning_value = 128
        elif self.iter < 2000:
            winning_value = 256
        elif self.iter < 3000:
            winning_value = 512
        elif self.iter < 4000:
            winning_value = 1024
            
        env = Environment2048(winning_value=winning_value)
        observation, prompt = env.reset()
        
        # Grab a dedicated LLM server for the trajectory
        async with self.server.dedicated_server() as server:
            system_message = env.system_message
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Current game state:\n{prompt}"}
            ]
            
            done = False
            steps = 0
            max_steps = 200  # Limit trajectory length
            max_tile = 2  # Track highest tile achieved
            
            while not done and steps < max_steps:
                # Check if we'll exceed maximum tokens
                if len(self.tokenizer.apply_chat_template(messages)) > self.config.max_token_length - 200:
                    break
                    
                max_tokens = self.config.max_token_length - len(
                    self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                )
                
                # Get action from LLM
                chat_completion = await server.chat_completion(
                    messages=messages,
                    n=1,
                    max_tokens=max_tokens,
                )
                
                action = chat_completion.choices[0].message.content
                messages.append({"role": "assistant", "content": action})
                
                # Take step in environment
                observation, prompt, done, reward, info = env.step(action)
                
                # Track highest tile
                max_tile = max(max_tile, observation["max_tile"])
                
                if not done:
                    messages.append(
                        {"role": "user", "content": f"Current game state:\n{prompt}"}
                    )
                
                steps += 1
            
            # Calculate final score based on game performance
            score = 0
            if max_tile >= winning_value:
                # Successfully reached winning tile
                score = 1.0
            else:
                # Partial credit based on highest tile achieved
                score = (np.log2(max_tile) / np.log2(winning_value)) - 0.2
            
            # Add to metrics tracking
            self.percent_correct_buffer.append(1 if max_tile >= winning_value else 0)
            self.max_tile_buffer.append(max_tile)
            
            # Tokenize the conversation
            tokens = self.tokenizer.apply_chat_template(messages)
            masks = []
            
            for i, msg in enumerate(messages):
                if i == len(messages) - 1:
                    masks.extend(tokens[len(masks):])
                else:
                    curr_tokens = self.tokenizer.apply_chat_template(
                        messages[:i + 1],
                        add_generation_prompt=messages[i + 1]["role"] == "assistant",
                    )
                    if messages[i]["role"] == "user":
                        masks.extend([-100] * (len(curr_tokens) - len(masks)))
                    else:
                        masks.extend(curr_tokens[len(masks):])
            
            # Create scored data item
            scored_data_item = ScoredDataItem(
                messages=messages,
                finish_reason="stop" if done else "length",
                tokens=tokens,
                masks=masks,
                scores=score,
            )
            
            return scored_data_item, []

    async def get_next_item(self):
        """
        Get the next item for a trajectory.
        """
        next_item = {"seed": self.iter}
        self.iter += 1
        return next_item


if __name__ == "__main__":
    # Run the environment from command line
    Atropos2048Env.cli() 