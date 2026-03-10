"""
T1 Tool-Integrated Reasoning environment for Atropos.

Multi-step tool calling with full trajectory tracking:
- Loads the capitalone/T1 dataset from HuggingFace
- Walks through complete conversations, feeding model's actual responses back
- One ManagedServer session per trajectory → one extending node with aligned tokens
- GRPO over group_size independent trajectories per conversation
"""

import logging
from typing import Dict, List, Optional, Tuple

from t1_core import collect_multistep_trajectory  # noqa: E402
from t1_data import SINGLE_DOMAINS, load_t1_split  # noqa: E402
from t1_tools import T1_TOOLS  # noqa: E402

from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.envs.server_handling.server_baseline import APIServerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)


class T1ToolPlanningEnv(BaseEnv):
    """T1 Tool-Integrated Reasoning environment — multi-step trajectories.

    Each trajectory walks a full conversation:
      - Model generates responses at each user turn
      - Model's actual output is fed back (not GT) for the next turn
      - One ManagedServer session → one extending node per trajectory
      - GRPO compares group_size independent trajectories on the same conversation
    """

    name = "t1_tool_planning"

    def __init__(self, config, server_configs, slurm=True, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        # BaseEnv doesn't pass tool_parser to ServerManager — set it here
        # so ManagedServer creates a ToolCallTranslator for hermes-style tool calls
        self.server.tool_parser = "hermes"
        self.reward_buffer = []
        self.tc_f1_buffer = []
        self.tp_f1_buffer = []
        self.eval_metrics = []
        self.iter = 0

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        model_name = "Qwen/Qwen3-1.7B"
        env_config = BaseEnvConfig(
            tokenizer_name=model_name,
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8002",
            total_steps=200,
            batch_size=16,
            steps_per_eval=25,
            max_token_length=4096,
            start_tok_length=4096,
            wandb_name="t1-tool-planning",
            eval_limit_ratio=0.1,
            max_num_workers_per_node=8,
        )
        server_config = APIServerConfig(
            model_name=model_name,
            base_url="http://localhost:9001/v1",
            api_key="x",
            server_type="vllm",
        )
        # MUST return as a list — single APIServerConfig (not in list) causes
        # ServerManager to ignore base_url and auto-generate ports 9004-9007
        return env_config, [server_config]

    async def setup(self):
        logger.info("=== T1ToolPlanningEnv.setup() starting ===")

        # Load real T1 dataset from HuggingFace
        # Start with single-domain, 2 files per domain (~30 convos each = ~120 total)
        # Increase max_files_per_domain for more data
        self.train_conversations, self.eval_conversations = load_t1_split(
            domains=SINGLE_DOMAINS,
            eval_ratio=0.1,
        )

        logger.info(
            f"Setup complete: {len(self.train_conversations)} train conversations, "
            f"{len(self.eval_conversations)} eval conversations"
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if self.reward_buffer:
            wandb_metrics["train/avg_reward"] = sum(self.reward_buffer) / len(
                self.reward_buffer
            )
        if self.tc_f1_buffer:
            wandb_metrics["train/tool_call_f1"] = sum(self.tc_f1_buffer) / len(
                self.tc_f1_buffer
            )
        if self.tp_f1_buffer:
            wandb_metrics["train/tool_param_f1"] = sum(self.tp_f1_buffer) / len(
                self.tp_f1_buffer
            )
        self.reward_buffer = []
        self.tc_f1_buffer = []
        self.tp_f1_buffer = []
        for k, v in self.eval_metrics:
            wandb_metrics[k] = v
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def evaluate(self, *args, **kwargs):
        logger.info(
            f"=== evaluate() starting ({len(self.eval_conversations)} conversations) ==="
        )
        all_rewards = []
        all_tc_f1 = []
        all_tp_f1 = []

        for convo in self.eval_conversations:
            turn_results, nodes = await collect_multistep_trajectory(
                server=self.server,
                tokenizer=self.tokenizer,
                conversation=convo,
                tools=T1_TOOLS,
                max_tokens=512,
                temperature=0.0,
                tool_choice="auto",
            )
            for tr in turn_results:
                all_rewards.append(tr["scores"]["reward"])
                all_tc_f1.append(tr["scores"]["tool_call_f1"])
                all_tp_f1.append(tr["scores"]["tool_param_f1"])

        if all_rewards:
            self.eval_metrics.append(
                ("eval/avg_reward", sum(all_rewards) / len(all_rewards))
            )
            self.eval_metrics.append(
                ("eval/tool_call_f1", sum(all_tc_f1) / len(all_tc_f1))
            )
            self.eval_metrics.append(
                ("eval/tool_param_f1", sum(all_tp_f1) / len(all_tp_f1))
            )

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        convo = item
        user_turns = sum(1 for t in convo if t["Role"].strip().lower() == "user")
        logger.info(
            f"collect_trajectories: {len(convo)} turns ({user_turns} user), group_size={self.config.group_size}"
        )

        scored = ScoredDataGroup()
        scored["tokens"] = []
        scored["masks"] = []
        scored["scores"] = []
        scored["inference_logprobs"] = []

        # Run group_size independent trajectories on the same conversation
        for g in range(self.config.group_size):
            turn_results, nodes = await collect_multistep_trajectory(
                server=self.server,
                tokenizer=self.tokenizer,
                conversation=convo,
                tools=T1_TOOLS,
                max_tokens=512,
                temperature=1.0,
                tool_choice="auto",
            )

            if not nodes or not turn_results:
                logger.debug(f"  trajectory[{g}]: no nodes/results, skipping")
                continue

            # One node per trajectory (extending across all turns)
            node = nodes[0]
            unmasked = len([t for t in node.masked_tokens if t != -100])
            if unmasked < 5:
                logger.debug(
                    f"  trajectory[{g}]: only {unmasked} unmasked tokens, skipping"
                )
                continue

            # Trajectory reward = average across all turns
            avg_reward = sum(tr["scores"]["reward"] for tr in turn_results) / len(
                turn_results
            )
            avg_tc_f1 = sum(tr["scores"]["tool_call_f1"] for tr in turn_results) / len(
                turn_results
            )
            avg_tp_f1 = sum(tr["scores"]["tool_param_f1"] for tr in turn_results) / len(
                turn_results
            )

            scored["tokens"].append(node.tokens)
            scored["masks"].append(node.masked_tokens)
            scored["inference_logprobs"].append(node.logprobs)
            scored["scores"].append(avg_reward)

            self.reward_buffer.append(avg_reward)
            self.tc_f1_buffer.append(avg_tc_f1)
            self.tp_f1_buffer.append(avg_tp_f1)

            logger.info(
                f"  trajectory[{g}]: {len(turn_results)} turns, "
                f"{len(node.tokens)} tokens, reward={avg_reward:.3f}"
            )

        if not scored["tokens"]:
            logger.info("  -> None (no valid trajectories)")
            return None, []
        if all(s == scored["scores"][0] for s in scored["scores"]):
            logger.info(f"  -> None (all scores identical: {scored['scores'][0]:.3f})")
            return None, []

        logger.info(
            f"  -> valid group: {len(scored['tokens'])} trajectories, scores={[f'{s:.3f}' for s in scored['scores']]}"
        )
        return scored, []

    async def get_next_item(self):
        convo = self.train_conversations[self.iter % len(self.train_conversations)]
        self.iter += 1
        logger.debug(f"get_next_item: iter={self.iter}")
        return convo


if __name__ == "__main__":
    T1ToolPlanningEnv.cli()
