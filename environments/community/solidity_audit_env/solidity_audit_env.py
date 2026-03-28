"""
Solidity Smart Contract Security Audit Environment for Atropos

Trains LLMs to detect security vulnerabilities in Solidity smart contracts.
Uses the darkknight25/Smart_Contract_Vulnerability_Dataset from HuggingFace
with multi-component reward scoring.
"""

import random
import time
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from dataset_loader import load_vulnerability_dataset
from scoring import compute_total_reward, extract_audit_response
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are a smart contract security auditor. Given a Solidity code snippet,
analyze it for security vulnerabilities.

You are allocated a maximum of 2048 tokens, please strive to use less.

Provide your audit result inside \\boxed{} as valid YAML with these exact fields:

\\boxed{
vulnerable: true/false
category: "vulnerability type (e.g. reentrancy, access_control, integer_overflow)"
description: "Brief explanation of the vulnerability found (or why the code is safe)"
fix: "Suggested fix or mitigation (or 'N/A' if not vulnerable)"
}

Important:
- The content inside \\boxed{} must be pure YAML that can be parsed by yaml.safe_load()
- Use double quotes around string values
- Always include all four fields: vulnerable, category, description, fix
- Be specific about the vulnerability location and mechanism

So please end your answer with \\boxed{your YAML audit result here}"""


class VulnerabilityEntry(TypedDict):
    code_snippet: str
    category: str
    description: str
    severity: str
    vulnerable: bool


class SolidityAuditEnv(BaseEnv):
    """
    Environment for training LLMs to detect security vulnerabilities
    in Solidity smart contracts.

    Uses the darkknight25/Smart_Contract_Vulnerability_Dataset and scores
    responses based on vulnerability detection accuracy, category matching,
    description quality, and format compliance.
    """

    name = "solidity_audit"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.reward_buffer = list()
        self.vuln_detection_buffer = list()
        self.category_accuracy_buffer = list()
        self.eval_metrics = list()

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="solidity_audit",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log custom metrics to WandB."""
        if wandb_metrics is None:
            wandb_metrics = {}

        try:
            wandb_metrics["train/avg_reward"] = sum(self.reward_buffer) / len(
                self.reward_buffer
            )
        except ZeroDivisionError:
            pass

        try:
            wandb_metrics["train/vuln_detection_accuracy"] = sum(
                self.vuln_detection_buffer
            ) / len(self.vuln_detection_buffer)
        except ZeroDivisionError:
            pass

        try:
            wandb_metrics["train/category_accuracy"] = sum(
                self.category_accuracy_buffer
            ) / len(self.category_accuracy_buffer)
        except ZeroDivisionError:
            pass

        self.reward_buffer = list()
        self.vuln_detection_buffer = list()
        self.category_accuracy_buffer = list()

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        """Load the vulnerability dataset and prepare train/test splits."""
        print("Loading Smart Contract Vulnerability Dataset...")
        self.train, self.test = load_vulnerability_dataset()
        print(f"Loaded {len(self.train)} training and {len(self.test)} test examples")
        random.shuffle(self.train)
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        """Save checkpoint with iteration state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    def _format_user_prompt(self, entry: Dict) -> str:
        """Format a vulnerability entry as a user prompt."""
        return (
            "Analyze the following Solidity code for security vulnerabilities:\n\n"
            f"```solidity\n{entry['code_snippet']}\n```"
        )

    async def rollout_and_score_eval(self, entry: Dict) -> Dict:
        """Rollout and score a single evaluation item."""
        user_content = self._format_user_prompt(entry)

        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.6,
            split="eval",
        )
        response_content = completion.choices[0].message.content

        parsed, _ = extract_audit_response(response_content)
        reward = compute_total_reward(
            predicted=parsed,
            actual_vulnerable=entry["vulnerable"],
            actual_category=entry["category"],
            actual_description=entry["description"],
            raw_response=response_content,
        )

        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response_content},
            ],
            "gold_vulnerable": entry["vulnerable"],
            "gold_category": entry["category"],
            "score": reward,
            "finish_reason": completion.choices[0].finish_reason,
        }

        return {"score": reward, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on test set."""
        start_time = time.time()

        eval_tasks = []
        eval_size = min(200, len(self.test))
        for entry in self.test[:eval_size]:
            eval_tasks.append(self.rollout_and_score_eval(entry))
        results = await tqdm_asyncio.gather(*eval_tasks)

        scores = [r["score"] for r in results]
        samples = [r["sample"] for r in results]

        avg_score = sum(scores) / len(scores) if scores else 0

        end_time = time.time()

        self.eval_metrics.append(("eval/avg_reward", avg_score))

        eval_metrics = {
            "eval/avg_reward": avg_score,
        }

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.6,
                "max_tokens": self.config.max_token_length,
            },
        )

    async def collect_trajectories(
        self, item: VulnerabilityEntry
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        """Generate audit responses for a given code snippet."""
        user_content = self._format_user_prompt(item)
        user_message = {"role": "user", "content": user_content}

        chat_completions = await self.server.chat_completion(
            messages=[{"role": "system", "content": system_prompt}, user_message],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=1.0,
        )

        to_score = list()
        to_backlog = list()

        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "ground_truth": item,
                    "finish_reason": chat_completion.finish_reason,
                }
            )

        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """Score audit responses using multi-component reward."""
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        ground_truth = rollout_group_data[0]["ground_truth"]

        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            out_dict = tokenize_for_trainer(
                self.tokenizer, item["messages"], finish_reason=item["finish_reason"]
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            response_content = item["messages"][-1]["content"]
            parsed, _ = extract_audit_response(response_content)

            reward = compute_total_reward(
                predicted=parsed,
                actual_vulnerable=ground_truth["vulnerable"],
                actual_category=ground_truth["category"],
                actual_description=ground_truth["description"],
                raw_response=response_content,
            )

            # Track detection accuracy for wandb
            if parsed is not None:
                from scoring import normalize_bool, score_category_match

                pred_vuln = normalize_bool(parsed.get("vulnerable"))
                if pred_vuln is not None:
                    self.vuln_detection_buffer.append(
                        1.0 if pred_vuln == ground_truth["vulnerable"] else 0.0
                    )
                pred_cat = str(parsed.get("category", ""))
                cat_score = score_category_match(pred_cat, ground_truth["category"])
                self.category_accuracy_buffer.append(cat_score)

            # Skip very short responses
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(reward)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        if not scores["tokens"]:
            return None

        for score_val in scores["scores"]:
            self.reward_buffer.append(score_val)

        # Length penalty when all scores are high (>= 0.9)
        if all(s >= 0.9 for s in scores["scores"]):
            token_lengths = [len(t) for t in scores["tokens"]]
            if max(token_lengths) == 0:
                return None

            max_allowed_length = self.config.max_token_length
            length_threshold = max_allowed_length * 0.5

            scores["scores"] = []
            for length in token_lengths:
                if length <= length_threshold:
                    scores["scores"].append(1.0)
                else:
                    pct_range = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    scores["scores"].append(1.0 - min(pct_range, 1.0))

        # No learning signal if all scores are identical
        if all(scores["scores"][0] == s for s in scores["scores"]):
            return None

        return scores

    async def get_next_item(self) -> VulnerabilityEntry:
        """Get the next training item."""
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    SolidityAuditEnv.cli()
