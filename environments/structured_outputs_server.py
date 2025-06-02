import ast
import json
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import wandb
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from jsonschema import validate as json_validate, ValidationError

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)

def _extract_json(text: str) -> Optional[dict]:
    """
    Extract the *first* JSON object that appears immediately after the model’s
    internal <think> … </think> block.

    • Any non‑whitespace characters between </think> and the opening “{” cause
      the extraction to fail (we want the model to output pure JSON).

    • Uses json.JSONDecoder.raw_decode for robust brace matching instead of a
      greedy regex.
    """
    # Keep only the area after the *last* </think>
    after_think = text.split("</think>")[-1]

    # Allow leading whitespace/newlines but *nothing else*
    stripped_leading = after_think.lstrip()
    leading_before_json = after_think[: len(after_think) - len(stripped_leading)]
    if stripped_leading.startswith("{") is False:
        # Either we have non‑whitespace chatter or no JSON at all
        return None
    if any(c not in " \t\r\n" for c in leading_before_json):
        # Non‑whitespace junk before the JSON
        return None

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(stripped_leading)
        return obj
    except json.JSONDecodeError:
        return None

def _subset_match(candidate: dict, reference: dict) -> bool:
    for k, v in reference.items():
        if k not in candidate or candidate[k] != v:
            return False
    return True

def _ensure_schema_dict(schema_raw) -> Optional[dict]:
    if schema_raw is None:
        return None
    return json.loads(schema_raw)


class StructuredOutputsEnv(BaseEnv):
    name = "json_struct"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = []
        self.eval_metrics = []
        # Rollout visualisation
        self.rollouts_for_wandb =[]
        self.completion_lengths = []

    @classmethod
    def config_init(self) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_cfg = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=16 * 1024,
            inference_weight=1.0,
            wandb_name="json_struct",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        srv_cfgs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            )
        ]
        return env_cfg, srv_cfgs

    async def setup(self):
        ds = load_dataset("interstellarninja/json-mode-agentic", split="train").shuffle(
            seed=42
        )
        split = ds.train_test_split(0.02, seed=42)
        self.train, self.test = split["train"], split["test"]
        self.iter = 0

        self.percent_correct_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []

    def _score(self, cand_txt: str, gold_txt: str, schema: Optional[dict]) -> int:
        # Assumes schema is either dict or None
        cand = _extract_json(cand_txt)
        gold = _extract_json(gold_txt)
        if cand is None or gold is None:
            return 0
        if schema:
            try:
                json_validate(cand, schema)
            except ValidationError:
                return 0
        return 1 if _subset_match(cand, gold) else 0

    async def rollout_and_score_eval(self, item) -> int:
        conv = item["conversations"]
        sys = next((m for m in conv if m["from"] == "system"), None)
        usr = next((m for m in conv if m["from"] == "human"), None)
        gold = next((m for m in conv if m["from"] == "gpt"), None)
        if not usr or not gold:
            return 0
        schema = _ensure_schema_dict(item["schema"])
        msgs = [
            {
                "role": "system",
                "content": system_prompt + "\n\n" + (sys["value"] if sys else ""),
            },
            {"role": "user", "content": usr["value"]},
        ]
        prompt = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        comp = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=self.config.max_token_length - len(prompt),
            temperature=0.0,
            split="eval",
        )
        return self._score(comp.choices[0].text, gold["value"], schema)

    async def evaluate(self, *_, **__):
        scs = await tqdm_asyncio.gather(
            *[self.rollout_and_score_eval(t) for t in self.test]
        )
        self.eval_metrics.append(("eval/percent_correct", sum(scs) / len(scs)))

    async def get_next_item(self):
        row = self.train[self.iter % len(self.train)]
        self.iter += 1
        conv = row["conversations"]
        sys = next((m for m in conv if m["from"] == "system"), None)
        usr = next((m for m in conv if m["from"] == "human"), None)
        gold = next((m for m in conv if m["from"] == "gpt"), None)

        prompt = []
        if sys:
            prompt.append(
                frozenset(
                    {
                        "role": "system",
                        "content": system_prompt + "\n\n" + sys["value"],
                    }.items()
                )
            )
        prompt.append(frozenset({"role": "user", "content": usr["value"]}.items()))
        answer = gold["value"] if gold else ""
        schema = _ensure_schema_dict(row["schema"])
        return (tuple(prompt), answer, schema)

    async def collect_trajectories(
        self, itm
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        msgs = [dict(p) for p in itm[0]]
        schema = itm[2]
        prompt = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        comps = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length - len(prompt),
            temperature=0.8,
        )
        bunch = []
        for ch in comps.choices:
            bunch.append(
                (
                    msgs + [{"role": "assistant", "content": ch.text}],
                    itm[1],
                    schema,
                    ch.finish_reason,
                )
            )

        scored = await self.score(bunch)
        if scored is not None:
            await self.add_rollouts_for_wandb(scored, itm)
        return scored, []

    async def score(self, data) -> Optional[ScoredDataGroup]:
        sd = ScoredDataGroup(tokens=[], masks=[], scores=[])
        random.shuffle(data)
        for msgs, gold, schema, fin in data:
            reward = self._score(msgs[-1]["content"], gold, schema)
            out = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=msgs,
                finish_reason=fin,
                include_messages=True,
            )
            if len([m for m in out["masks"] if m != -100]) < 10:
                continue
            sd["tokens"].append(out["tokens"])
            sd["masks"].append(out["masks"])
            sd["scores"].append(1.0 if reward else -1.0)
            if len(sd["tokens"]) >= self.config.group_size:
                break
        # Apply length penalty if all responses are initially correct
        if sd["scores"] and all(score == 1.0 for score in sd["scores"]):
            token_lengths = [len(toks) for toks in sd["tokens"]]
            max_allowed_length = self.config.max_token_length
            length_threshold = max_allowed_length * 0.5 
            sd["scores"] = []
            for length in token_lengths:
                if length <= length_threshold:
                    sd["scores"].append(1.0)
                else:
                    pct = (length - length_threshold) / (max_allowed_length - length_threshold)
                    pct = min(pct, 1.0)
                    sd["scores"].append(1.0 - pct)
        self.percent_correct_buffer.extend(max(0, s) for s in sd["scores"])
        if len(sd["tokens"]) < self.config.group_size:
            return None
        if all(s == sd["scores"][0] for s in sd["scores"]):
            return None
        return sd


    async def create_rollout_table(self, wandb_metrics: Dict):
        if self.rollouts_for_wandb:
            table = wandb.Table(columns=["generation", "score", "expected_json"])
            for group in self.rollouts_for_wandb:
                for gen, score, expected in group:
                    table.add_data(gen, score, expected)
            wandb_metrics["train/rollouts"] = table

        self.rollouts_for_wandb = []
        return wandb_metrics

    async def add_rollouts_for_wandb(
        self,
        scored_data: ScoredDataGroup,
        item: Item,
    ):
        num_keep = getattr(self.config, "num_rollouts_per_group_for_logging", -1)
        if num_keep == -1:
            num_keep = self.config.group_size
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored_data["tokens"][i]),
                    scored_data["scores"][i],
                    item[1],  # expected JSON string
                )
                for i in range(num_keep)
            ]
        )
        if len(self.rollouts_for_wandb) > getattr(
            self.config, "num_rollouts_to_keep", 4
        ):
            self.rollouts_for_wandb.pop(0)

    async def wandb_log(self, metrics: Optional[Dict] = None):
        metrics = metrics or {}
        metrics = await self.create_rollout_table(metrics)
        if self.percent_correct_buffer:
            metrics["train/percent_correct"] = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
            self.percent_correct_buffer.clear()
        for k, v in self.eval_metrics:
            metrics[k] = v
        self.eval_metrics.clear()
        await super().wandb_log(metrics)


if __name__ == "__main__":
    StructuredOutputsEnv.cli()