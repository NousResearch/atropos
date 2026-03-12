"""
MoE Routing Environment — Atropos RL for Heterogeneous Expert Selection.

Trains a language model to act as a gating network for a heterogeneous
Mixture-of-Experts inference mesh. The model learns which frozen expert
handles which query type — purely from reward signals.

Architecture:
  - 7 experts at different scales (0.8B → 35B parameters)
  - Experts are frozen pre-trained models (no fine-tuning)
  - Only the routing policy learns, via RL reward
  - Gate sees: query + expert descriptions
  - Gate outputs: JSON array of expert IDs (top-k selection)
  - Reward: ideal_match + capability_alignment + cost_efficiency

This makes MoE practical on consumer hardware where you can't afford
to train experts jointly. The gate is tiny — its training could be
distributed via DisTrO across edge nodes.

Author: Thomas Perry
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Dict, List, Optional, Tuple

from pydantic import Field

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig
from atroposlib.type_definitions import Item

# ─── Expert Definitions ──────────────────────────────────────

EXPERTS = [
    {
        "id": "g0",
        "name": "triage",
        "model": "Qwen3.5-0.8B",
        "size_gb": 0.5,
        "cost": 0.1,
        "capabilities": ["triage", "routing", "intent"],
    },
    {
        "id": "g1",
        "name": "classifier",
        "model": "Qwen3.5-2B",
        "size_gb": 1.2,
        "cost": 0.2,
        "capabilities": ["classify", "annotate", "mediate"],
    },
    {
        "id": "a0",
        "name": "synthesizer",
        "model": "Qwen3.5-9B",
        "size_gb": 5.5,
        "cost": 0.5,
        "capabilities": ["synthesize", "reason", "assemble"],
    },
    {
        "id": "a1",
        "name": "challenger",
        "model": "Qwen3.5-9B",
        "size_gb": 5.5,
        "cost": 0.5,
        "capabilities": ["refute", "challenge", "adversarial"],
    },
    {
        "id": "v0",
        "name": "validator",
        "model": "Qwen3.5-27B",
        "size_gb": 15.0,
        "cost": 0.8,
        "capabilities": ["validate", "critique", "verify"],
    },
    {
        "id": "b0",
        "name": "executor",
        "model": "Qwen3.5-35B",
        "size_gb": 20.0,
        "cost": 1.0,
        "capabilities": ["execute", "distill", "plan"],
    },
    {
        "id": "q0",
        "name": "quorum",
        "model": "Qwen3.5-9B",
        "size_gb": 5.5,
        "cost": 0.5,
        "capabilities": ["simulate", "quorum", "generate"],
    },
]

EXPERT_IDS = {e["id"] for e in EXPERTS}

EXPERT_DESC = "\n".join(
    f"- {e['id']} ({e['name']}): {e['model']}, {e['size_gb']}GB, "
    f"capabilities={e['capabilities']}, cost={e['cost']}"
    for e in EXPERTS
)


# ─── Query Templates ─────────────────────────────────────────

QUERY_TEMPLATES = [
    {
        "intent": "triage",
        "query": "What kind of question is this: '{topic}'?",
        "ideal": ["g0", "g1"],
    },
    {
        "intent": "synthesis",
        "query": "Synthesize a comprehensive analysis of {topic}.",
        "ideal": ["a0", "v0"],
    },
    {
        "intent": "challenge",
        "query": "What are the strongest counterarguments to {topic}?",
        "ideal": ["a1", "v0"],
    },
    {
        "intent": "validation",
        "query": "Verify whether this claim is accurate: {topic}.",
        "ideal": ["v0", "g1"],
    },
    {
        "intent": "execution",
        "query": "Create a step-by-step plan to implement {topic}.",
        "ideal": ["b0", "a0"],
    },
    {
        "intent": "simulation",
        "query": "Simulate three possible outcomes of {topic}.",
        "ideal": ["q0", "a0"],
    },
    {
        "intent": "classify",
        "query": "Classify {topic} into the most relevant categories.",
        "ideal": ["g1", "g0"],
    },
    {
        "intent": "research",
        "query": "Conduct a deep research review on {topic}.",
        "ideal": ["a0", "v0", "b0"],
    },
]

TOPICS = [
    "transformer attention mechanisms",
    "mixture of experts scaling laws",
    "RLHF vs constitutional AI",
    "distributed training with DisTrO",
    "self-hosted inference on consumer hardware",
    "agent skill acquisition via reinforcement learning",
    "heterogeneous model routing",
    "perimeter-based safety constraints",
    "post-hoc routing in MoE architectures",
    "federated learning of routing policies",
    "open source vs closed AI development",
    "MLX optimization on Apple Silicon",
    "token-level reward attribution",
    "curriculum learning for language models",
    "emergent capabilities in small models",
]


# ─── Config ──────────────────────────────────────────────────


class MoERoutingConfig(BaseEnvConfig):
    """Configuration for the MoE Routing environment."""

    top_k: int = Field(default=2, description="Number of experts to select per query")
    cost_weight: float = Field(
        default=0.1, description="Weight for cost efficiency in reward"
    )
    capability_weight: float = Field(
        default=0.3, description="Weight for capability match in reward"
    )
    ideal_weight: float = Field(
        default=0.5, description="Weight for ideal expert match in reward"
    )


# ─── Environment ─────────────────────────────────────────────


class MoERoutingEnv(BaseEnv):
    """
    Trains a language model to route queries to the right experts in a
    heterogeneous MoE mesh.

    The model acts as a gating network:
      Input:  query text + expert descriptions
      Output: JSON array of expert IDs (top-k)
      Reward: weighted combination of ideal match, capability alignment,
              and cost efficiency

    This environment demonstrates that routing decisions over frozen experts
    can be learned via RL — making MoE practical on consumer hardware.
    """

    name = "moe_routing"
    env_config_cls = MoERoutingConfig

    def __init__(self, config: MoERoutingConfig, server_configs, **kwargs):
        super().__init__(config, server_configs, **kwargs)
        self._items: List[Dict] = []
        self._item_idx = 0
        self._episode_count = 0
        self._reward_sum = 0.0

    async def setup(self):
        """Generate query items from templates x topics."""
        for template in QUERY_TEMPLATES:
            for topic in TOPICS:
                self._items.append(
                    {
                        "intent": template["intent"],
                        "query": template["query"].format(topic=topic),
                        "ideal": template["ideal"],
                        "topic": topic,
                    }
                )
        random.shuffle(self._items)
        await self.setup_wandb()

    async def get_next_item(self) -> Item:
        """Cycle through items."""
        if not self._items:
            await asyncio.sleep(1)
            return None
        item = self._items[self._item_idx % len(self._items)]
        self._item_idx += 1
        return item

    def _build_prompt(self, item: Dict) -> List[Dict[str, str]]:
        """Build the routing prompt for the model."""
        system = (
            "You are a routing controller for a Mixture-of-Experts inference mesh. "
            "Select the best experts to handle a given query based on intent, "
            "capabilities, and cost efficiency.\n\n"
            f"Available experts:\n{EXPERT_DESC}\n\n"
            f"Select exactly {self.config.top_k} experts. "
            'Respond with ONLY a JSON array of expert IDs, e.g. ["a0", "v0"]. '
            "No explanation, no markdown, just the JSON array."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Route this query:\n\n{item['query']}"},
        ]

    def _parse_selection(self, text: str) -> List[str]:
        """Parse model output into expert IDs."""
        text = text.strip()
        # Try direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, str) and x in EXPERT_IDS]
        except json.JSONDecodeError:
            pass
        # Fallback: extract IDs from text
        return [
            w.strip("\"'[].,") for w in text.split() if w.strip("\"'[].,") in EXPERT_IDS
        ]

    def _score(self, selected: List[str], item: Dict) -> Tuple[float, Dict]:
        """
        Score expert selection with three components:

        1. Ideal match (Jaccard similarity with known-best experts)
        2. Capability match (do selected experts have relevant capabilities?)
        3. Cost efficiency (prefer cheaper experts when quality is equal)

        Returns (score, breakdown_dict).
        """
        if not selected:
            return -1.0, {"reason": "no_valid_selection"}

        ideal = set(item["ideal"])
        chosen = set(selected[: self.config.top_k])

        # 1. Ideal match — Jaccard similarity
        overlap = len(ideal & chosen)
        union = len(ideal | chosen)
        ideal_score = overlap / union if union > 0 else 0.0

        # 2. Capability match — does the expert's capability list contain the intent?
        intent = item["intent"].lower()
        cap_score = 0.0
        for eid in chosen:
            expert = next((e for e in EXPERTS if e["id"] == eid), None)
            if expert:
                for cap in expert["capabilities"]:
                    if cap in intent or intent in cap:
                        cap_score += 0.5
        cap_score = min(cap_score, 1.0)

        # 3. Cost efficiency — normalized inverse cost
        total_cost = sum(
            next((e["cost"] for e in EXPERTS if e["id"] == eid), 1.0) for eid in chosen
        )
        max_cost = sum(
            sorted([e["cost"] for e in EXPERTS], reverse=True)[: self.config.top_k]
        )
        cost_score = 1.0 - (total_cost / max_cost) if max_cost > 0 else 0.5

        # Weighted combination
        raw = (
            self.config.ideal_weight * ideal_score
            + self.config.capability_weight * cap_score
            + self.config.cost_weight * cost_score
        )
        # Normalize to [-1, 1]
        score = max(-1.0, min(1.0, raw * 2 - 0.5))

        breakdown = {
            "ideal_score": round(ideal_score, 3),
            "cap_score": round(cap_score, 3),
            "cost_score": round(cost_score, 3),
            "total_cost": round(total_cost, 3),
            "selected": list(chosen),
            "ideal": list(ideal),
            "intent": item["intent"],
            "final_score": round(score, 3),
        }
        return score, breakdown

    async def collect_trajectory(self, item: Item) -> Tuple[Optional[Dict], List[Item]]:
        """
        Collect a single routing trajectory:
        1. Build prompt with query + expert descriptions
        2. Model generates expert selection (JSON array)
        3. Score the selection
        4. Return scored tokens for training
        """
        messages = self._build_prompt(item)

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.completion(
                prompt=messages,
                n=1,
                max_tokens=64,
                temperature=0.3,
            )
            state = managed.get_state()
            node = state["nodes"][0]

        # Parse and score
        generated = (
            completion.choices[0].text
            if hasattr(completion.choices[0], "text")
            else str(completion.choices[0])
        )
        selected = self._parse_selection(generated)
        score, breakdown = self._score(selected, item)

        # Track metrics
        self._episode_count += 1
        self._reward_sum += score

        return {
            "tokens": node.tokens,
            "masks": node.masked_tokens,
            "scores": score,
        }, []

    async def evaluate(self, *args, **kwargs):
        """Log evaluation metrics."""
        if self._episode_count > 0:
            avg = self._reward_sum / self._episode_count
            print(
                f"[MoE Routing] Episodes: {self._episode_count}, Avg Reward: {avg:.4f}"
            )

    @classmethod
    def config_init(cls):
        """Default configuration for CLI usage."""
        return (
            MoERoutingConfig(
                tokenizer_name="Qwen/Qwen3-8B",
                group_size=4,
                max_num_workers=2,
                steps_per_eval=50,
                max_token_length=512,
                total_steps=1000,
                top_k=2,
                use_wandb=True,
            ),
            [
                APIServerConfig(
                    model_name="Qwen/Qwen3-8B",
                    base_url="http://127.0.0.1:8378/v1",
                    api_key="local",  # pragma: allowlist secret
                )
            ],
        )


if __name__ == "__main__":
    MoERoutingEnv.cli()
