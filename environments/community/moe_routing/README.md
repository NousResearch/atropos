# Graph of Tiered Experts Architecture

Train a language model to act as a routing policy for a Graph of Tiered Experts Architecture built from frozen Qwen3.5 model tiers.

## Motivation

Standard MoE trains experts and gates jointly during pre-training — requiring massive compute. This environment instead learns post-hoc routing across frozen same-family model tiers. The experts are pre-trained Qwen3.5 variants at different scales (0.8B to 35B parameters). Only the routing policy learns, from RL reward signals.

The model learns to be a router: given a query and expert descriptions, it selects which experts should handle the request. Reward is based on whether it picked the right experts for the query's intent, whether those experts have relevant capabilities, and whether it chose cost-efficient options.

## Architecture

```
Query + Expert Descriptions → LM (routing policy) → Expert Selection → Reward
                                    ↑                                    │
                                    └──────── REINFORCE update ──────────┘
```

**7 tiered experts from one model family:**

| ID | Role | Model | Size | Cost |
|----|------|-------|------|------|
| g0 | Triage | Qwen3.5-0.8B | 0.5 GB | 0.1 |
| g1 | Classifier | Qwen3.5-2B | 1.2 GB | 0.2 |
| a0 | Synthesizer | Qwen3.5-9B | 5.5 GB | 0.5 |
| a1 | Challenger | Qwen3.5-9B | 5.5 GB | 0.5 |
| v0 | Validator | Qwen3.5-27B | 15 GB | 0.8 |
| b0 | Executor | Qwen3.5-35B | 20 GB | 1.0 |
| q0 | Quorum | Qwen3.5-9B | 5.5 GB | 0.5 |

## Reward Function

Three-component weighted reward:

1. **Ideal Match** (weight: 0.5) — Jaccard similarity between selected experts and known-best experts for the query intent
2. **Capability Alignment** (weight: 0.3) — Whether selected experts' capabilities match the query intent
3. **Cost Efficiency** (weight: 0.1) — Preference for smaller/cheaper experts when quality is equal

Final score is normalized to [-1, 1].

## Dataset

120 items generated from 8 query templates x 15 topics. Query templates span intent types: triage, synthesis, challenge, validation, execution, simulation, classification, and research. Topics cover AI/ML domains including model scaling, distributed training, RLHF, and inference optimization.

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run with a local model server

```bash
# Start a vLLM or mlx_lm server on port 8378
mlx_lm.server --model Qwen/Qwen3-8B --port 8378

# Run the environment
python environments/community/moe_routing/moe_routing_env.py serve
```

### Run in process mode (local testing)

```bash
python environments/community/moe_routing/moe_routing_env.py process --num_trajectories 100
```

### Custom configuration

The environment accepts standard Atropos configuration plus:
- `top_k` (int, default=2): Number of experts to select per query
- `ideal_weight` (float, default=0.5): Weight for ideal expert match
- `capability_weight` (float, default=0.3): Weight for capability alignment
- `cost_weight` (float, default=0.1): Weight for cost efficiency

## Research Applications

- **Tiered expert graphs**: Study routing across same-family experts at different parameter scales
- **Post-hoc routing**: Train routing policies over frozen pre-trained models
- **Cost-aware inference**: Learn to balance quality vs. compute cost
- **Distributed routing**: The learned gate is tiny — could be shared via federated learning across edge nodes

## Generalization

The routing-over-frozen-experts pattern generalizes beyond LLMs. The same reward structure applies to any domain with multiple specialized classifiers and a feedback signal: RF signal detection, medical triage, content moderation, and network security.

## License

MIT
