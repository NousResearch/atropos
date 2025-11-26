#!/usr/bin/env python
"""
Launch rejection sampling for Cline trajectories.

This script collects multiple parallel rollouts per item (group_size) and saves
them to a JSONL file. Each row contains a ScoredDataGroup with:
- Multiple trajectory variants (tokens, masks, messages)
- Scores for each variant (currently dataset_target based)
- Cline metadata for each variant

For rejection sampling, you'd typically:
1. Run this to collect N trajectories per item
2. Use a reward model to score each trajectory
3. Select the best trajectory from each group
4. Use the selected trajectories for training

Usage:
    python launch_rejection_sampling.py --group-size 2 --num-items 5 --languages Python
    python launch_rejection_sampling.py --group-size 4 --num-items 10 --output data/rejection_samples.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from atroposlib.envs.base import APIServerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_rejection_sampling(
    languages: Optional[List[str]],
    group_size: int,
    num_items: int,
    output_path: Path,
    use_nomad: bool = True,
) -> None:
    """Collect rejection sampling data with parallel rollouts per item."""
    
    # Import here to ensure dotenv is loaded first
    from environments.cline_env.cline_agent_env import ClineAgentEnv, ClineAgentEnvConfig
    from environments.cline_env.profile_registry import supported_languages
    
    # Set CLINE_USE_NOMAD environment for worker manager
    if use_nomad:
        os.environ["CLINE_USE_NOMAD"] = "true"
    
    # Configure environment
    env_config = ClineAgentEnvConfig(
        tokenizer_name=os.getenv("TOKENIZER_NAME", "gpt2"),
        group_size=group_size,
        use_wandb=False,
        use_cline_worker=True,
        allowed_languages=languages or list(supported_languages()),
        include_messages=True,
    )
    
    # Configure API server (Anthropic via Cline worker)
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
    
    if not anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY must be set in environment or .env file")
    
    server_configs = [
        APIServerConfig(
            model_name=anthropic_model,
            base_url=anthropic_base_url,
            api_key=anthropic_api_key,
            num_requests_for_eval=0,
        )
    ]
    
    # Create environment
    env = ClineAgentEnv(env_config, server_configs, slurm=False, testing=True)
    await env.setup()
    
    logger.info(
        "Starting rejection sampling: group_size=%d, num_items=%d, languages=%s",
        group_size,
        num_items,
        languages or "all",
    )
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect trajectories
    completed = 0
    failed = 0
    
    with output_path.open("w", encoding="utf-8") as f:
        for item_idx in range(num_items):
            try:
                item = await env.get_next_item()
                logger.info(
                    "Processing item %d/%d: instance_id=%s, language=%s",
                    item_idx + 1,
                    num_items,
                    item.get("instance_id"),
                    item.get("language"),
                )
                
                # Collect group_size parallel trajectories
                scored_group, backlog = await env.collect_trajectories(item)
                
                if scored_group is None or not scored_group.get("tokens"):
                    logger.warning(
                        "No trajectories collected for item %s",
                        item.get("instance_id"),
                    )
                    failed += 1
                    continue
                
                # Build output row
                row = {
                    "instance_id": item.get("instance_id"),
                    "language": item.get("language"),
                    "repo_name": item.get("repo_name"),
                    "issue_text": item.get("issue_text"),
                    "group_size": len(scored_group.get("tokens", [])),
                    "scores": scored_group.get("scores", []),
                    "messages": scored_group.get("messages"),
                    "tokens": scored_group.get("tokens"),
                    "masks": scored_group.get("masks"),
                    "overrides": scored_group.get("overrides"),
                }
                
                f.write(json.dumps(row))
                f.write("\n")
                f.flush()
                
                completed += 1
                logger.info(
                    "Collected %d trajectories for item %s (scores: %s)",
                    row["group_size"],
                    item.get("instance_id"),
                    row["scores"],
                )
                
            except Exception as e:
                logger.error("Error processing item %d: %s", item_idx + 1, e)
                failed += 1
    
    logger.info(
        "Rejection sampling complete: %d items completed, %d failed -> %s",
        completed,
        failed,
        output_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect rejection sampling data with parallel Cline rollouts."
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=2,
        help="Number of parallel rollouts per item (default: 2)",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=5,
        help="Number of dataset items to process (default: 5)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Languages to filter dataset (default: all supported)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rejection_sampling.jsonl",
        help="Output JSONL file path (default: data/rejection_sampling.jsonl)",
    )
    parser.add_argument(
        "--no-nomad",
        action="store_true",
        help="Use local worker instead of Nomad (not recommended for parallel)",
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    
    asyncio.run(
        run_rejection_sampling(
            languages=args.languages,
            group_size=args.group_size,
            num_items=args.num_items,
            output_path=output_path,
            use_nomad=not args.no_nomad,
        )
    )


if __name__ == "__main__":
    main()
