#!/usr/bin/env python
"""
Collect Cline trajectories using Modal for execution.

This script runs Cline tasks on Modal's cloud infrastructure instead of
local Nomad workers. Each task runs in an isolated container.

Usage:
    # First set up Modal secrets:
    modal secret create dockerhub-creds REGISTRY_USERNAME=xxx REGISTRY_PASSWORD=xxx
    modal secret create anthropic-key ANTHROPIC_API_KEY=sk-ant-xxx
    
    # Then run:
    python dump_trajectories_modal.py --languages Python --num-episodes 1
"""

import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Dataset configuration
DATASET_NAME = "NousResearch/swe-agent-13k-2025-06-15"
SYSTEM_PROMPT = (
    "You are a senior software engineer helping to resolve a GitHub issue. "
    "Read the issue description carefully and propose a clear, concrete patch "
    "or explanation of how to resolve it."
)


def get_issue_text_from_row(row: Dict[str, Any]) -> str:
    """Extract the issue text from a dataset row."""
    conversations = row.get("conversations", [])
    
    if isinstance(conversations, list) and len(conversations) > 1:
        second = conversations[1]
        if isinstance(second, dict) and second.get("from") in ("human", "user"):
            return second.get("value") or ""
    
    if isinstance(conversations, list) and conversations:
        first = conversations[0]
        if isinstance(first, dict):
            return first.get("value") or ""
    
    return ""


def filter_dataset_by_languages(dataset, languages: List[str]) -> List[int]:
    """Get dataset indices that match the specified languages."""
    allowed = set(languages)
    indices = []
    for idx in range(len(dataset)):
        row = dataset[idx]
        lang = row.get("language")
        if lang in allowed:
            indices.append(idx)
    return indices


async def run_task_on_modal(
    issue_text: str,
    language: str = "Python",
    repo_url: Optional[str] = None,
    repo_branch: Optional[str] = None,
    idle_timeout_s: float = 30.0,
    task_timeout_s: float = 300.0,
) -> Dict[str, Any]:
    """Run a single task on Modal."""
    import modal
    
    # Import the Modal function
    # Note: This requires the Modal app to be deployed first
    run_python_task = modal.Function.lookup("cline-workers", "run_python_task")
    
    # Call the Modal function
    result = run_python_task.remote(
        issue_text=issue_text,
        repo_url=repo_url,
        repo_branch=repo_branch,
        idle_timeout_s=idle_timeout_s,
        task_timeout_s=task_timeout_s,
    )
    
    return result


async def collect_trajectories(
    languages: List[str],
    num_episodes: int,
    output_path: Path,
    idle_timeout_s: float = 30.0,
    task_timeout_s: float = 300.0,
) -> None:
    """Collect trajectories using Modal."""
    # Load dataset
    logger.info("Loading dataset: %s", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME, split="train")
    
    # Filter by languages
    indices = filter_dataset_by_languages(dataset, languages)
    if not indices:
        raise RuntimeError(f"No dataset rows matched languages: {languages}")
    
    logger.info("Found %d rows matching languages %s", len(indices), languages)
    random.shuffle(indices)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    completed = 0
    failed = 0
    
    with output_path.open("w", encoding="utf-8") as f:
        for i in range(min(num_episodes, len(indices))):
            idx = indices[i]
            row = dataset[idx]
            
            instance_id = row.get("id", f"idx-{idx}")
            language = row.get("language", "unknown")
            issue_text = get_issue_text_from_row(row)
            
            if not issue_text:
                logger.warning("Skipping row %s: no issue text", instance_id)
                failed += 1
                continue
            
            logger.info(
                "Processing %d/%d: instance_id=%s, language=%s",
                i + 1,
                num_episodes,
                instance_id,
                language,
            )
            
            try:
                result = await run_task_on_modal(
                    issue_text=issue_text,
                    language=language,
                    repo_url=row.get("repo_url"),
                    repo_branch=row.get("branch"),
                    idle_timeout_s=idle_timeout_s,
                    task_timeout_s=task_timeout_s,
                )
                
                if not result.get("success"):
                    logger.error(
                        "Task failed for %s: %s",
                        instance_id,
                        result.get("error"),
                    )
                    failed += 1
                    continue
                
                # Build output row
                out_row = {
                    "id": instance_id,
                    "language": language,
                    "repo_name": row.get("repo_name"),
                    "repo_url": row.get("repo_url"),
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human", "value": issue_text},
                        {"from": "assistant", "value": result.get("assistant_content", "")},
                    ],
                    "cline_metadata": {
                        "task_id": result.get("task_id"),
                        "cline_messages": result.get("cline_messages", []),
                        "files_created": result.get("files_created", []),
                        "execution_time_s": result.get("execution_time_s"),
                    },
                }
                
                f.write(json.dumps(out_row))
                f.write("\n")
                f.flush()
                
                logger.info(
                    "Completed %s: %d messages, %.1fs execution time",
                    instance_id,
                    len(result.get("cline_messages", [])),
                    result.get("execution_time_s", 0),
                )
                completed += 1
                
            except Exception as e:
                logger.error("Error processing %s: %s", instance_id, e)
                failed += 1
    
    logger.info(
        "Collection complete: %d completed, %d failed -> %s",
        completed,
        failed,
        output_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect Cline trajectories using Modal."
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to collect (default: 1)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["Python"],
        help="Languages to process (default: Python)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/modal_trajectories.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=30.0,
        help="Idle timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=300.0,
        help="Task timeout in seconds (default: 300)",
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    
    asyncio.run(
        collect_trajectories(
            languages=args.languages,
            num_episodes=args.num_episodes,
            output_path=output_path,
            idle_timeout_s=args.idle_timeout,
            task_timeout_s=args.task_timeout,
        )
    )


if __name__ == "__main__":
    main()
