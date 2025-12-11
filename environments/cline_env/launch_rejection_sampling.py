#!/usr/bin/env python
"""
Launch rejection sampling for Cline trajectories using Modal.

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
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_modal_task(
    item: Dict[str, Any],
    language: str,
    task_timeout_s: float = 300.0,
) -> Optional[Dict[str, Any]]:
    """Run a single task on Modal and return the result."""
    try:
        from environments.cline_env.modal_worker import get_function_for_language
        
        func = get_function_for_language(language)
        
        # Run on Modal
        result = func.remote(
            issue_text=item.get("issue_text", ""),
            task_timeout_s=task_timeout_s,
        )
        
        return result
    except Exception as e:
        logger.error("Modal task failed: %s", e)
        return None


async def process_single_item(
    item: Dict[str, Any],
    item_idx: int,
    num_items: int,
    group_size: int,
    task_timeout_s: float = 300.0,
) -> Optional[Dict[str, Any]]:
    """Process a single item and return the result row."""
    try:
        instance_id = item.get("instance_id", f"item_{item_idx}")
        language = item.get("language", "Python")
        
        logger.info(
            "Processing item %d/%d: instance_id=%s, language=%s",
            item_idx + 1,
            num_items,
            instance_id,
            language,
        )
        
        # Run group_size parallel tasks on Modal
        tasks = [
            run_modal_task(item, language, task_timeout_s)
            for _ in range(group_size)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        scores = []
        messages = []
        conversation_histories = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Task %d failed with exception: %s", i, result)
                scores.append(0.0)
                messages.append([])
                conversation_histories.append([])
            elif result is None:
                logger.warning("Task %d returned None", i)
                scores.append(0.0)
                messages.append([])
                conversation_histories.append([])
            else:
                # Score based on success and files created
                success = result.get("success", False)
                files_created = result.get("files_created", [])
                score = 1.0 if success and files_created else 0.0
                
                scores.append(score)
                messages.append(result.get("conversation_history", []))
                conversation_histories.append(result.get("conversation_history", []))
        
        # Build output row
        row = {
            "instance_id": instance_id,
            "language": language,
            "repo_name": item.get("repo_name"),
            "issue_text": item.get("issue_text"),
            "group_size": group_size,
            "scores": scores,
            "messages": messages,
            "conversation_histories": conversation_histories,
        }
        
        logger.info(
            "Collected %d trajectories for item %s (scores: %s)",
            group_size,
            instance_id,
            scores,
        )
        return row
        
    except Exception as e:
        logger.error("Error processing item %d (%s): %s", item_idx + 1, item.get("instance_id"), e)
        return None


async def run_rejection_sampling(
    languages: Optional[List[str]],
    group_size: int,
    num_items: int,
    max_concurrent_items: int,
    output_path: Path,
    task_timeout_s: float = 300.0,
) -> None:
    """Collect rejection sampling data with parallel rollouts on Modal.
    
    Parallelism:
    - max_concurrent_items: How many items to process in parallel
    - group_size: How many rollouts per item (parallel Modal workers per item)
    - Total concurrent workers = max_concurrent_items × group_size
    
    Example: --max-concurrent-items 2 --group-size 2 = 4 concurrent Modal workers
    """
    
    from environments.cline_env.profile_registry import supported_languages
    from datasets import load_dataset
    
    # Load dataset
    dataset = load_dataset("NousResearch/cline_synthetic_1k", split="train")
    
    # Filter by languages if specified
    allowed_langs = set(languages) if languages else set(supported_languages())
    items = []
    for row in dataset:
        if row.get("language") in allowed_langs:
            items.append(row)
            if len(items) >= num_items:
                break
    
    if len(items) < num_items:
        logger.warning("Only found %d items (requested %d)", len(items), num_items)
    
    total_concurrent = max_concurrent_items * group_size
    logger.info(
        "Starting rejection sampling: group_size=%d, max_concurrent_items=%d, "
        "total_concurrent_workers=%d, num_items=%d, languages=%s",
        group_size,
        max_concurrent_items,
        total_concurrent,
        len(items),
        languages or "all",
    )
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process items in parallel batches
    completed = 0
    failed = 0
    
    # Use semaphore to limit concurrent items
    semaphore = asyncio.Semaphore(max_concurrent_items)
    
    async def process_with_semaphore(item, idx):
        async with semaphore:
            return await process_single_item(
                item, idx, len(items), group_size, task_timeout_s
            )
    
    # Launch all tasks
    tasks = [
        process_with_semaphore(item, idx)
        for idx, item in enumerate(items)
    ]
    
    # Gather results as they complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Write results to file
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            if isinstance(result, Exception):
                logger.error("Task failed with exception: %s", result)
                failed += 1
            elif result is None:
                failed += 1
            else:
                f.write(json.dumps(result))
                f.write("\n")
                completed += 1
    
    logger.info(
        "Rejection sampling complete: %d items completed, %d failed -> %s",
        completed,
        failed,
        output_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect rejection sampling data with parallel Cline rollouts on Modal."
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=2,
        help="Number of parallel rollouts per item (default: 2)",
    )
    parser.add_argument(
        "--max-concurrent-items",
        type=int,
        default=1,
        help="Number of items to process in parallel (default: 1). "
             "Total workers = max_concurrent_items × group_size",
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
        "--task-timeout",
        type=float,
        default=300.0,
        help="Timeout per task in seconds (default: 300)",
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
            max_concurrent_items=args.max_concurrent_items,
            output_path=output_path,
            task_timeout_s=args.task_timeout,
        )
    )


if __name__ == "__main__":
    main()
