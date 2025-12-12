import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig
from environments.cline_env.cline_agent_env import ClineAgentEnv, ClineAgentEnvConfig
from environments.cline_env.profile_registry import supported_languages

# Load API keys and other settings from .env in the repo root.
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def collect_language(
    language: str, 
    num_episodes: int, 
    output_path: Path, 
    task_timeout_s: float = 300.0,
    max_parallel: int = 1,
    group_size: int = 1,
) -> None:
    """Collect trajectories for a specific language with group sampling support.
    
    Args:
        language: Programming language to filter tasks
        num_episodes: Number of different tasks to process
        output_path: Path to output JSONL file
        task_timeout_s: Timeout for each Modal worker task
        max_parallel: Maximum number of Modal containers to run simultaneously
        group_size: Number of trajectories to collect per task (for rejection sampling)
    """
    env_config, _ = ClineAgentEnv.config_init()
    env_config.use_wandb = False
    env_config.group_size = 1  # We handle grouping ourselves for parallel execution
    env_config.allowed_languages = [language]
    
    # Always use Modal workers for scalable datagen
    env_config.use_modal_worker = True
    env_config.use_cline_worker = False
    env_config.modal_task_timeout_s = task_timeout_s
    
    total_trajectories = num_episodes * group_size
    logger.info(
        "Using Modal cloud workers (timeout=%ds, max_parallel=%d, group_size=%d)",
        task_timeout_s, max_parallel, group_size
    )
    logger.info(
        "Will collect %d tasks × %d trajectories/task = %d total trajectories",
        num_episodes, group_size, total_trajectories
    )

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")

    if not anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY must be set in .env for dump_trajectories")

    server_configs = [
        APIServerConfig(
            model_name=anthropic_model,
            base_url=anthropic_base_url,
            api_key=anthropic_api_key,
            num_requests_for_eval=0,
        )
    ]

    env = ClineAgentEnv(env_config, server_configs, slurm=False, testing=True)
    await env.setup()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all items first (to get unique instance_ids)
    items_to_process: List[Dict[str, Any]] = []
    for _ in range(num_episodes):
        item = await env.get_next_item()
        items_to_process.append(item)
    
    logger.info("Collected %d unique tasks from dataset", len(items_to_process))
    
    # Create work units: (item, trajectory_index) for each trajectory to collect
    # This allows multiple trajectories of the same task to run in parallel
    work_units: List[Tuple[Dict[str, Any], int]] = []
    for item in items_to_process:
        for traj_idx in range(group_size):
            work_units.append((item, traj_idx))
    
    logger.info("Created %d work units (%d tasks × %d trajectories)", len(work_units), num_episodes, group_size)
    
    def process_single_trajectory_sync(work_unit: Tuple[Dict[str, Any], int]) -> Optional[Dict[str, Any]]:
        """Process a single trajectory synchronously (Modal .remote() is blocking)."""
        item, traj_idx = work_unit
        instance_id = item.get("instance_id", "unknown")
        
        try:
            # collect_trajectory is async but Modal .remote() inside is blocking
            # We run this in a thread to enable true parallelism
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                scored, _ = loop.run_until_complete(env.collect_trajectory(item, skip_tokenization=True))
            finally:
                loop.close()
            
            if not scored:
                logger.error(
                    "Skipping trajectory %d for id=%s language=%s due to Cline worker failure",
                    traj_idx, instance_id, language,
                )
                return None
            
            row = env.dump_trajectory(item, scored)
            # Add trajectory index for grouping
            row["trajectory_index"] = traj_idx
            
            logger.info(
                "Collected trajectory %d/%d for id=%s score=%.3f",
                traj_idx + 1, group_size, instance_id, row.get("score", 0.0),
            )
            return row
        except Exception as e:
            logger.error(
                "Error processing trajectory %d for id=%s: %s",
                traj_idx, instance_id, e,
            )
            return None
    
    # Process all work units in parallel batches
    results: List[Dict[str, Any]] = []
    for batch_start in range(0, len(work_units), max_parallel):
        batch_end = min(batch_start + max_parallel, len(work_units))
        batch_work_units = work_units[batch_start:batch_end]
        
        logger.info(
            "Processing batch %d-%d of %d work units...",
            batch_start + 1, batch_end, len(work_units),
        )
        
        # Run batch in parallel using threads (Modal .remote() is blocking)
        batch_tasks = [asyncio.to_thread(process_single_trajectory_sync, wu) for wu in batch_work_units]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect successful results
        for result in batch_results:
            if isinstance(result, BaseException):
                logger.error("Task raised exception: %s", result)
            elif result is not None and isinstance(result, dict):
                results.append(result)
    
    # Write all results to file
    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row))
            f.write("\n")
    
    # Log summary statistics
    unique_tasks = len(set(r.get("id") for r in results))
    logger.info(
        "Wrote %d trajectories for %d unique tasks to %s (success rate: %.1f%%)",
        len(results), unique_tasks, output_path,
        100.0 * len(results) / total_trajectories if total_trajectories > 0 else 0,
    )
    
    # Log per-task statistics if group_size > 1
    if group_size > 1:
        from collections import defaultdict
        task_scores: Dict[str, List[float]] = defaultdict(list)
        for row in results:
            task_id = row.get("id", "unknown")
            score = row.get("score", 0.0)
            task_scores[task_id].append(score)
        
        logger.info("=== Per-task score summary ===")
        for task_id, scores in task_scores.items():
            logger.info(
                "Task %s: %d trajectories, scores=[%s], best=%.3f",
                task_id[:30], len(scores),
                ", ".join(f"{s:.3f}" for s in scores),
                max(scores) if scores else 0.0,
            )


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Collect Cline trajectories and write them to a JSONL file."
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of different tasks to process.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Number of trajectories to collect per task (for rejection sampling). "
             "Total trajectories = num_episodes × group_size.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cline_trajectories.jsonl",
        help="Output JSONL filename (saved under environments/cline_env/data by default).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Languages to process (default: run for all supported mappings).",
    )
    parser.add_argument(
        "--output-template",
        type=str,
        default="cline_{language}_trajectories.jsonl",
        help="Filename template when multiple languages are specified (supports {language}).",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=900.0,
        help="Timeout in seconds for Modal worker tasks (default: 900 = 15 min).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of Modal containers to run simultaneously (default: 1). "
             "Increase for faster data collection, but be mindful of Modal/API rate limits.",
    )
    args = parser.parse_args()

    if args.languages:
        languages = args.languages
    else:
        languages = list(supported_languages())

    base_output = Path(args.output)
    template_path = Path(args.output_template)

    for language in languages:
        slug = language.lower().replace(" ", "_")
        
        # Always use language-based naming unless user explicitly specified --output with a single language
        if len(languages) == 1 and args.output != "cline_trajectories.jsonl":
            # User explicitly specified --output, use it directly
            out_path = base_output
        else:
            # Use template-based naming for all languages (including single language with default output)
            template = template_path
            if "{language}" in template.name:
                name = template.name.format(language=slug)
                out_path = template.with_name(name)
            else:
                out_dir = template.parent if template.name else Path(".")
                stem = template.stem or "cline"
                suffix = template.suffix or ".jsonl"
                out_path = out_dir / f"{stem}_{slug}_trajectories{suffix}"

        if not out_path.is_absolute():
            out_path = base_dir / "data" / out_path

        logger.info("Collecting %d task(s) × %d trajectory(ies) for language %s -> %s", 
                    args.num_episodes, args.group_size, language, out_path)
        asyncio.run(collect_language(
            language, 
            args.num_episodes, 
            out_path,
            task_timeout_s=args.task_timeout,
            max_parallel=args.max_parallel,
            group_size=args.group_size,
        ))


if __name__ == "__main__":
    main()
