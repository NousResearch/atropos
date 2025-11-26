import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig
from environments.cline_env.cline_agent_env import ClineAgentEnv, ClineAgentEnvConfig
from environments.cline_env.profile_registry import supported_languages

# Load API keys and other settings from .env in the repo root.
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def collect_language(language: str, num_episodes: int, output_path: Path) -> None:
    env_config, _ = ClineAgentEnv.config_init()
    env_config.use_wandb = False
    env_config.group_size = 1
    env_config.use_cline_worker = True
    env_config.allowed_languages = [language]

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
    with output_path.open("w", encoding="utf-8") as f:
        for _ in range(num_episodes):
            item = await env.get_next_item()
            scored, _ = await env.collect_trajectory(item, skip_tokenization=True)
            if not scored:
                logger.error(
                    "Skipping trajectory for id=%s language=%s due to Cline worker failure",
                    item.get("instance_id"),
                    language,
                )
                continue
            row = env.dump_trajectory(item, scored)
            f.write(json.dumps(row))
            f.write("\n")
            f.flush()
            logger.info(
                "Dumped trajectory for id=%s language=%s -> %s",
                item.get("instance_id"),
                language,
                output_path,
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
        help="Number of dataset rows to process.",
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

        logger.info("Collecting %d episode(s) for language %s -> %s", args.num_episodes, language, out_path)
        asyncio.run(collect_language(language, args.num_episodes, out_path))


if __name__ == "__main__":
    main()
