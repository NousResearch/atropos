import asyncio
import os
from pathlib import Path

from atroposlib.envs.base import APIServerConfig
from environments.browserbase.webvoyager_env import (
    WebVoyagerBrowserbaseEnv,
    WebVoyagerBrowserbaseEnvConfig,
)


def load_dotenv(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


async def main():
    load_dotenv()
    dataset_path = str(Path(__file__).with_name("webvoyager_smoke.jsonl"))

    required_env = [
        "BROWSERBASE_API_KEY",
        "BROWSERBASE_PROJECT_ID",
        "MODEL_API_KEY",
        "OPENAI_API_KEY",
    ]
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    config = WebVoyagerBrowserbaseEnvConfig(
        tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
        dataset_path=dataset_path,
        num_examples=1,
        include_messages=True,
        mode="dom",
        max_turns=6,
        tool_parser="hermes",
        proxy_model_to_stagehand=False,
        use_wandb=False,
    )

    env = WebVoyagerBrowserbaseEnv(
        config=config,
        server_configs=[
            APIServerConfig(
                model_name="NousResearch/Hermes-3-Llama-3.1-8B",
                base_url="http://localhost:9001/v1",
                api_key="x",
                server_type="sglang",
            )
        ],
        testing=False,
    )

    env._judge_completion = lambda prompt: asyncio.sleep(0, result="yes")

    try:
        await env.setup()
        item = await env.get_next_item()
        scored_item, backlog = await env.collect_trajectory(item)

        print("score:", scored_item["scores"])
        print("backlog:", backlog)
        print("overrides:", scored_item["overrides"])
        print("tokens:", len(scored_item["tokens"]))
        print("masks:", len(scored_item["masks"]))
        print("logprobs:", len(scored_item["inference_logprobs"]))
        print("messages:")
        for msg in scored_item["messages"] or []:
            print(msg)
    finally:
        await env.teardown()


if __name__ == "__main__":
    asyncio.run(main())
