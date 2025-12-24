"""
Base class for evaluation environments.

based on PR #290  for eval-only environments.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import jsonlines
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)


def evaluate_log(
    metrics: Dict,
    eval_dir: Optional[str] = None,
    task_name: Optional[str] = None,
    model_name: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    generation_parameters: Optional[Dict] = None,
    samples: Optional[List[Dict]] = None,
    verbose: bool = True,
):
    if eval_dir is None:
        logger.warning("eval_dir is not set, skipping evaluation logging")
        return

    os.makedirs(eval_dir, exist_ok=True)
    filepath = os.path.join(eval_dir, "metrics.json")

    if start_time is None:
        start_time = time.time()
    if end_time is None:
        end_time = time.time()
    if generation_parameters is None:
        generation_parameters = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {task_name}")
        print(f"{'='*60}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"  Time: {end_time - start_time:.1f}s")
        print(f"{'='*60}\n")

    task_key = f"atropos|{task_name}|0"
    eval_result = {
        "config_general": {
            "model_name": model_name,
            "total_evaluation_time_seconds": str(end_time - start_time),
            "generation_parameters": generation_parameters,
        },
        "results": {
            task_key: metrics,
            "all": metrics,
        },
    }

    with open(filepath, "w") as f:
        json.dump(eval_result, f, indent=2)

    print(f"Evaluation results saved to {filepath}")

    if samples:
        samples_filepath = os.path.join(eval_dir, "samples.jsonl")
        with jsonlines.open(samples_filepath, "w") as writer:
            for sample in samples:
                writer.write(sample)
        print(f"Evaluation samples saved to {samples_filepath}")


class EvalBase(ABC):
    """
    Base class for evaluation environments.

    Subclasses must implement:
    - setup_data(): Returns list of data items to evaluate
    - run_item(client, data_item): Process one item, returns (metrics_dict, sample)
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.data = self.setup_data()

    def get_generation_params(self) -> dict:
        return {
            "temperature": getattr(self, "temperature", 0.0),
            "max_tokens": getattr(self, "max_tokens", 4096),
            "n": getattr(self, "n", 1),
        }

    async def chat_completion(
        self, client: AsyncOpenAI, messages: List[dict]
    ) -> ChatCompletion:
        gen_params = self.get_generation_params()
        return await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **gen_params,
        )

    @abstractmethod
    def setup_data(self) -> list:
        raise NotImplementedError

    @abstractmethod
    async def run_item(
        self, client: AsyncOpenAI, data_item: dict
    ) -> Tuple[dict, dict]:
        """
        Process a single data item.

        Returns:
            Tuple[dict, dict]: (metrics_dict, sample_dict)
                - metrics_dict: keys like "accuracy" with numeric values
                - sample_dict: the sample data for logging
        """
        raise NotImplementedError

    async def __call__(self, client: AsyncOpenAI):
        start_time = time.time()

        task_coros = [self.run_item(client, item) for item in self.data]
        task_results = await tqdm_asyncio.gather(
            *task_coros, desc=f"Evaluating {self.__class__.__name__}"
        )

        end_time = time.time()

        metrics_list = [result[0] for result in task_results]
        samples = [result[1] for result in task_results]

        keys = list(metrics_list[0].keys())
        metrics = {
            key: sum(result[key] for result in metrics_list) / len(metrics_list)
            for key in keys
        }

        task_name = self.__class__.__name__

        evaluate_log(
            metrics,
            eval_dir=getattr(self, "eval_dir", None),
            task_name=task_name,
            model_name=self.model_name,
            start_time=start_time,
            end_time=end_time,
            generation_parameters=self.get_generation_params(),
            samples=samples,
            verbose=True,
        )

        return metrics


async def eval_runner(eval_cls, **eval_kwargs):
    """
    CLI runner for evaluation environments.

    Usage in __main__:
        if __name__ == "__main__":
            import asyncio
            from eval_base import eval_runner
            asyncio.run(eval_runner(MyEval, temperature=0.0, max_tokens=4096))
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for OpenAI-compatible API",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="x",
        help="API key (use 'x' for local servers)",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )

    args, _ = parser.parse_known_args()

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    eval_kwargs["model_name"] = args.model_name
    eval_kwargs["eval_dir"] = args.eval_dir

    eval_env = eval_cls(**eval_kwargs)
    return await eval_env(client)
