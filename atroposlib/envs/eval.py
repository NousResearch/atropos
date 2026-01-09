import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import jsonlines
import numpy as np
from openai.types.chat import ChatCompletion
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.server_handling.server_manager import ServerManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pass_at_k(m, c, k):
    """
    m: total samples
    c: correct samples
    k: k in pass@k
    """
    if m - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(m - c + 1, m + 1))


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
    """
    Log evaluation results to a JSON file in the format expected by nous-evals.

    Args:
        metrics: Dictionary of metrics to log (same format as wandb_log)
        eval_dir: Directory to save evaluation results to
        task_name: Name of the evaluation task (defaults to env name)
        model_name: Name of the model being evaluated
        start_time: Start time of evaluation (unix timestamp)
        end_time: End time of evaluation (unix timestamp)
        generation_parameters: Dictionary of generation parameters used
        samples: List of sample dictionaries to save to samples.jsonl
        verbose: If True, print a markdown table of the metrics
    """
    if eval_dir is None:
        logger.warning("eval_dir is not set, skipping evaluation logging")
        return
    # Create directory if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)

    # Generate filename
    filename = "metrics.json"
    filepath = os.path.join(eval_dir, filename)

    if start_time is None:
        start_time = time.time()
    if end_time is None:
        end_time = time.time()
    if generation_parameters is None:
        generation_parameters = {}

    # Print metrics table if verbose
    if verbose:
        from atroposlib.utils.display import display_metrics_table

        display_metrics_table(task_name, metrics, start_time, end_time)

    # Build evaluation result structure - skeleton of lighteval's
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

    # Write main results to JSON file
    with open(filepath, "w") as f:
        json.dump(eval_result, f, indent=2)

    print(f"Evaluation results saved to {filepath}")

    # Write samples to JSONL file if provided
    if samples:
        samples_filepath = os.path.join(eval_dir, "samples.jsonl")
        with jsonlines.open(samples_filepath, "w") as writer:
            for sample in samples:
                writer.write(sample)
        print(f"Evaluation samples saved to {samples_filepath}")


class EvalBase(ABC):
    """ """

    def __init__(self, pass_at_n=1, pass_at_n_samples=1, **kwargs):
        self.pass_at_n = pass_at_n
        self.pass_at_n_samples = pass_at_n_samples
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.data = self.setup_data()

    def get_generation_params(self):
        """
        Generation params to be sent to an openai server
        """
        temp = getattr(self, "temperature", 0.0)
        n = max(self.pass_at_n_samples, self.pass_at_n)
        top_p = getattr(self, "top_p", 1.0)
        max_tokens = getattr(self, "max_tokens", -1)
        return {"temperature": temp, "n": n, "top_p": top_p, "max_tokens": max_tokens}

    async def chat_completion(self, server, messages) -> ChatCompletion:
        gen_params = self.get_generation_params()
        return await server.chat_completion(messages=messages, **gen_params)

    @abstractmethod
    def setup_data(self) -> list:
        raise NotImplementedError("Setup data method must be implemented in subclass")

    @abstractmethod
    async def run_item(
        self, server: ServerManager, data_item: dict
    ) -> Tuple[dict, list]:
        """
        An abstract method that must be overridden in a subclass to define how a
        specific item should be processed. This method encapsulates the logic required
        to run or process the given data item on the provided server instance.

        Args:
            server (ServerManager): An instance of ServerManager used to manage and
                interact with server operations during the item processing.
            data_item (dict): A dictionary representing the data item to be processed.
                The structure and content of the dictionary would depend on the
                specific application and use case.

        Returns:
            Tuple[dict, list]: A tuple where the first element is a dictionary
                containing the processed results or output of the operation, and
                the second element is a list containing any additional data generated
                or collected during the item's processing.

        Raises:
            NotImplementedError: This error is raised when the method is not
                implemented in subclasses.
        """
        raise NotImplementedError("Run item method must be implemented in subclass")

    async def __call__(self, server_manager: ServerManager):
        task_coros = list()
        start_time = time.time()
        for data_item in self.data:
            task_coros.append(self.run_item(server_manager, data_item))
        task_results = await tqdm_asyncio.gather(*task_coros)
        end_time = time.time()
        # grab metrics and generation params
        metrics_list = [result[0] for result in task_results]
        # aggregate metrics
        keys = list(metrics_list[0].keys())
        metrics = {
            key: sum([result[key] for result in metrics_list]) / len(metrics_list)
            for key in keys
        }
        # check if all generation params are the same
        samples = [result[1] for result in task_results]
        task_name = self.__class__.__name__
        task_name += f"@{self.pass_at_n}"
        if self.pass_at_n != self.pass_at_n_samples:
            task_name += f":{self.pass_at_n_samples}"
        print(f"{task_name} metrics: {metrics}")
        evaluate_log(
            metrics,
            eval_dir=getattr(self, "eval_dir", None),
            task_name=task_name,
            model_name=server_manager.servers[0].config.model_name,
            start_time=start_time,
            end_time=end_time,
            generation_parameters=self.get_generation_params(),
            samples=samples,
            verbose=getattr(self, "verbose", False),
        )

        return metrics


async def eval_runner(eval_env: EvalBase):
    import argparse

    from atroposlib.envs.server_handling.server_baseline import APIServerConfig

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the server to connect to.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Model name")
    args = parser.parse_args()
    server_manager = ServerManager(
        configs=[
            APIServerConfig(
                api_key="dummy",
                base_url=args.server_url,
                model_name=args.model_name,
                health_check=False,
            ),
        ]
    )
    return await eval_env(server_manager)
