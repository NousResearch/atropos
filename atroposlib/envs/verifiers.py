import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import verifiers as vf
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item

logger = logging.getLogger(__name__)


class VfEnvConfig(BaseEnvConfig):
    """Configuration for the Verifiers Environment."""

    vf_env_name: str = Field(
        default="", description="Name of the verifiers environment to load"
    )
    env_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments for environment initialization",
    )


class VerifiersEnv(BaseEnv):
    """
    Environment wrapper for the 'verifiers' library.
    Allows running RL/inference on tasks defined in the verifiers ecosystem.
    """

    name = "verifiers"
    env_config_cls = VfEnvConfig

    def __init__(
        self,
        config: VfEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        logger.info(f"Loading verifiers environment: {config.vf_env_name}")
        self.vf_env = vf.load_environment(config.vf_env_name, **config.env_args)
        self.rubric = self.vf_env.rubric

        self.parser = self.rubric.parser
        self.reward_funcs = self.rubric.get_reward_funcs()
        self.reward_weights = self.rubric.get_reward_weights()
        total_weight = sum(self.reward_weights)
        self.reward_scales = [
            weight / total_weight if total_weight != 0 else 0
            for weight in self.reward_weights
        ]
        self.system_prompt = self.vf_env.system_prompt
        self.train_data: List[Item] = []
        self.test_data: List[Item] = []
        self.iter = 0

    async def setup(self) -> None:
        """Setup the environment by loading datasets"""
        self.train_data = self.vf_env.get_dataset()
        raw_test_data = self.vf_env.get_eval_dataset()

        self.test_data = [
            {"question": item["question"], "answer": item["answer"]}
            for item in raw_test_data
        ]
        self.iter = 0

        await self.setup_wandb()
        await self.register_env()

    async def get_next_item(self) -> Item:
        if not self.train_data:
            raise ValueError("Training data is empty.")
        next_item = self.train_data[self.iter % len(self.train_data)]
        self.iter += 1
        return next_item

    async def _perform_rollout(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
        state: Optional[Any] = None,
        temperature: float = 1.0,
    ) -> Tuple[Optional[str], float, List[Dict[str, str]], List[int], List[int]]:
        """
        Shared logic for generating a response, parsing it, and calculating the score.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        try:
            completion = await self.server.chat_completion(
                messages=messages,
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Error during chat completion: {e}")
            return None, 0.0, [], [], []

        response_content = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": response_content})

        # Calculate rewards
        rewards = [
            await self.rubric.call_reward_func(
                func=func,
                prompt=question,
                completion=messages,
                answer=gold_answer,
                info=info,
                state=state,
            )
            for func in self.reward_funcs
        ]

        weighted_rewards = [
            reward * scale for reward, scale in zip(rewards, self.reward_scales)
        ]
        total_score = sum(weighted_rewards)

        # Tokenize (Strict)
        if hasattr(self.tokenizer, "apply_chat_template"):
            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        else:
            # Simple fallback for tokenizers without chat templates
            # We strictly log a warning rather than silently formatting poorly.
            logger.warning(
                "Tokenizer does not support apply_chat_template. Falling back to simple encoding."
            )
            chat_text = f"{self.system_prompt}\n\nUser: {question}\n\nAssistant: {response_content}"
            tokens = self.tokenizer.encode(chat_text)

        masks = [1] * len(tokens)

        return response_content, total_score, messages, tokens, masks

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        question = item.get("question")
        gold_answer = item.get("answer")

        if not question:
            logger.error("Item missing 'question' field")
            return None, []

        (
            response_content,
            total_score,
            messages,
            tokens,
            masks,
        ) = await self._perform_rollout(
            question=question,
            gold_answer=gold_answer,
            info=item.get("info"),
            state=item.get("state"),
            temperature=1.0,
        )

        if response_content is None:
            return None, []

        scored_item: ScoredDataItem = {
            "tokens": tokens,
            "masks": masks,
            "scores": float(total_score),
            "advantages": None,
            "ref_logprobs": None,
            "messages": messages,
            "group_overrides": None,
            "overrides": None,
            "images": None,
        }

        return scored_item, []

    async def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        start_time = time.time()
        eval_tasks = []

        async def evaluate_single(item: Item) -> Dict[str, Any]:
            question = item["question"]
            gold_answer = item["answer"]

            response_content, score, messages, _, _ = await self._perform_rollout(
                question=question,
                gold_answer=gold_answer,
                temperature=0.0,
            )

            answer_parsed = (
                self.parser.parse_answer(completion=response_content)
                if response_content
                else None
            )

            return {
                "score": score,
                "sample": {
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_parsed": str(answer_parsed) if answer_parsed else None,
                    "response": response_content,
                    "score": score,
                },
            }

        for item in self.test_data:
            eval_tasks.append(evaluate_single(item))

        results = await tqdm_asyncio.gather(*eval_tasks)

        scores = [result["score"] for result in results]
        samples = [result["sample"] for result in results]
        avg_total_score = sum(scores) / (len(scores) or 1)

        end_time = time.time()
        eval_metrics = {"eval/avg_total_score": avg_total_score}

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": self.config.max_token_length,
            },
        )

        return eval_metrics
