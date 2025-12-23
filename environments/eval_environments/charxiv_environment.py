import base64
import io
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import openai
from datasets import load_dataset
from PIL import Image
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    Item,
    ScoredDataGroup,
)


class CharXivConfig(BaseEnvConfig):
    mode: str = Field(
        default="reasoning",
        description="Evaluation mode: 'descriptive' or 'reasoning'",
    )
    split: str = Field(default="val", description="Dataset split: 'val' or 'test'")
    images_path: Optional[str] = Field(
        default=None,
        description="Path to images folder. If None, downloads from HuggingFace.",
    )
    judge_model: str = Field(
        default="gpt-4o",
        description="Model for grading descriptive answers",
    )
    judge_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for judge model API",
    )
    judge_api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable for judge API key",
    )
    eval_temperature: float = Field(
        default=0.0,
        description="Temperature for eval completions",
    )
    eval_max_tokens: int = Field(
        default=4096,
        description="Max tokens for eval completions (increased for thinking models)",
    )


class CharXivEnv(BaseEnv):
    name = "charxiv"
    env_config_cls = CharXivConfig

    CATEGORY_NAMES = {
        "descriptive": {
            0: "INEX",  # Information Extraction
            1: "ENUM",  # Enumeration
            2: "PATT",  # Pattern Recognition
            3: "CNTG",  # Counting
            4: "COMP",  # Compositionality
        },
        "reasoning": {
            0: "TC",  # Text-in-Chart
            1: "TG",  # Text-in-General
            2: "NC",  # Number-in-Chart
            3: "NG",  # Number-in-General
        },
    }

    def __init__(
        self,
        config: CharXivConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: CharXivConfig = config

        judge_api_key = os.environ.get(self.config.judge_api_key_env)
        if self.config.mode == "descriptive" and not judge_api_key:
            raise ValueError(
                f"Judge API key required for descriptive mode. Set {self.config.judge_api_key_env}"
            )

        if judge_api_key:
            self.judge_client = openai.AsyncOpenAI(
                api_key=judge_api_key,
                base_url=self.config.judge_base_url,
            )

    @classmethod
    def config_init(cls) -> Tuple[CharXivConfig, List[APIServerConfig]]:
        config = CharXivConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            use_wandb=False,
            data_dir_to_save_evals="./eval_results/charxiv",
            max_eval_workers=32,
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4o",
                base_url="https://api.openai.com/v1",
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                num_requests_for_eval=32,
            )
        ]
        return config, server_configs

    async def setup(self):
        dataset_name = "princeton-nlp/CharXiv"
        split_name = "validation" if self.config.split == "val" else "test"

        raw_dataset = load_dataset(dataset_name, split=split_name)

        # Transform dataset based on mode
        self.dataset = []
        for item in raw_dataset:
            if self.config.mode == "reasoning":
                # Reasoning mode: single question per image
                self.dataset.append(
                    {
                        "image": item["image"],
                        "figure_id": item.get("original_id", ""),
                        "query": item["reasoning_q"],
                        "answer": item["reasoning_a"],
                        "qa_source": item.get("reasoning_q_source", 0),
                        "category": item.get("category", ""),
                    }
                )
            else:
                # Descriptive mode: up to 4 questions per image
                for i in range(1, 5):
                    q_key = f"descriptive_q{i}"
                    a_key = f"descriptive_a{i}"
                    if item.get(q_key) and item.get(a_key):
                        self.dataset.append(
                            {
                                "image": item["image"],
                                "figure_id": item.get("original_id", ""),
                                "query": item[q_key],
                                "answer": item[a_key],
                                "inst_category": i - 1,
                                "category": item.get("category", ""),
                            }
                        )

        print(
            f"Loaded {len(self.dataset)} examples from CharXiv ({self.config.mode}, {self.config.split})"
        )

    async def get_next_item(self) -> Item:
        raise NotImplementedError("Eval-only environment")

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List]:
        raise NotImplementedError("Eval-only environment")

    def encode_image_from_pil(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def encode_image_from_path(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        if self.config.images_path:
            figure_id = item.get("figure_id", item.get("id", 0))
            image_path = Path(self.config.images_path) / f"{figure_id}.png"
            if not image_path.exists():
                image_path = Path(self.config.images_path) / f"{figure_id}.jpg"
            return self.encode_image_from_path(str(image_path))

        if "image" in item and item["image"] is not None:
            return self.encode_image_from_pil(item["image"])

        raise ValueError(
            f"Could not find image for item: {item.get('figure_id', 'unknown')}"
        )

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        query = item.get("query", item.get("question", ""))

        if self.config.mode == "descriptive":
            instruction = (
                "Answer the question about this chart. Be concise and specific."
            )
        else:
            instruction = "Analyze this chart and answer the question. Provide your answer directly."

        prompt = f"{instruction}\n\nQuestion: {query}"

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def score_reasoning(self, prediction: str, answer: str) -> bool:
        pred = prediction.strip().lower()
        ans = answer.strip().lower()

        if ans in pred:
            return True
        if pred == ans:
            return True

        pred_clean = re.sub(r"[^a-z0-9\s]", "", pred)
        ans_clean = re.sub(r"[^a-z0-9\s]", "", ans)

        return ans_clean in pred_clean or pred_clean == ans_clean

    async def score_descriptive(self, query: str, prediction: str, answer: str) -> bool:
        prompt = f"""Evaluate if the model's response correctly answers the question about the chart.

Question: {query}
Correct Answer: {answer}
Model Response: {prediction}

Does the model's response correctly answer the question? Consider:
- The core information matches
- Minor wording differences are acceptable
- The response addresses what was asked

Output only "1" if the response is correct, or "0" if incorrect."""

        try:
            response = await self.judge_client.chat.completions.create(
                model=self.config.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            result = response.choices[0].message.content.strip()
            return "1" in result
        except Exception as e:
            print(f"Judge error: {e}")
            return False

    async def evaluate_single(self, item: dict) -> dict:
        try:
            messages = self.build_messages(item)

            completion = await self.server.chat_completion(
                messages=messages,
                n=1,
                max_tokens=self.config.eval_max_tokens,
                temperature=self.config.eval_temperature,
                split="eval",
            )

            if not completion.choices:
                return {"correct": False, "error": "Empty response"}

            message = completion.choices[0].message
            response = message.content or ""
            if hasattr(message, "reasoning") and message.reasoning and not response:
                response = message.reasoning
            if not response and hasattr(message, "model_extra"):
                reasoning = message.model_extra.get("reasoning", "")
                if reasoning:
                    response = reasoning

            if not response:
                return {"correct": False, "error": "Empty response"}
            answer = item.get("answer", "")
            query = item.get("query", item.get("question", ""))

            if self.config.mode == "descriptive":
                correct = await self.score_descriptive(query, response, answer)
            else:
                correct = self.score_reasoning(response, answer)

            category_id = item.get("inst_category", item.get("qa_source", 0))
            category_name = self.CATEGORY_NAMES.get(self.config.mode, {}).get(
                category_id, "unknown"
            )

            return {
                "figure_id": item.get("figure_id", ""),
                "question": query,
                "answer": answer,
                "prediction": response,
                "correct": correct,
                "category": category_name,
            }

        except Exception as e:
            return {"correct": False, "error": str(e)}

    def compute_metrics(self, results: List[dict]) -> dict:
        valid_results = [r for r in results if "error" not in r]
        correct = sum(1 for r in valid_results if r["correct"])
        total = len(valid_results)

        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "errors": len(results) - len(valid_results),
        }

        categories = set(r.get("category", "unknown") for r in valid_results)
        for cat in categories:
            cat_results = [r for r in valid_results if r.get("category") == cat]
            if cat_results:
                cat_correct = sum(1 for r in cat_results if r["correct"])
                metrics[f"accuracy_{cat}"] = cat_correct / len(cat_results)

        return metrics

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        tasks = [self.evaluate_single(item) for item in self.dataset]
        results = await tqdm_asyncio.gather(
            *tasks, desc=f"Evaluating CharXiv ({self.config.mode})"
        )

        metrics = self.compute_metrics(results)
        end_time = time.time()

        samples = [
            {
                "figure_id": r.get("figure_id", ""),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "prediction": r.get("prediction", ""),
                "correct": r.get("correct", False),
                "category": r.get("category", ""),
            }
            for r in results
            if "error" not in r
        ]

        await self.evaluate_log(
            metrics=metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            task_name=f"charxiv_{self.config.mode}",
            generation_parameters={
                "temperature": self.config.eval_temperature,
                "max_tokens": self.config.eval_max_tokens,
                "mode": self.config.mode,
            },
        )


if __name__ == "__main__":
    CharXivEnv.cli()
