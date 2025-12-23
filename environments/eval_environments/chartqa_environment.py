import base64
import io
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

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


class ChartQAConfig(BaseEnvConfig):
    subset: str = Field(
        default="human",
        description="Subset to evaluate: 'human' or 'augmented'",
    )
    images_path: Optional[str] = Field(
        default=None,
        description="Path to images folder. If None, uses HuggingFace dataset.",
    )
    relaxed_tolerance: float = Field(
        default=0.05,
        description="Tolerance for relaxed numeric matching (5% by default)",
    )
    eval_temperature: float = Field(
        default=0.0,
        description="Temperature for eval completions",
    )
    eval_max_tokens: int = Field(
        default=2048,
        description="Max tokens for eval completions (increased for thinking models)",
    )


class ChartQAEnv(BaseEnv):
    name = "chartqa"
    env_config_cls = ChartQAConfig

    def __init__(
        self,
        config: ChartQAConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: ChartQAConfig = config

    @classmethod
    def config_init(cls) -> Tuple[ChartQAConfig, List[APIServerConfig]]:
        config = ChartQAConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            use_wandb=False,
            data_dir_to_save_evals="./eval_results/chartqa",
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
        self.dataset = load_dataset(
            "ahmed-masry/ChartQA",
            split="test",
            trust_remote_code=True,
        )

        if self.config.subset == "human":
            self.dataset = self.dataset.filter(lambda x: x.get("type", "") == "human")
        elif self.config.subset == "augmented":
            self.dataset = self.dataset.filter(
                lambda x: x.get("type", "") == "augmented"
            )

        print(
            f"Loaded {len(self.dataset)} examples from ChartQA ({self.config.subset})"
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
            imgname = item.get("imgname", "")
            image_path = Path(self.config.images_path) / imgname
            return self.encode_image_from_path(str(image_path))

        if "image" in item and item["image"] is not None:
            img = item["image"]
            if isinstance(img, bytes):
                return base64.b64encode(img).decode("utf-8")
            elif isinstance(img, Image.Image):
                return self.encode_image_from_pil(img)
            else:
                raise ValueError(f"Unknown image type: {type(img)}")

        raise ValueError("Could not find image for item")

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        query = item.get("query", item.get("question", ""))

        prompt = f"""Answer this question about the chart. Provide only the answer, nothing else.

Question: {query}"""

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

    def extract_answer(self, response: str) -> str:
        response = response.strip()

        patterns = [
            r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
            r"^(\d+[\d,\.]*%?)$",
            r"^(yes|no)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        if len(response.split()) <= 5:
            return response

        first_line = response.split("\n")[0]
        return first_line.strip()

    def score_relaxed(self, prediction: str, answer: str) -> bool:
        pred = prediction.strip()
        ans = answer.strip()

        try:
            pred_clean = pred.replace(",", "").replace("%", "").replace("$", "")
            ans_clean = ans.replace(",", "").replace("%", "").replace("$", "")

            pred_num = float(pred_clean)
            ans_num = float(ans_clean)

            if ans_num == 0:
                return abs(pred_num) < 1e-6

            return (
                abs(pred_num - ans_num) / abs(ans_num) <= self.config.relaxed_tolerance
            )
        except ValueError:
            pass

        return pred.lower() == ans.lower()

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
            extracted = self.extract_answer(response)
            answer = item.get("label", item.get("answer", ""))
            correct = self.score_relaxed(extracted, answer)

            is_numeric = False
            try:
                float(answer.replace(",", "").replace("%", "").replace("$", ""))
                is_numeric = True
            except ValueError:
                pass

            return {
                "question": item.get("query", item.get("question", "")),
                "answer": answer,
                "response": response,
                "extracted": extracted,
                "correct": correct,
                "is_numeric": is_numeric,
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

        numeric_results = [r for r in valid_results if r.get("is_numeric", False)]
        text_results = [r for r in valid_results if not r.get("is_numeric", False)]

        if numeric_results:
            num_correct = sum(1 for r in numeric_results if r["correct"])
            metrics["accuracy_numeric"] = num_correct / len(numeric_results)

        if text_results:
            text_correct = sum(1 for r in text_results if r["correct"])
            metrics["accuracy_text"] = text_correct / len(text_results)

        return metrics

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        tasks = [self.evaluate_single(item) for item in self.dataset]
        results = await tqdm_asyncio.gather(
            *tasks, desc=f"Evaluating ChartQA ({self.config.subset})"
        )

        metrics = self.compute_metrics(results)
        end_time = time.time()

        samples = [
            {
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "prediction": r.get("extracted", ""),
                "correct": r.get("correct", False),
                "is_numeric": r.get("is_numeric", False),
            }
            for r in results
            if "error" not in r
        ]

        await self.evaluate_log(
            metrics=metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            task_name=f"chartqa_{self.config.subset}",
            generation_parameters={
                "temperature": self.config.eval_temperature,
                "max_tokens": self.config.eval_max_tokens,
                "subset": self.config.subset,
                "relaxed_tolerance": self.config.relaxed_tolerance,
            },
        )


if __name__ == "__main__":
    ChartQAEnv.cli()
