import base64
import io
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

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


class LogicVistaConfig(BaseEnvConfig):
    data_path: str = Field(
        default="./data/dataset.json",
        description="Path to LogicVista dataset JSON file",
    )
    images_base: str = Field(
        default="./data",
        description="Base path for images (image_path is relative to this)",
    )
    eval_temperature: float = Field(
        default=0.0,
        description="Temperature for eval completions",
    )
    eval_max_tokens: int = Field(
        default=256,
        description="Max tokens for eval completions",
    )


class LogicVistaEnv(BaseEnv):
    name = "logicvista"
    env_config_cls = LogicVistaConfig

    REASONING_SKILLS = [
        "Inductive Reasoning",
        "Deductive Reasoning",
        "Numerical Reasoning",
        "Spatial Reasoning",
        "Mechanical Reasoning",
    ]

    CAPABILITIES = [
        "diagrams",
        "OCR",
        "patterns",
        "graphs",
        "tables",
        "3D Shapes",
        "puzzles",
        "sequences",
        "physics",
    ]

    def __init__(
        self,
        config: LogicVistaConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: LogicVistaConfig = config

    @classmethod
    def config_init(cls) -> Tuple[LogicVistaConfig, List[APIServerConfig]]:
        config = LogicVistaConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            use_wandb=False,
            data_dir_to_save_evals="./eval_results/logicvista",
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
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        with open(data_path, "r") as f:
            self.dataset = json.load(f)

        print(f"Loaded {len(self.dataset)} examples from LogicVista")

    async def get_next_item(self) -> Item:
        raise NotImplementedError("Eval-only environment")

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List]:
        raise NotImplementedError("Eval-only environment")

    def encode_image_from_path(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def encode_image_from_pil(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        image_path = item.get("image_path", "")
        full_path = Path(self.config.images_base) / image_path
        return self.encode_image_from_path(str(full_path))

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")
        options = item.get("options", [])

        options_text = "\n".join(
            [f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)]
        )

        prompt = f"""Look at the image and answer the multiple choice question.

Question: {question}

Options:
{options_text}

Answer with only the letter (A, B, C, or D)."""

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

    def extract_answer(self, response: str) -> Optional[str]:
        response = response.strip().upper()

        for char in response:
            if char in "ABCD":
                return char

        return None

    def score(self, prediction: Optional[str], answer_idx: int) -> bool:
        if prediction is None:
            return False

        pred_idx = ord(prediction) - ord("A")
        return pred_idx == answer_idx

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

            if not completion.choices or not completion.choices[0].message.content:
                return {"correct": False, "error": "Empty response"}

            response = completion.choices[0].message.content
            extracted = self.extract_answer(response)
            answer_idx = item.get("answer", 0)
            correct = self.score(extracted, answer_idx)

            correct_letter = chr(65 + answer_idx)

            return {
                "index": item.get("index", ""),
                "question": item.get("question", ""),
                "options": item.get("options", []),
                "answer": correct_letter,
                "response": response,
                "extracted": extracted,
                "correct": correct,
                "skill": item.get("skill", ""),
                "caps": item.get("caps", []),
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

        for skill in self.REASONING_SKILLS:
            skill_results = [r for r in valid_results if r.get("skill") == skill]
            if skill_results:
                skill_correct = sum(1 for r in skill_results if r["correct"])
                key = skill.lower().replace(" ", "_")
                metrics[f"accuracy_{key}"] = skill_correct / len(skill_results)

        for cap in self.CAPABILITIES:
            cap_results = [r for r in valid_results if cap in r.get("caps", [])]
            if cap_results:
                cap_correct = sum(1 for r in cap_results if r["correct"])
                key = cap.lower().replace(" ", "_")
                metrics[f"accuracy_{key}"] = cap_correct / len(cap_results)

        return metrics

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        tasks = [self.evaluate_single(item) for item in self.dataset]
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating LogicVista")

        metrics = self.compute_metrics(results)
        end_time = time.time()

        samples = [
            {
                "index": r.get("index", ""),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "prediction": r.get("extracted", ""),
                "correct": r.get("correct", False),
                "skill": r.get("skill", ""),
                "caps": r.get("caps", []),
            }
            for r in results
            if "error" not in r
        ]

        await self.evaluate_log(
            metrics=metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            task_name="logicvista",
            generation_parameters={
                "temperature": self.config.eval_temperature,
                "max_tokens": self.config.eval_max_tokens,
            },
        )


if __name__ == "__main__":
    LogicVistaEnv.cli()
