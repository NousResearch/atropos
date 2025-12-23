import base64
import io
import os
import re
import time
from collections import defaultdict
from typing import List, Tuple

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


class WeMathConfig(BaseEnvConfig):
    split: str = Field(
        default="testmini",
        description="Dataset split to evaluate",
    )
    include_4d_metrics: bool = Field(
        default=True,
        description="Include four-dimensional metrics (IK, IG, CM, RM)",
    )
    eval_temperature: float = Field(
        default=0.0,
        description="Temperature for eval completions",
    )
    eval_max_tokens: int = Field(
        default=4096,
        description="Max tokens for eval completions (increased for thinking models)",
    )


class WeMathEnv(BaseEnv):
    name = "wemath"
    env_config_cls = WeMathConfig

    def __init__(
        self,
        config: WeMathConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: WeMathConfig = config

    @classmethod
    def config_init(cls) -> Tuple[WeMathConfig, List[APIServerConfig]]:
        config = WeMathConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            use_wandb=False,
            data_dir_to_save_evals="./eval_results/wemath",
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
            "We-Math/We-Math",
            split=self.config.split,
            trust_remote_code=True,
        )
        print(f"Loaded {len(self.dataset)} examples from We-Math ({self.config.split})")

    async def get_next_item(self) -> Item:
        raise NotImplementedError("Eval-only environment")

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List]:
        raise NotImplementedError("Eval-only environment")

    def encode_image_from_pil(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        img = item.get("image_path") or item.get("image")
        if img is not None:
            if isinstance(img, Image.Image):
                return self.encode_image_from_pil(img)
            elif isinstance(img, bytes):
                return base64.b64encode(img).decode("utf-8")
        raise ValueError(
            f"Could not find image for item {item.get('ID', item.get('problem_id', 'unknown'))}"
        )

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")

        prompt = f"""Look at the image and answer the math question.

Question: {question}

Provide only the final answer (a number or short phrase)."""

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
            r"(?:answer|result)[\s:=]+(.+?)(?:\.|$)",
            r"(\d+(?:\.\d+)?)",
            r"^(.+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return response

    def score(self, prediction: str, answer: str) -> bool:
        pred = prediction.strip().lower()
        ans = answer.strip().lower()

        try:
            pred_num = float(pred)
            ans_num = float(ans)
            return abs(pred_num - ans_num) < 1e-6
        except ValueError:
            pass

        return pred == ans

    def get_problem_version(self, problem_id: str) -> str:
        parts = problem_id.rsplit("_", 1)
        return parts[0] if len(parts) > 1 else problem_id

    def get_step_level(self, item: dict) -> str:
        return item.get("step", "1step")

    def calculate_4d_metrics(self, results: dict, items_by_id: dict) -> dict:
        metrics = {
            "IK": 0,
            "IG": 0,
            "CM": 0,
            "RM": 0,
            "total_groups": 0,
        }

        groups = defaultdict(dict)
        for pid, correct in results.items():
            version = self.get_problem_version(pid)
            item = items_by_id.get(pid, {})
            step = self.get_step_level(item)
            groups[version][step] = correct

        for version, steps in groups.items():
            if len(steps) < 2:
                continue

            metrics["total_groups"] += 1

            has_1step = "1step" in steps
            has_2step = "2step" in steps
            has_3step = "3step" in steps

            if has_1step and has_2step:
                p1 = steps["1step"]
                p2 = steps["2step"]

                if p1 and not p2:
                    metrics["IK"] += 1
                elif not p1 and p2:
                    metrics["RM"] += 1
                elif p1 and p2:
                    metrics["CM"] += 1
                else:
                    metrics["IG"] += 1

            elif has_2step and has_3step:
                p2 = steps["2step"]
                p3 = steps["3step"]

                if p2 and not p3:
                    metrics["IK"] += 1
                elif not p2 and p3:
                    metrics["RM"] += 1
                elif p2 and p3:
                    metrics["CM"] += 1
                else:
                    metrics["IG"] += 1

        total = metrics["total_groups"]
        if total > 0:
            for key in ["IK", "IG", "CM", "RM"]:
                metrics[f"{key}_pct"] = metrics[key] / total

        return metrics

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
            answer = item.get("answer", "")
            correct = self.score(extracted, answer)

            return {
                "problem_id": item.get("problem_id", ""),
                "problem_version": item.get("problem_version", ""),
                "question": item.get("question", ""),
                "answer": answer,
                "response": response,
                "extracted": extracted,
                "correct": correct,
                "step": item.get("step", "1step"),
                "knowledge": item.get("knowledge", ""),
                "level": item.get("level", ""),
                "subfield": item.get("subfield", ""),
            }

        except Exception as e:
            return {"correct": False, "error": str(e)}

    def compute_metrics(self, results: List[dict], items_by_id: dict) -> dict:
        valid_results = [r for r in results if "error" not in r]
        correct = sum(1 for r in valid_results if r["correct"])
        total = len(valid_results)

        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "errors": len(results) - len(valid_results),
        }

        for step in ["1step", "2step", "3step"]:
            step_results = [r for r in valid_results if r.get("step") == step]
            if step_results:
                step_correct = sum(1 for r in step_results if r["correct"])
                metrics[f"accuracy_{step}"] = step_correct / len(step_results)

        levels = set(r.get("level", "") for r in valid_results if r.get("level"))
        for level in levels:
            level_results = [r for r in valid_results if r.get("level") == level]
            if level_results:
                level_correct = sum(1 for r in level_results if r["correct"])
                metrics[f"accuracy_{level}"] = level_correct / len(level_results)

        if self.config.include_4d_metrics:
            results_dict = {r["problem_id"]: r["correct"] for r in valid_results}
            fd_metrics = self.calculate_4d_metrics(results_dict, items_by_id)
            metrics.update(fd_metrics)

        return metrics

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        items_by_id = {item.get("problem_id", ""): item for item in self.dataset}

        tasks = [self.evaluate_single(item) for item in self.dataset]
        results = await tqdm_asyncio.gather(
            *tasks, desc=f"Evaluating We-Math ({self.config.split})"
        )

        metrics = self.compute_metrics(results, items_by_id)
        end_time = time.time()

        samples = [
            {
                "problem_id": r.get("problem_id", ""),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "prediction": r.get("extracted", ""),
                "correct": r.get("correct", False),
                "step": r.get("step", ""),
                "knowledge": r.get("knowledge", ""),
                "level": r.get("level", ""),
            }
            for r in results
            if "error" not in r
        ]

        await self.evaluate_log(
            metrics=metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            task_name=f"wemath_{self.config.split}",
            generation_parameters={
                "temperature": self.config.eval_temperature,
                "max_tokens": self.config.eval_max_tokens,
                "include_4d_metrics": self.config.include_4d_metrics,
            },
        )


if __name__ == "__main__":
    WeMathEnv.cli()
