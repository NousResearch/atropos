import base64
import io
import os
import re
import time
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


class MathVistaConfig(BaseEnvConfig):
    split: str = Field(
        default="testmini",
        description="Dataset split: 'testmini' (1000 examples with answers) or 'test' (5141, answers withheld)",
    )
    use_query: bool = Field(
        default=True,
        description="Use provided query prompts with task-specific hints",
    )
    eval_temperature: float = Field(
        default=0.0,
        description="Temperature for eval completions",
    )
    eval_max_tokens: int = Field(
        default=4096,
        description="Max tokens for eval completions (increased for thinking models)",
    )


class MathVistaEnv(BaseEnv):
    name = "mathvista"
    env_config_cls = MathVistaConfig

    TASK_TYPES = ["FQA", "GPS", "MWP", "TQA", "VQA"]
    SKILL_TYPES = ["ALG", "ARI", "GEO", "LOG", "NUM", "SCI", "STA"]

    def __init__(
        self,
        config: MathVistaConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: MathVistaConfig = config

    @classmethod
    def config_init(cls) -> Tuple[MathVistaConfig, List[APIServerConfig]]:
        config = MathVistaConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            use_wandb=False,
            data_dir_to_save_evals="./eval_results/mathvista",
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
            "AI4Math/MathVista",
            split=self.config.split,
            trust_remote_code=True,
        )
        print(
            f"Loaded {len(self.dataset)} examples from MathVista ({self.config.split})"
        )

    async def get_next_item(self) -> Item:
        raise NotImplementedError("Eval-only environment")

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List]:
        raise NotImplementedError("Eval-only environment")

    def encode_image_from_pil(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        if "decoded_image" in item and item["decoded_image"] is not None:
            return self.encode_image_from_pil(item["decoded_image"])
        if "image" in item and item["image"] is not None:
            if isinstance(item["image"], Image.Image):
                return self.encode_image_from_pil(item["image"])
        raise ValueError(f"Could not find image for item {item.get('pid', 'unknown')}")

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)

        if self.config.use_query and "query" in item:
            prompt = item["query"]
        else:
            prompt = self._build_custom_prompt(item)

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

    def _build_custom_prompt(self, item: dict) -> str:
        question = item.get("question", "")
        question_type = item.get("question_type", "free_form")
        answer_type = item.get("answer_type", "text")
        precision = item.get("precision", 2)

        if question_type == "multi_choice":
            choices = item.get("choices", [])
            choices_text = "\n".join(choices) if choices else ""
            hint = "Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
            return f"Hint: {hint}\nQuestion: {question}\nChoices:\n{choices_text}"

        if answer_type == "integer":
            hint = (
                "Please answer the question requiring an integer answer "
                "and provide the final value, e.g., 1, 2, 3, at the end."
            )
        elif answer_type == "float":
            hint = (
                f"Please answer the question requiring a floating-point number "
                f"with {precision} decimal place(s) and provide the final value at the end."
            )
        elif answer_type == "list":
            hint = (
                "Please answer the question requiring a Python list as an answer "
                "and provide the final list, e.g., [1, 2, 3], at the end."
            )
        else:
            hint = "Please answer the question and provide the final answer at the end."

        return f"Hint: {hint}\nQuestion: {question}"

    def extract_answer(
        self, response: str, answer_type: str, question_type: str
    ) -> str:
        response = response.strip()

        if question_type == "multi_choice":
            for char in reversed(response.upper()):
                if char in "ABCDEFGH":
                    return char
            return ""

        if answer_type == "integer":
            numbers = re.findall(r"-?\d+", response)
            return numbers[-1] if numbers else ""

        if answer_type == "float":
            numbers = re.findall(r"-?\d+\.?\d*", response)
            return numbers[-1] if numbers else ""

        if answer_type == "list":
            match = re.search(r"\[[\d\.,\s-]+\]", response)
            return match.group(0) if match else ""

        return response

    def score(
        self, prediction: str, answer: str, answer_type: str, precision: int = 0
    ) -> bool:
        pred = prediction.strip()
        ans = answer.strip()

        if not pred:
            return False

        if answer_type == "text":
            return pred.upper() == ans.upper()

        if answer_type == "integer":
            try:
                return int(float(pred)) == int(float(ans))
            except (ValueError, OverflowError):
                return False

        if answer_type == "float":
            try:
                tolerance = 10 ** (-precision) if precision > 0 else 0.01
                return abs(float(pred) - float(ans)) < tolerance
            except ValueError:
                return False

        if answer_type == "list":
            try:
                pred_list = eval(pred)
                ans_list = eval(ans)
                return pred_list == ans_list
            except Exception:
                return False

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
            answer_type = item.get("answer_type", "text")
            question_type = item.get("question_type", "free_form")
            precision = item.get("precision", 0)

            extracted = self.extract_answer(response, answer_type, question_type)
            answer = item.get("answer", "")
            correct = self.score(extracted, answer, answer_type, precision)

            metadata = item.get("metadata", {})
            task = metadata.get("task", "")
            skills = metadata.get("skills", [])

            task_abbrev = ""
            for abbrev in self.TASK_TYPES:
                if (
                    task.lower().startswith(abbrev.lower())
                    or abbrev.lower() in task.lower()
                ):
                    task_abbrev = abbrev
                    break

            skill_abbrevs = []
            for skill in skills:
                skill_lower = skill.lower()
                for abbrev in self.SKILL_TYPES:
                    if abbrev.lower() in skill_lower or skill_lower.startswith(
                        abbrev.lower()[:3]
                    ):
                        skill_abbrevs.append(abbrev)
                        break

            return {
                "pid": item.get("pid", ""),
                "question": item.get("question", ""),
                "answer": answer,
                "response": response,
                "extracted": extracted,
                "correct": correct,
                "question_type": question_type,
                "answer_type": answer_type,
                "task": task_abbrev,
                "skills": skill_abbrevs,
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

        for task in self.TASK_TYPES:
            task_results = [r for r in valid_results if r.get("task") == task]
            if task_results:
                task_correct = sum(1 for r in task_results if r["correct"])
                metrics[f"accuracy_{task}"] = task_correct / len(task_results)

        for skill in self.SKILL_TYPES:
            skill_results = [r for r in valid_results if skill in r.get("skills", [])]
            if skill_results:
                skill_correct = sum(1 for r in skill_results if r["correct"])
                metrics[f"accuracy_{skill}"] = skill_correct / len(skill_results)

        for qtype in ["multi_choice", "free_form"]:
            qtype_results = [
                r for r in valid_results if r.get("question_type") == qtype
            ]
            if qtype_results:
                qtype_correct = sum(1 for r in qtype_results if r["correct"])
                metrics[f"accuracy_{qtype}"] = qtype_correct / len(qtype_results)

        return metrics

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        tasks = [self.evaluate_single(item) for item in self.dataset]
        results = await tqdm_asyncio.gather(
            *tasks, desc=f"Evaluating MathVista ({self.config.split})"
        )

        metrics = self.compute_metrics(results)
        end_time = time.time()

        samples = [
            {
                "pid": r.get("pid", ""),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "prediction": r.get("extracted", ""),
                "correct": r.get("correct", False),
                "question_type": r.get("question_type", ""),
                "answer_type": r.get("answer_type", ""),
                "task": r.get("task", ""),
                "skills": r.get("skills", []),
            }
            for r in results
            if "error" not in r
        ]

        await self.evaluate_log(
            metrics=metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            task_name=f"mathvista_{self.config.split}",
            generation_parameters={
                "temperature": self.config.eval_temperature,
                "max_tokens": self.config.eval_max_tokens,
                "use_query": self.config.use_query,
            },
        )


if __name__ == "__main__":
    MathVistaEnv.cli()
