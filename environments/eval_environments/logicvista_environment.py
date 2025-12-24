import asyncio
import base64
import io
import json
import subprocess
from pathlib import Path
from typing import List, Tuple

from openai import AsyncOpenAI
from PIL import Image

from environments.eval_environments.eval_base import EvalBase, eval_runner

LOGICVISTA_REPO = "https://github.com/Yijia-Xiao/LogicVista.git"
DEFAULT_DATA_DIR = Path.home() / ".cache" / "logicvista"


class LogicVista(EvalBase):
    """
    LogicVista evaluation environment.

    A benchmark for logical reasoning in visual contexts.
    https://logicvista.github.io/

    448 visual multiple-choice questions across 5 reasoning skills:
    - Inductive Reasoning
    - Deductive Reasoning
    - Numerical Reasoning
    - Spatial Reasoning
    - Mechanical Reasoning
    """

    REASONING_SKILLS = [
        "inductive",
        "deductive",
        "numerical",
        "spatial",
        "mechanical",
    ]

    CAPABILITIES = [
        "diagram",
        "ocr",
        "patterns",
        "graphs",
        "tables",
        "3d shapes",
        "puzzles",
        "sequences",
        "physics",
    ]

    def _download_data(self, data_dir: Path) -> None:
        """Clone LogicVista repo if not present."""
        if data_dir.exists() and (data_dir / "data" / "dataset.json").exists():
            return

        print(f"Downloading LogicVista dataset to {data_dir}...")
        data_dir.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["git", "clone", "--depth", "1", LOGICVISTA_REPO, str(data_dir)],
            check=True,
            capture_output=True,
        )
        print("Download complete!")

    def setup_data(self) -> list:
        """
        Load and return dataset as a list.

        Auto-downloads the LogicVista repo if data_path is not specified
        or doesn't exist.
        """
        data_path = getattr(self, "data_path", None)

        if data_path is None:
            data_dir = DEFAULT_DATA_DIR
            self._download_data(data_dir)
            data_path = data_dir / "data" / "dataset.json"
            self.images_base = str(data_dir / "data" / "images")
        else:
            data_path = Path(data_path)
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {data_path}. "
                    "Remove data_path argument to auto-download."
                )

        with open(data_path, "r") as f:
            raw_data = json.load(f)

        dataset = []
        for item_id, item in raw_data.items():
            dataset.append(
                {
                    "id": item_id,
                    "imagename": item.get("imagename", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "reasoning": item.get("reasoning", ""),
                    "skill": item.get("skill", []),
                    "broad_capability": item.get("broad_capability", []),
                }
            )

        print(f"Loaded {len(dataset)} examples from LogicVista")
        return dataset

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        images_base = getattr(self, "images_base", "./data/images")
        imagename = item.get("imagename", "")
        full_path = Path(images_base) / imagename
        with open(full_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")

        prompt = f"""Look at the image and answer the multiple choice question.

{question}

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

    def extract_answer(self, response: str) -> str:
        response = response.strip().upper()

        for char in response:
            if char in "ABCD":
                return char

        return ""

    def score(self, prediction: str, answer: str) -> bool:
        if not prediction:
            return False
        return prediction.upper() == answer.upper()

    async def run_item(self, client: AsyncOpenAI, data_item: dict) -> Tuple[dict, dict]:
        try:
            messages = self.build_messages(data_item)

            gen_params = self.get_generation_params()
            completion = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=gen_params["temperature"],
                max_tokens=gen_params["max_tokens"],
            )

            if not completion.choices:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            message = completion.choices[0].message
            response = message.content or ""
            if hasattr(message, "reasoning") and message.reasoning and not response:
                response = message.reasoning
            if not response and hasattr(message, "model_extra"):
                reasoning = message.model_extra.get("reasoning", "")
                if reasoning:
                    response = reasoning

            if not response:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            extracted = self.extract_answer(response)
            answer = data_item.get("answer", "")
            correct = self.score(extracted, answer)

            skill = data_item.get("skill", [])
            skill_str = skill[0] if skill else ""

            sample = {
                "id": data_item.get("id", ""),
                "question": data_item.get("question", ""),
                "answer": answer,
                "prediction": extracted,
                "correct": correct,
                "skill": skill_str,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            LogicVista,
            temperature=0.0,
            max_tokens=256,
        )
    )
