import asyncio
import base64
import io
import json
import zipfile
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import hf_hub_download
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.eval_environments.eval import EvalBase, eval_runner

DEFAULT_DATA_DIR = Path.home() / ".cache" / "visulogic_hf"


class VisuLogic(EvalBase):
    TAGS = [
        "Quantitative Reasoning",
        "Spatial Reasoning",
        "Positional Reasoning",
        "Attribute Reasoning",
        "Stylistic Reasoning",
        "Other",
    ]

    def _download_data(self, data_dir: Path) -> None:
        jsonl_path = data_dir / "data.jsonl"
        images_dir = data_dir / "images"

        if jsonl_path.exists() and images_dir.exists():
            return

        print(f"Downloading VisuLogic dataset to {data_dir}...")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Download data.jsonl
        hf_hub_download(
            repo_id="VisuLogic/VisuLogic",
            filename="data.jsonl",
            repo_type="dataset",
            local_dir=data_dir,
        )

        # Download and extract images.zip
        images_zip_path = hf_hub_download(
            repo_id="VisuLogic/VisuLogic",
            filename="images.zip",
            repo_type="dataset",
            local_dir=data_dir,
        )

        print("Extracting images...")
        with zipfile.ZipFile(images_zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        print("Download complete!")

    def setup_data(self) -> list:
        """
        Load and return dataset as a list.

        Auto-downloads the VisuLogic dataset if data_path is not specified
        or doesn't exist.
        """
        data_path = getattr(self, "data_path", None)

        if data_path is None:
            data_dir = DEFAULT_DATA_DIR
            self._download_data(data_dir)
            jsonl_path = data_dir / "data.jsonl"
            self.images_base = str(data_dir)
        else:
            data_dir = Path(data_path)
            jsonl_path = data_dir / "data.jsonl"
            self.images_base = str(data_dir)

            if not jsonl_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {jsonl_path}. "
                    "Remove data_path argument to auto-download."
                )

        dataset = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                dataset.append(item)

        print(f"Loaded {len(dataset)} examples from VisuLogic")
        return dataset

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        image_path = item.get("image_path", "")
        full_path = Path(self.images_base) / image_path
        if full_path.exists():
            with Image.open(full_path) as img:
                return self.encode_image(img)
        raise ValueError(f"Could not find image at {full_path}")

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")

        prompt = f"""{question}

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

        for char in reversed(response):
            if char in "ABCD":
                return char

        return ""

    def score(self, prediction: str, answer: str) -> bool:
        if not prediction:
            return False
        return prediction.upper() == answer.upper()

    async def run_item(self, server: ServerManager, data_item: dict) -> Tuple[dict, dict]:
        try:
            messages = self.build_messages(data_item)

            completion = await self.chat_completion(server, messages)

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
            answer = data_item.get("label", "")
            correct = self.score(extracted, answer)

            sample = {
                "question": data_item.get("question", ""),
                "answer": answer,
                "prediction": extracted,
                "correct": correct,
                "tag": data_item.get("tag", ""),
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(eval_runner(VisuLogic(temperature=0.0, max_tokens=256)))
