"""OCRBench evaluation environment."""

import asyncio
import base64
import io
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from environments.eval_environments.eval import EvalBase, eval_runner
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager


class OCRBench(EvalBase):
    """OCRBench evaluation - OCR capabilities benchmark."""

    # Categories and their scoring
    CATEGORIES = [
        "Regular Text Recognition",
        "Irregular Text Recognition",
        "Artistic Text Recognition",
        "Handwriting Recognition",
        "Digit String Recognition",
        "Non-Semantic Text Recognition",
        "Scene Text-centric VQA",
        "Doc-oriented VQA",
        "Key Information Extraction",
        "Handwritten Mathematical Expression Recognition",
    ]

    def setup_data(self) -> list:
        split = getattr(self, "split", "test")

        try:
            dataset = load_dataset("echo840/OCRBench", split=split)
            print(f"Loaded {len(dataset)} examples from OCRBench ({split})")
            return list(dataset)
        except Exception as e:
            print(f"Warning: Could not load OCRBench: {e}")
            try:
                dataset = load_dataset("lmms-lab/OCRBench", split=split)
                print(f"Loaded {len(dataset)} examples from OCRBench ({split})")
                return list(dataset)
            except Exception:
                raise ValueError(f"Could not load OCRBench dataset: {e}")

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> Optional[str]:
        for key in ["image", "decoded_image"]:
            if key in item and item[key] is not None:
                if isinstance(item[key], Image.Image):
                    return self.encode_image(item[key])
        return None

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")

        prompt = f"{question}\n\nAnswer the question using a single word or phrase."

        content = []
        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def score_ocr(self, prediction: str, answers: List[str], category: str) -> bool:
        """Category-specific scoring for OCR tasks."""
        predict = prediction.strip()

        if category == "Handwritten Mathematical Expression Recognition":
            predict_clean = predict.replace("\n", " ").replace(" ", "")
            for answer in answers:
                answer_clean = answer.strip().replace("\n", " ").replace(" ", "")
                if answer_clean in predict_clean:
                    return True
        else:
            predict_lower = predict.lower().replace("\n", " ")
            for answer in answers:
                answer_lower = answer.lower().strip().replace("\n", " ")
                if answer_lower in predict_lower:
                    return True

        return False

    async def run_item(
        self, server: ServerManager, data_item: dict
    ) -> Tuple[dict, dict]:
        try:
            messages = self.build_messages(data_item)
            completion = await self.chat_completion(server, messages)

            if not completion.choices:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            message = completion.choices[0].message
            response = message.content or ""

            if not response:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            answers = data_item.get("answer", [])
            if isinstance(answers, str):
                try:
                    answers = eval(answers)
                except Exception:
                    answers = [answers]
            if not isinstance(answers, list):
                answers = [answers]

            category = data_item.get("category", "")

            correct = self.score_ocr(response, answers, category)

            sample = {
                "id": data_item.get("index", data_item.get("id", "")),
                "question": data_item.get("question", "")[:200],
                "category": category,
                "answer": answers[0] if answers else "",
                "prediction": response[:200],
                "correct": correct,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(eval_runner(OCRBench(split="test", temperature=0.0, max_tokens=256)))
