"""ChartQA evaluation environment."""

import asyncio
import base64
import io
import re
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import load_dataset
from environments.eval_environments.eval import EvalBase, eval_runner
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager


class ChartQA(EvalBase):
    """
    ChartQA evaluation environment.

    A benchmark for question answering about charts with relaxed accuracy scoring.
    """

    def setup_data(self) -> list:
        subset = getattr(self, "subset", "human")
        dataset = load_dataset("ahmed-masry/ChartQA", split="test")

        if subset == "human":
            dataset = dataset.filter(lambda x: x.get("type", "") == "human")
        elif subset == "augmented":
            dataset = dataset.filter(lambda x: x.get("type", "") == "augmented")

        print(f"Loaded {len(dataset)} examples from ChartQA ({subset})")
        return list(dataset)

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        images_path: Optional[str] = getattr(self, "images_path", None)
        if images_path:
            imgname = item.get("imgname", "")
            image_path = Path(images_path) / imgname
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        if "image" in item and item["image"] is not None:
            img = item["image"]
            if isinstance(img, bytes):
                return base64.b64encode(img).decode("utf-8")
            elif isinstance(img, Image.Image):
                return self.encode_image(img)
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

    def _to_float(self, text: str) -> Optional[float]:
        """
        Convert string to float, handling percentages.

        Following VLMEvalKit: percentages are converted to decimals (5% -> 0.05).
        """
        text = str(text).strip()
        try:
            # Remove commas and dollar signs
            text = text.replace(",", "").replace("$", "")
            if text.endswith("%"):
                # Convert percentage to decimal (VLMEvalKit behavior)
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    def score_relaxed(self, prediction: str, answer: str) -> bool:
        """
        Calculate relaxed correctness following VLMEvalKit.

        For numeric answers: allows 5% relative tolerance.
        For non-numeric answers: exact match (case-insensitive).

        Reference: https://arxiv.org/pdf/2203.10244.pdf, section 5.1
        """
        pred = str(prediction).strip()
        ans = str(answer).strip()

        relaxed_tolerance = getattr(self, "relaxed_tolerance", 0.05)

        pred_float = self._to_float(pred)
        ans_float = self._to_float(ans)

        if pred_float is not None and ans_float is not None:
            if ans_float == 0:
                return abs(pred_float) < 1e-6
            relative_change = abs(pred_float - ans_float) / abs(ans_float)
            return relative_change <= relaxed_tolerance

        # Non-numeric: exact match (case-insensitive)
        return pred.lower() == ans.lower()

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
            if hasattr(message, "reasoning") and message.reasoning and not response:
                response = message.reasoning
            if not response and hasattr(message, "model_extra"):
                reasoning = message.model_extra.get("reasoning", "")
                if reasoning:
                    response = reasoning

            if not response:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            extracted = self.extract_answer(response)
            answer = data_item.get("label", data_item.get("answer", ""))
            correct = self.score_relaxed(extracted, answer)

            sample = {
                "question": data_item.get("query", data_item.get("question", "")),
                "answer": answer,
                "prediction": extracted,
                "correct": correct,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            ChartQA(
                subset="human", relaxed_tolerance=0.05, temperature=0.0, max_tokens=2048
            )
        )
    )
