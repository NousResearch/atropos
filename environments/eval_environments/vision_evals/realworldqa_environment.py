import asyncio
import base64
import io
from typing import List, Tuple

from datasets import load_dataset
from environments.eval_environments.eval import EvalBase, eval_runner
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager


class RealWorldQA(EvalBase):
    def setup_data(self) -> list:
        split = getattr(self, "split", "test")
        dataset = load_dataset("xai-org/RealworldQA", split=split)
        print(f"Loaded {len(dataset)} examples from RealWorldQA ({split})")
        return list(dataset)

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        if "image" in item and item["image"] is not None:
            if isinstance(item["image"], Image.Image):
                return self.encode_image(item["image"])
        raise ValueError("Could not find image for item")

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")

        prompt = f"""{question}

Provide a brief, direct answer."""

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
        lines = response.split("\n")
        if lines:
            return lines[0].strip()
        return response

    def score(self, prediction: str, answer: str) -> bool:
        pred = prediction.strip().lower()
        ans = answer.strip().lower()

        if not pred:
            return False

        if pred == ans:
            return True

        if ans in pred or pred in ans:
            return True

        pred_words = set(pred.split())
        ans_words = set(ans.split())
        overlap = pred_words & ans_words
        if len(overlap) >= len(ans_words) * 0.5:
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

            sample = {
                "question": data_item.get("question", ""),
                "answer": answer,
                "prediction": extracted,
                "correct": correct,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(eval_runner(RealWorldQA(split="test", temperature=0.0, max_tokens=256)))
