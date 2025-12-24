import asyncio
import base64
import io
import re
from typing import List, Tuple

from datasets import load_dataset
from openai import AsyncOpenAI
from PIL import Image

from environments.eval_environments.eval_base import EvalBase, eval_runner


class WeMath(EvalBase):
    """
    We-Math evaluation environment.

    A benchmark for visual mathematical reasoning.
    https://we-math.github.io/
    """

    def setup_data(self) -> list:
        split = getattr(self, "split", "testmini")
        dataset = load_dataset("We-Math/We-Math", split=split)
        print(f"Loaded {len(dataset)} examples from We-Math ({split})")
        return list(dataset)

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        img = item.get("image_path") or item.get("image")
        if img is not None:
            if isinstance(img, Image.Image):
                return self.encode_image(img)
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

            sample = {
                "problem_id": data_item.get("problem_id", ""),
                "question": data_item.get("question", ""),
                "answer": answer,
                "prediction": extracted,
                "correct": correct,
                "step": data_item.get("step", "1step"),
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            WeMath,
            split="testmini",
            temperature=0.0,
            max_tokens=4096,
        )
    )
