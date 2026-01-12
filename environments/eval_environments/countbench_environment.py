"""CountBench evaluation environment."""

import asyncio
import base64
import io
import re
from typing import List, Optional, Tuple

from datasets import load_dataset
from openai import AsyncOpenAI
from PIL import Image

from environments.eval_environments.eval_base import EvalBase, eval_runner


class CountBench(EvalBase):
    """CountBench evaluation - object counting benchmark."""

    def setup_data(self) -> list:
        split = getattr(self, "split", "train")  # CountBench only has train split

        try:
            dataset = load_dataset("nielsr/countbench", split=split)
            print(f"Loaded {len(dataset)} examples from CountBench ({split})")
            return list(dataset)
        except Exception as e:
            print(f"Warning: Could not load CountBench: {e}")
            try:
                # Try train split explicitly
                dataset = load_dataset("nielsr/countbench", split="train")
                print(f"Loaded {len(dataset)} examples from CountBench (train)")
                return list(dataset)
            except Exception:
                try:
                    dataset = load_dataset("google-research/countbenchqa", split="train")
                    print(f"Loaded {len(dataset)} examples from CountBench (train)")
                    return list(dataset)
                except Exception:
                    raise ValueError(f"Could not load CountBench dataset: {e}")

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

        prompt = f"{question}\n\nNote: Answer with a number directly, e.g., 3. Do not include any additional text."

        content = []
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            })
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def extract_number(self, response: str) -> Optional[str]:
        """Extract a number from the response."""
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            return numbers[0]
        return None

    def score(self, prediction: str, answer: str) -> bool:
        """Score counting answer - check if answer appears in prediction."""
        answer_str = str(answer).strip()

        if answer_str in prediction:
            return True

        extracted = self.extract_number(prediction)
        if extracted and extracted == answer_str:
            return True

        try:
            pred_num = int(self.extract_number(prediction) or prediction.strip())
            ans_num = int(answer_str)
            return pred_num == ans_num
        except (ValueError, TypeError):
            pass

        return False

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

            if not response:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            answer = data_item.get("answer", data_item.get("number", ""))

            correct = self.score(response, answer)
            extracted = self.extract_number(response)

            sample = {
                "id": data_item.get("index", data_item.get("id", "")),
                "question": data_item.get("question", "")[:200],
                "answer": answer,
                "prediction": extracted or response[:50],
                "raw_response": response[:200],
                "correct": correct,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            CountBench,
            split="test",
            temperature=0.0,
            max_tokens=64,
        )
    )
