"""POPE (Polling-based Object Probing Evaluation) evaluation environment."""

import asyncio
import base64
import io
import re
from typing import List, Optional, Tuple

from datasets import load_dataset
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.eval_environments.eval import EvalBase, eval_runner


class POPE(EvalBase):
    """POPE evaluation - object hallucination benchmark with yes/no questions."""

    def setup_data(self) -> list:
        split = getattr(self, "split", "test")
        variant = getattr(self, "variant", "random")  # random, popular, adversarial

        try:
            dataset = load_dataset("lmms-lab/POPE", split=split)
            print(f"Loaded {len(dataset)} examples from POPE ({split})")
            return list(dataset)
        except Exception as e:
            print(f"Warning: Could not load POPE: {e}")
            try:
                dataset = load_dataset("OpenGVLab/POPE", split=split)
                print(f"Loaded {len(dataset)} examples from POPE ({split})")
                return list(dataset)
            except Exception:
                raise ValueError(f"Could not load POPE dataset: {e}")

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

        prompt = f"{question}\n\nPlease answer yes or no."

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

    def extract_yorn(self, response: str) -> str:
        """Extract Yes/No from response."""
        response_lower = response.lower().strip()

        if response_lower.startswith("yes"):
            return "Yes"
        if response_lower.startswith("no"):
            return "No"

        yes_patterns = [r"\byes\b", r"\btrue\b", r"\bcorrect\b", r"\baffirmative\b"]
        no_patterns = [r"\bno\b", r"\bfalse\b", r"\bincorrect\b", r"\bnegative\b"]

        for pattern in yes_patterns:
            if re.search(pattern, response_lower):
                return "Yes"

        for pattern in no_patterns:
            if re.search(pattern, response_lower):
                return "No"

        return "Unknown"

    async def run_item(self, server: ServerManager, data_item: dict) -> Tuple[dict, dict]:
        try:
            messages = self.build_messages(data_item)
            completion = await self.chat_completion(server, messages)

            if not completion.choices:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            message = completion.choices[0].message
            response = message.content or ""

            if not response:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            answer = data_item.get("answer", "")
            extracted = self.extract_yorn(response)

            answer_norm = answer.strip().lower()
            if answer_norm in ["yes", "true", "1"]:
                answer_norm = "Yes"
            elif answer_norm in ["no", "false", "0"]:
                answer_norm = "No"
            else:
                answer_norm = answer.strip()

            correct = extracted == answer_norm

            sample = {
                "id": data_item.get("index", data_item.get("question_id", "")),
                "question": data_item.get("question", "")[:200],
                "category": data_item.get("category", ""),
                "answer": answer_norm,
                "prediction": extracted,
                "raw_response": response[:200],
                "correct": correct,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(eval_runner(POPE(split="test", temperature=0.0, max_tokens=64)))
