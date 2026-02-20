"""VLMBlind (VLMs are Blind) evaluation environment."""

import asyncio
import base64
import io
import re
from typing import List, Optional, Tuple

from datasets import load_dataset
from environments.eval_environments.eval import EvalBase, eval_runner
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager


class VLMBlind(EvalBase):
    """VLMBlind evaluation - tests basic visual perception abilities of VLMs."""

    TASK_PATTERNS = {
        "Subway Connections": r"\{([^}]+)\}",
        "Nested Squares": r"\{([^}]+)\}",
        "Line Plot Intersections": r"\{([^}]+)\}",
        "Touching Circles": None,  # Substring match
        "Counting Grid": r"(\d+)\s*(?:rows?|r).*?(\d+)\s*(?:columns?|cols?|c)|(\d+)\s*[xX×]\s*(\d+)",
        "Olympic Counting": None,  # Substring match
        "Circled Letter": r"\{([^}]+)\}",
    }

    def setup_data(self) -> list:
        # XAI/vlmsareblind only has 'valid' split
        split = getattr(self, "split", "valid")

        try:
            dataset = load_dataset("XAI/vlmsareblind", split=split)
            print(f"Loaded {len(dataset)} examples from VLMBlind ({split})")
            return list(dataset)
        except Exception as e:
            print(f"Warning: Could not load VLMBlind: {e}")
            try:
                # Try valid split explicitly
                dataset = load_dataset("XAI/vlmsareblind", split="valid")
                print(f"Loaded {len(dataset)} examples from VLMBlind (valid)")
                return list(dataset)
            except Exception:
                raise ValueError(f"Could not load VLMBlind dataset: {e}")

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
        # XAI/vlmsareblind uses 'prompt' instead of 'question'
        question = item.get("prompt", item.get("question", ""))

        prompt = f"{question}\n\nProvide your answer."

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

    def extract_and_score(
        self, response: str, answer: str, task: str
    ) -> Tuple[bool, str]:
        """Task-specific answer extraction and scoring."""
        response_lower = response.lower().strip()
        answer_lower = str(answer).lower().strip()

        if task in [
            "Subway Connections",
            "Nested Squares",
            "Line Plot Intersections",
            "Circled Letter",
        ]:
            match = re.search(r"\{([^}]+)\}", response)
            if match:
                extracted = match.group(1).strip().lower()
                return extracted == answer_lower, extracted
            return answer_lower in response_lower, response_lower[:50]

        elif task == "Touching Circles":
            return answer_lower in response_lower, response_lower[:50]

        elif "Counting Grid" in task or "Grid" in task:
            patterns = [
                r"(\d+)\s*[xX×]\s*(\d+)",
                r"(\d+)\s*(?:rows?|r).*?(\d+)\s*(?:columns?|cols?|c)",
                r"(\d+)\s*(?:columns?|cols?|c).*?(\d+)\s*(?:rows?|r)",
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    groups = match.groups()
                    extracted = f"{groups[0]}x{groups[1]}"
                    ans_match = re.search(r"(\d+)\s*[xX×,]\s*(\d+)", answer)
                    if ans_match:
                        answer_parsed = f"{ans_match.group(1)}x{ans_match.group(2)}"
                        return extracted == answer_parsed, extracted
            return answer_lower in response_lower, response_lower[:50]

        elif "Olympic" in task or "Counting" in task:
            return answer_lower in response_lower, response_lower[:50]

        else:
            return answer_lower in response_lower, response_lower[:50]

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

            # XAI/vlmsareblind uses 'groundtruth' instead of 'answer'
            answer = data_item.get("groundtruth", data_item.get("answer", ""))
            task = data_item.get("task", data_item.get("category", ""))

            correct, extracted = self.extract_and_score(response, answer, task)

            sample = {
                "id": data_item.get("index", data_item.get("id", "")),
                "question": data_item.get("prompt", data_item.get("question", ""))[
                    :200
                ],
                "task": task,
                "answer": answer,
                "prediction": extracted,
                "raw_response": response[:500],
                "correct": correct,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(eval_runner(VLMBlind(split="test", temperature=0.0, max_tokens=512)))
