"""DynaMath evaluation environment."""

import asyncio
import base64
import io
import json
import re
from string import ascii_uppercase
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.eval_environments.eval import EvalBase, eval_runner


class DynaMath(EvalBase):
    """DynaMath evaluation - dynamic mathematical reasoning benchmark."""

    GUIDE = """
## Answer Instruction
Please provide an answer to the question outlined above. Your response should adhere to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. {INST}

Example of expected JSON response format:

{{
    "solution": "[Detailed step-by-step explanation]",
    "short answer": "[Concise Answer]"
}}
"""

    def setup_data(self) -> list:
        # DynaMath_Sample uses variant splits: sample_variant1, sample_variant2, etc.
        split = getattr(self, "split", "sample_variant1")

        try:
            # DynaMath_Sample is the publicly available dataset
            dataset = load_dataset("DynaMath/DynaMath_Sample", split=split)
            print(f"Loaded {len(dataset)} examples from DynaMath ({split})")
            return list(dataset)
        except Exception as e:
            print(f"Warning: Could not load DynaMath: {e}")
            try:
                # Try sample_variant1 explicitly
                dataset = load_dataset(
                    "DynaMath/DynaMath_Sample", split="sample_variant1"
                )
                print(f"Loaded {len(dataset)} examples from DynaMath (sample_variant1)")
                return list(dataset)
            except Exception:
                try:
                    dataset = load_dataset("lmms-lab/DynaMath", split="test")
                    print(f"Loaded {len(dataset)} examples from DynaMath (test)")
                    return list(dataset)
                except Exception:
                    raise ValueError(f"Could not load DynaMath dataset: {e}")

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
        answer_type = item.get("answer_type", "free_form")

        use_json_format = getattr(self, "use_json_format", True)

        if use_json_format:
            if answer_type == "multiple choice":
                inst = "Provide the corresponding choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
            elif answer_type == "float":
                inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
            else:
                inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."

            prompt = f"## Question\n{question}" + self.GUIDE.format(INST=inst)
        else:
            prompt = question

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

    def preprocess_response(self, response: str) -> str:
        """Preprocess response to extract JSON."""
        response = str(response)
        if 0 <= response.find("{") < response.rfind("}"):
            response = response[response.find("{") : response.rfind("}") + 1]
        response = response.replace("\\", "").replace("\\n", "\n")
        return response

    def transfer_pi(self, value: str) -> float:
        """Convert pi symbol to numeric value."""
        if "\u03c0" in value:
            parts = value.split("\u03c0")
            return float(parts[0]) * np.pi
        return float(value)

    def parse_answer(self, answer: str, answer_type: str) -> Tuple[bool, Optional[str]]:
        """Parse answer based on type."""
        if answer_type == "float":
            if answer.isdigit():
                return True, str(float(answer))
            parts = answer.split(" ")
            answer = parts[0]
            try:
                result = self.transfer_pi(answer)
                return True, str(result)
            except Exception:
                return False, None

        elif answer_type == "multiple choice":
            if len(answer) == 1 and answer.upper() in ascii_uppercase[:5]:
                return True, answer.upper()
            # Check if any letter appears
            for ch in ascii_uppercase[:5]:
                if ch in answer.upper():
                    return True, ch
            return False, None

        else:
            return True, answer

    def extract_answer(
        self, response: str, answer_type: str
    ) -> Tuple[bool, Optional[str]]:
        """Extract answer from response."""
        processed = self.preprocess_response(response)

        try:
            dj = json.loads(processed, strict=False)
            short_answer = dj.get("short answer")
            if short_answer is not None:
                return self.parse_answer(str(short_answer), answer_type)
        except Exception:
            pass

        if answer_type == "multiple choice":
            for ch in ascii_uppercase[:5]:
                if response.strip().upper().startswith(ch):
                    return True, ch
            for ch in ascii_uppercase[:5]:
                if ch in response.upper()[:20]:
                    return True, ch
        elif answer_type == "float":
            numbers = re.findall(r"-?\d+\.?\d*", response)
            if numbers:
                try:
                    return True, str(float(numbers[0]))
                except ValueError:
                    pass

        return False, None

    def score_answer(
        self, extracted: Optional[str], answer: str, answer_type: str, parsed: bool
    ) -> bool:
        """Score the extracted answer against ground truth."""
        if not parsed or extracted is None:
            # Check if answer appears in raw response for MC
            return False

        if answer_type == "float":
            try:
                pred_val = float(extracted)
                ans_val = float(answer)
                return abs(pred_val - ans_val) <= 0.001
            except (ValueError, TypeError):
                return False

        elif answer_type == "multiple choice":
            return extracted.upper() == str(answer).upper()

        else:
            # Free form: substring match
            return (
                extracted.lower() in answer.lower()
                or answer.lower() in extracted.lower()
            )

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

            answer = data_item.get("ground_truth", data_item.get("answer", ""))
            answer_type = data_item.get("answer_type", "free_form")

            parsed, extracted = self.extract_answer(response, answer_type)
            correct = self.score_answer(extracted, answer, answer_type, parsed)

            sample = {
                "id": data_item.get("index", data_item.get("id", "")),
                "question": data_item.get("question", "")[:200],
                "subject": data_item.get("subject", ""),
                "knowledge_level": data_item.get("knowledge_level", ""),
                "answer_type": answer_type,
                "answer": answer,
                "prediction": extracted,
                "parsed": parsed,
                "raw_response": response[:500],
                "correct": correct,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            DynaMath(split="test", use_json_format=True, temperature=0.0, max_tokens=1024)
        )
    )
