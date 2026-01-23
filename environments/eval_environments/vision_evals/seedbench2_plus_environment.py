"""SEED-Bench2-Plus evaluation environment."""

import asyncio
import base64
import io
from string import ascii_uppercase
from typing import List, Optional, Tuple

from datasets import load_dataset
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.eval_environments.eval import EvalBase, eval_runner
from environments.eval_environments.eval_helpers import (
    extract_letter_from_answer_tag,
    extract_mcqa_answer_with_fallback,
)


class SEEDBench2Plus(EvalBase):
    """SEED-Bench2-Plus evaluation - comprehensive visual understanding benchmark."""

    def setup_data(self) -> list:
        split = getattr(self, "split", "test")
        max_samples = getattr(self, "max_samples", None)

        try:
            # Use streaming to avoid memory issues with this large dataset
            dataset = load_dataset("lmms-lab/SEED-Bench-2", split=split, streaming=True)

            # Take samples from streaming dataset
            if max_samples:
                data = list(dataset.take(max_samples))
            else:
                # Default to 1000 samples to avoid loading entire 24k dataset
                data = list(dataset.take(1000))

            print(f"Loaded {len(data)} examples from SEED-Bench2 ({split}, streaming)")
            return data
        except Exception as e:
            print(f"Warning: Could not load SEED-Bench2: {e}")
            try:
                dataset = load_dataset(
                    "lmms-lab/SEED-Bench", split=split, streaming=True
                )
                if max_samples:
                    data = list(dataset.take(max_samples))
                else:
                    data = list(dataset.take(1000))
                print(
                    f"Loaded {len(data)} examples from SEED-Bench ({split}, streaming)"
                )
                return data
            except Exception:
                raise ValueError(f"Could not load SEED-Bench2-Plus dataset: {e}")

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> Optional[str]:
        for key in ["image", "decoded_image"]:
            if key in item and item[key] is not None:
                val = item[key]
                if isinstance(val, Image.Image):
                    return self.encode_image(val)
                elif isinstance(val, list) and len(val) > 0:
                    # SEED-Bench-2 stores images as a list of PIL images
                    if isinstance(val[0], Image.Image):
                        return self.encode_image(val[0])
        return None

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")

        options = {}
        for letter in ascii_uppercase[:6]:
            # Check for choice_a, choice_b format
            choice_key = f"choice_{letter.lower()}"
            if choice_key in item and item[choice_key] is not None:
                val = item[choice_key]
                if isinstance(val, str) and val.strip():
                    options[letter] = val
            elif letter in item and item[letter] is not None:
                val = item[letter]
                if isinstance(val, str) and val.strip():
                    options[letter] = val

        if not options:
            choices = item.get("choices", [])
            if isinstance(choices, str):
                try:
                    choices = eval(choices)
                except Exception:
                    choices = []
            for i, choice in enumerate(choices):
                options[ascii_uppercase[i]] = choice

        prompt = f"Question: {question}\n"
        if options:
            prompt += "Options:\n"
            for letter in sorted(options.keys()):
                prompt += f"{letter}. {options[letter]}\n"
            prompt += "\nPlease select the correct answer from the options above."

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

    def extract_answer(
        self, response: str, num_choices: int
    ) -> Tuple[Optional[str], str]:
        valid_letters = set(ascii_uppercase[:num_choices])

        letter, method = extract_letter_from_answer_tag(response, valid_letters)
        if letter:
            return letter, method

        letter, method = extract_mcqa_answer_with_fallback(response, num_choices)
        return letter, method

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

            choices = data_item.get("choices", [])
            if isinstance(choices, str):
                try:
                    choices = eval(choices)
                except Exception:
                    choices = []

            num_choices = len(choices) if choices else 4
            if num_choices == 0:
                num_choices = sum(
                    1
                    for letter in ascii_uppercase[:6]
                    if letter in data_item and data_item[letter] is not None
                )
                num_choices = max(num_choices, 4)

            extracted, method = self.extract_answer(response, num_choices)

            correct = False
            if extracted and answer:
                correct = extracted.upper() == str(answer).upper()

            sample = {
                "id": data_item.get("index", data_item.get("question_id", "")),
                "question": data_item.get("question", "")[:200],
                "category": data_item.get(
                    "question_type_id", data_item.get("category", "")
                ),
                "answer": answer,
                "prediction": extracted,
                "raw_response": response[:500],
                "correct": correct,
                "extraction_method": method,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(SEEDBench2Plus(split="test", temperature=0.0, max_tokens=256))
    )
