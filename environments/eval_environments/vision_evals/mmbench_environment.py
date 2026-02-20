"""MMBench evaluation environment."""

import asyncio
import base64
import io
from string import ascii_uppercase
from typing import List, Optional, Tuple

from datasets import load_dataset
from environments.eval_environments.eval import EvalBase, eval_runner
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.eval_environments.eval_helpers import (
    extract_letter_from_answer_tag,
    extract_mcqa_answer_with_fallback,
)


class MMBench(EvalBase):
    """MMBench evaluation - comprehensive multimodal benchmark."""

    def setup_data(self) -> list:
        split = getattr(self, "split", "dev")
        lang = getattr(self, "lang", "en")  # en, cn, cc
        version = getattr(self, "version", "v1.1")  # v1.0 or v1.1

        try:
            dataset = load_dataset("lmms-lab/MMBench", lang, split=split)
            print(f"Loaded {len(dataset)} examples from MMBench ({split}, {lang})")
            return list(dataset)
        except Exception as e:
            print(f"Warning: Could not load from lmms-lab: {e}")
            try:
                dataset = load_dataset("lmms-lab/MMBench_EN", split=split)
                print(f"Loaded {len(dataset)} examples from MMBench ({split})")
                return list(dataset)
            except Exception:
                raise ValueError(f"Could not load MMBench dataset: {e}")

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
        hint = item.get("hint", "")

        options = {}
        for letter in ascii_uppercase:
            if letter in item and item[letter] is not None:
                val = item[letter]
                if isinstance(val, str) and val.strip():
                    options[letter] = val
                elif not isinstance(val, float):
                    options[letter] = str(val)

        prompt = ""
        if hint and str(hint).strip() and str(hint).lower() != "nan":
            prompt += f"Hint: {hint}\n"
        prompt += f"Question: {question}\n"

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

            answer = data_item.get("answer", "")

            num_choices = 0
            for letter in ascii_uppercase:
                if letter in data_item and data_item[letter] is not None:
                    val = data_item[letter]
                    if isinstance(val, str) and val.strip():
                        num_choices += 1
                    elif not isinstance(val, float):
                        num_choices += 1
            num_choices = max(num_choices, 4)

            extracted, method = self.extract_answer(response, num_choices)

            correct = False
            if extracted and answer:
                correct = extracted.upper() == str(answer).upper()

            sample = {
                "id": data_item.get("index", data_item.get("id", "")),
                "question": data_item.get("question", "")[:200],
                "category": data_item.get("category", data_item.get("l2-category", "")),
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
        eval_runner(
            MMBench(
                split="dev", lang="en", version="v1.1", temperature=0.0, max_tokens=256
            )
        )
    )
