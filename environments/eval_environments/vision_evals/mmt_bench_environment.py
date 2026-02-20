"""MMT-Bench evaluation environment."""

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


class MMTBench(EvalBase):
    """MMT-Bench evaluation - multi-task multimodal benchmark."""

    def setup_data(self) -> list:
        split = getattr(self, "split", "train")
        max_samples = getattr(self, "max_samples", None)  # None = use all samples

        try:
            # Try full dataset download first
            dataset = load_dataset("OpenGVLab/MMT-Bench", split=split)
            data = list(dataset)
            if max_samples:
                data = data[:max_samples]
            print(f"Loaded {len(data)} examples from MMT-Bench ({split})")
            return data
        except Exception as e:
            print(f"Warning: Full download failed, using streaming: {e}")
            # Fallback to streaming if full download fails (known column mismatch issue)
            try:
                dataset = load_dataset(
                    "OpenGVLab/MMT-Bench", split=split, streaming=True
                )
                if max_samples:
                    data = list(dataset.take(max_samples))
                else:
                    # Stream all available samples
                    data = []
                    for i, item in enumerate(dataset):
                        data.append(item)
                        if i % 5000 == 0 and i > 0:
                            print(f"  Streamed {i} samples...")
                print(
                    f"Loaded {len(data)} examples from MMT-Bench ({split}, streaming)"
                )
                return data
            except Exception:
                raise ValueError(f"Could not load MMT-Bench dataset: {e}")

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
                elif isinstance(val, str) and len(val) > 100:
                    # Already base64-encoded string
                    return val
        return None

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")
        hint = item.get("hint", "")

        options = {}
        for letter in ascii_uppercase[:8]:  # Support up to 8 options
            if letter in item and item[letter] is not None:
                val = item[letter]
                if isinstance(val, str) and val.strip():
                    options[letter] = val

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

            num_choices = sum(
                1
                for letter in ascii_uppercase[:8]
                if letter in data_item
                and data_item[letter] is not None
                and isinstance(data_item[letter], str)
                and data_item[letter].strip()
            )
            num_choices = max(num_choices, 4)

            extracted, method = self.extract_answer(response, num_choices)

            correct = False
            if extracted and answer:
                correct = extracted.upper() == str(answer).upper()

            sample = {
                "id": data_item.get("index", data_item.get("id", "")),
                "question": data_item.get("question", "")[:200],
                "task": data_item.get("task", ""),
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
    asyncio.run(eval_runner(MMTBench(split="val", temperature=0.0, max_tokens=256)))
