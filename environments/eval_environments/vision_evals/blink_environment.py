"""BLINK evaluation environment."""

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


class BLINK(EvalBase):
    """BLINK evaluation - visual perception benchmark."""

    def setup_data(self) -> list:
        split = getattr(self, "split", "val")
        task = getattr(self, "task", "Counting")  # One of the BLINK task categories

        try:
            dataset = load_dataset("BLINK-Benchmark/BLINK", task, split=split)
            print(f"Loaded {len(dataset)} examples from BLINK ({split}, {task})")
            return list(dataset)
        except Exception as e:
            print(f"Warning: Could not load BLINK: {e}")
            try:
                tasks = [
                    "Counting",
                    "Spatial_Relation",
                    "Object_Localization",
                    "Visual_Similarity",
                ]
                all_data = []
                for t in tasks:
                    try:
                        ds = load_dataset("BLINK-Benchmark/BLINK", t, split=split)
                        for item in ds:
                            item["task"] = t
                            all_data.append(item)
                    except Exception:
                        pass
                if all_data:
                    print(f"Loaded {len(all_data)} examples from BLINK ({split})")
                    return all_data
                raise ValueError(f"Could not load BLINK dataset: {e}")
            except Exception:
                raise ValueError(f"Could not load BLINK dataset: {e}")

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_images(self, item: dict) -> List[str]:
        """Get all images from item (BLINK can have multiple images)."""
        images = []
        for i in range(1, 5):
            key = f"image_{i}"
            if key in item and item[key] is not None:
                if isinstance(item[key], Image.Image):
                    images.append(self.encode_image(item[key]))

        if not images and "image" in item and item["image"] is not None:
            if isinstance(item["image"], Image.Image):
                images.append(self.encode_image(item["image"]))

        return images

    def build_messages(self, item: dict) -> List[dict]:
        images = self.get_images(item)
        question = item.get("question", "")

        options = {}
        for letter in ascii_uppercase[:6]:
            if letter in item and item[letter] is not None:
                val = item[letter]
                if isinstance(val, str) and val.strip():
                    options[letter] = val

        prompt = f"Question: {question}\n"
        if options:
            prompt += "Options:\n"
            for letter in sorted(options.keys()):
                prompt += f"{letter}. {options[letter]}\n"
            prompt += "\nPlease select the correct answer from the options above."

        content = []
        for img_b64 in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
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
                for letter in ascii_uppercase[:6]
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
                "category": data_item.get("category", ""),
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
    asyncio.run(eval_runner(BLINK(split="val", temperature=0.0, max_tokens=256)))
