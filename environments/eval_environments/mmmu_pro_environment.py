"""MMMU-Pro evaluation environment."""

import asyncio
import base64
import io
import re
from string import ascii_uppercase
from typing import List, Optional, Tuple

from datasets import load_dataset
from openai import AsyncOpenAI
from PIL import Image

from environments.eval_environments.eval_base import EvalBase, eval_runner
from environments.eval_environments.eval_helpers import (
    extract_letter_from_answer_tag,
    extract_mcqa_answer_with_fallback,
)


class MMMUPro(EvalBase):
    """MMMU-Pro evaluation - harder version of MMMU with 10 choices."""

    def setup_data(self) -> list:
        split = getattr(self, "split", "test")
        variant = getattr(self, "variant", "standard")  # standard, vision, standard_4

        config_map = {
            "standard": "standard (10 options)",
            "standard_4": "standard (4 options)",
            "vision": "vision",
        }
        config = config_map.get(variant, "standard (10 options)")

        try:
            dataset = load_dataset("MMMU/MMMU_Pro", config, split=split)
            print(f"Loaded {len(dataset)} examples from MMMU-Pro ({split}, {config})")
            return list(dataset)
        except Exception as e:
            print(f"Error loading MMMU-Pro: {e}")
            try:
                dataset = load_dataset(
                    "MMMU/MMMU_Pro", "standard (10 options)", split="test"
                )
                print(f"Loaded {len(dataset)} examples from MMMU-Pro (test)")
                return list(dataset)
            except Exception:
                raise ValueError(f"Could not load MMMU-Pro dataset: {e}")

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_images(self, item: dict) -> List[str]:
        """Extract all images from the item."""
        images = []
        for i in range(1, 8):
            key = f"image_{i}"
            if key in item and item[key] is not None:
                if isinstance(item[key], Image.Image):
                    images.append(self.encode_image(item[key]))
        if "image" in item and item["image"] is not None:
            if isinstance(item["image"], Image.Image):
                images.append(self.encode_image(item["image"]))
        return images

    def build_messages(self, item: dict) -> List[dict]:
        images = self.get_images(item)
        question = item.get("question", "")
        options = item.get("options", [])

        if isinstance(options, str):
            try:
                options = eval(options)
            except Exception:
                options = []

        variant = getattr(self, "variant", "standard")

        if variant == "vision":
            prompt = "Answer the following multiple-choice question in the image. Answer directly with the option letter from the given choices."
        else:
            if options:
                options_text = "\n".join(
                    [f"{ascii_uppercase[i]}. {opt}" for i, opt in enumerate(options)]
                )
                prompt = f"Question: {question}\n\nOptions:\n{options_text}\n\n"

                if variant == "cot":
                    prompt += (
                        "Answer the following multiple-choice question. "
                        "The last line of your response should be of the following format: "
                        "'Answer: $LETTER' (without quotes) where LETTER is one of the options. "
                        "Think step by step before answering."
                    )
                else:
                    prompt += (
                        "Answer directly with the option letter from the given choices."
                    )
            else:
                prompt = f"Question: {question}\n\nProvide your answer."

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

    def extract_answer_cot(self, response: str) -> Optional[str]:
        """Extract answer from COT response format 'Answer: X'."""
        lines = response.strip().split("\n")
        lines = [x.strip() for x in lines]

        for line in reversed(lines):
            if line.startswith("Answer:"):
                rest = line[7:].strip()
                from collections import Counter

                letter_counts = Counter(
                    ch for ch in rest.upper() if ch in ascii_uppercase[:10]
                )
                if len(letter_counts) == 1:
                    return list(letter_counts.keys())[0]
                elif letter_counts:
                    for ch in rest.upper():
                        if ch in ascii_uppercase[:10]:
                            return ch
        return None

    def extract_answer(
        self, response: str, num_choices: int
    ) -> Tuple[Optional[str], str]:
        """Extract answer letter from response."""
        variant = getattr(self, "variant", "standard")

        if variant == "cot":
            cot_answer = self.extract_answer_cot(response)
            if cot_answer:
                return cot_answer, "cot_extraction"

        valid_letters = set(ascii_uppercase[:num_choices])

        letter, method = extract_letter_from_answer_tag(response, valid_letters)
        if letter:
            return letter, method

        letter, method = extract_mcqa_answer_with_fallback(response, num_choices)
        return letter, method

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

            answer = data_item.get("answer", "")
            options = data_item.get("options", [])
            if isinstance(options, str):
                try:
                    options = eval(options)
                except Exception:
                    options = []

            num_choices = len(options) if options else 10
            extracted, method = self.extract_answer(response, num_choices)

            correct = False
            if extracted and answer:
                correct = extracted.upper() == answer.upper()

            sample = {
                "id": data_item.get("id", ""),
                "question": data_item.get("question", "")[:200],
                "subject": data_item.get("subject", ""),
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
            MMMUPro,
            split="test",
            variant="standard",
            temperature=0.0,
            max_tokens=1024,
        )
    )
