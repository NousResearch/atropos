"""MMMU (Massive Multi-discipline Multimodal Understanding) evaluation environment."""

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


class MMMU(EvalBase):
    """MMMU evaluation - multi-discipline multimodal understanding benchmark."""

    def setup_data(self) -> list:
        split = getattr(self, "split", "validation")
        subset = getattr(self, "subset", None)

        if subset:
            dataset = load_dataset("MMMU/MMMU", subset, split=split)
        else:
            subjects = [
                "Accounting", "Agriculture", "Architecture_and_Engineering",
                "Art", "Art_Theory", "Basic_Medical_Science", "Biology",
                "Chemistry", "Clinical_Medicine", "Computer_Science",
                "Design", "Diagnostics_and_Laboratory_Medicine", "Economics",
                "Electronics", "Energy_and_Power", "Finance", "Geography",
                "History", "Literature", "Manage", "Marketing", "Materials",
                "Math", "Mechanical_Engineering", "Music", "Pharmacy",
                "Physics", "Psychology", "Public_Health", "Sociology"
            ]
            all_data = []
            for subj in subjects:
                try:
                    ds = load_dataset("MMMU/MMMU", subj, split=split)
                    for item in ds:
                        item["subject"] = subj
                        all_data.append(item)
                except Exception as e:
                    print(f"Warning: Could not load subject {subj}: {e}")
            print(f"Loaded {len(all_data)} examples from MMMU ({split})")
            return all_data

        print(f"Loaded {len(dataset)} examples from MMMU ({split}, {subset})")
        return list(dataset)

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_images(self, item: dict) -> List[str]:
        """Extract all images from the item (MMMU can have multiple images)."""
        images = []
        for i in range(1, 8):  # MMMU supports up to 7 images
            key = f"image_{i}"
            if key in item and item[key] is not None:
                if isinstance(item[key], Image.Image):
                    images.append(self.encode_image(item[key]))
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

        if options:
            options_text = "\n".join([
                f"({ascii_uppercase[i]}) {opt}" for i, opt in enumerate(options)
            ])
            prompt = f"Question: {question}\n\nOptions:\n{options_text}\n\nPlease select the correct answer from the options above."
        else:
            prompt = f"Question: {question}\n\nProvide your answer."

        content = []
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def extract_answer(self, response: str, num_choices: int) -> Tuple[Optional[str], str]:
        """Extract answer letter from response."""
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

            num_choices = len(options) if options else 4
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
            MMMU,
            split="validation",
            temperature=0.0,
            max_tokens=1024,
        )
    )
