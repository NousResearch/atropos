"""AI2D (AI2 Diagrams) evaluation environment."""

import asyncio
import base64
import io
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


class AI2D(EvalBase):
    """AI2D evaluation - diagram understanding benchmark."""

    def setup_data(self) -> list:
        split = getattr(self, "split", "test")
        use_mask = getattr(self, "use_mask", True)

        try:
            dataset = load_dataset("lmms-lab/ai2d", split=split)
            print(f"Loaded {len(dataset)} examples from AI2D ({split})")
            return list(dataset)
        except Exception as e:
            print(f"Warning: Could not load AI2D: {e}")
            try:
                dataset = load_dataset("allenai/ai2_diagrams", split=split)
                print(f"Loaded {len(dataset)} examples from AI2D ({split})")
                return list(dataset)
            except Exception:
                raise ValueError(f"Could not load AI2D dataset: {e}")

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

        choices = item.get("choices", [])
        if isinstance(choices, str):
            try:
                choices = eval(choices)
            except Exception:
                choices = []

        options = {}
        if choices:
            for i, choice in enumerate(choices):
                options[ascii_uppercase[i]] = choice
        else:
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

            choices = data_item.get("choices", [])
            if isinstance(choices, str):
                try:
                    choices = eval(choices)
                except Exception:
                    choices = []

            num_choices = len(choices) if choices else 4

            extracted, method = self.extract_answer(response, num_choices)

            correct = False
            if extracted and answer:
                if str(answer).isdigit():
                    answer_letter = ascii_uppercase[int(answer)]
                else:
                    answer_letter = str(answer).upper()
                correct = extracted.upper() == answer_letter

            sample = {
                "id": data_item.get("index", data_item.get("id", "")),
                "question": data_item.get("question", "")[:200],
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
            AI2D,
            split="test",
            temperature=0.0,
            max_tokens=256,
        )
    )
