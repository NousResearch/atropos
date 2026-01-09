"""MathVision evaluation environment."""

import asyncio
import base64
import io
import os
import re
from typing import Dict, List, Optional, Tuple

import openai
from datasets import load_dataset
from openai import AsyncOpenAI
from PIL import Image

from environments.eval_environments.eval_base import EvalBase, eval_runner

ICL_EXAMPLES = [
    """Hint: Please answer the question and provide the final answer at the end.
Question: Which number is missing?
Model response: The number missing in the sequence is 14.
Extracted answer: 14
""",
    "Hint: Please answer the question and provide the final answer at the end.\n"
    "Question: What is the fraction of females facing the camera?\n"
    "Model response: The fraction of females facing the camera is 0.6.\n"
    "Extracted answer: 0.6\n",
    """Hint: Please answer the question and provide the final answer at the end.
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.
Extracted answer: 1.45
""",
    """Hint: Please answer the question and provide the final answer at the end.
Question: Between which two years does the line graph saw its maximum peak?
Model response: The line graph saw its maximum peak between 2007 and 2008.
Extracted answer: [2007, 2008]
""",
    """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5
Model response: The correct answer is (B) 8/11.
Extracted answer: B
""",
]


def can_infer_option(answer: str, choices: Dict[str, str]) -> Optional[str]:
    if "Failed to obtain answer via API" in answer:
        return None

    answer_mod = answer
    for c in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(c, " ")

    splits = [x.strip() for x in answer_mod.split()]
    count = sum(1 for ch in choices if ch in splits)

    if count == 1:
        for ch in choices:
            if "A" in splits and len(splits) > 3:
                continue
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch

    return None


def can_infer_text(answer: str, choices: Dict[str, str]) -> Optional[str]:
    answer_lower = answer.lower()

    if len(answer_lower) > 2 * sum(len(str(v)) for v in choices.values()):
        return None

    cands = []
    for k, v in choices.items():
        if str(v).lower() in answer_lower:
            cands.append(k)

    if len(cands) == 1:
        return cands[0]

    return None


def can_infer(answer: str, choices: Dict[str, str]) -> Optional[str]:
    answer = str(answer)
    result = can_infer_option(answer, choices)
    if result:
        return result
    return can_infer_text(answer, choices)


def is_equal(asw: str, gt_asw: str) -> bool:
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()

    if gt_asw == asw:
        return True

    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(float(a) - float(b)) < 1e-6:
            return True
    except Exception:
        pass

    try:
        from latex2sympy2 import latex2sympy

        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
        if abs(float(a) - float(b)) < 1e-6:
            return True
    except Exception:
        pass

    return False


class MathVision(EvalBase):
    def setup_data(self) -> list:
        split = getattr(self, "split", "testmini")
        try:
            dataset = load_dataset("MathLLMs/MathVision", split=split)
        except Exception:
            dataset = load_dataset("MathLLMs/MathVision", "default", split=split)
        print(f"Loaded {len(dataset)} examples from MathVision ({split})")
        return list(dataset)

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        for key in ["decoded_image", "image"]:
            if key in item and item[key] is not None:
                if isinstance(item[key], Image.Image):
                    return self.encode_image(item[key])
        raise ValueError(f"Could not find image for item {item.get('id', 'unknown')}")

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")
        choices = item.get("choices", [])

        if choices:
            try:
                if isinstance(choices, str):
                    choices = eval(choices)
                choices_text = "\n".join(
                    [f"({chr(65+i)}) {c}" for i, c in enumerate(choices)]
                )
                hint = "Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
                prompt = f"Hint: {hint}\nQuestion: {question}\nChoices:\n{choices_text}"
            except Exception:
                hint = "Please answer the question and provide the final answer at the end."
                prompt = f"Hint: {hint}\nQuestion: {question}"
        else:
            hint = "Please answer the question and provide the final answer at the end."
            prompt = f"Hint: {hint}\nQuestion: {question}"

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _prefetch_answer(self, response: str, item: dict) -> Tuple[Optional[str], bool]:
        choices = item.get("choices", [])

        if choices:
            try:
                if isinstance(choices, str):
                    choices = eval(choices)
                if len(choices) > 0:
                    choices_dict = {chr(65 + i): val for i, val in enumerate(choices)}
                    result = can_infer(response, choices_dict)
                    if result:
                        return result, True
            except Exception:
                pass

        return None, False

    async def _extract_with_gpt(self, question: str, response: str) -> Optional[str]:
        judge_model = getattr(self, "judge_model", "gpt-4o-mini")
        judge_base_url = getattr(self, "judge_base_url", "https://api.openai.com/v1")
        judge_api_key = os.environ.get(
            getattr(self, "judge_api_key_env", "OPENAI_API_KEY"), ""
        )

        if not judge_api_key:
            return None

        try:
            judge_client = openai.AsyncOpenAI(
                api_key=judge_api_key,
                base_url=judge_base_url,
            )

            task_description = """Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.

"""
            prompt = task_description
            for example in ICL_EXAMPLES:
                prompt += example + "\n"
            prompt += question + "\n"
            prompt += f"Model response: {response}\n"
            prompt += "Extracted answer:"

            completion = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )

            result = completion.choices[0].message.content.strip()
            return result if result else None

        except Exception as e:
            print(f"GPT extraction error: {e}")
            return None

    def extract_answer_fallback(self, response: str) -> str:
        response = response.strip()

        for char in reversed(response.upper()):
            if char in "ABCDEFGH":
                return char

        numbers = re.findall(r"-?\d+\.?\d*", response)
        if numbers:
            return numbers[-1]

        return response[:100]

    def score(self, prediction: str, answer: str, item: dict) -> bool:
        choices = item.get("choices", [])

        if choices:
            try:
                if isinstance(choices, str):
                    choices = eval(choices)
                if len(choices) > 0:
                    choices_dict = {chr(65 + i): val for i, val in enumerate(choices)}
                    result = can_infer(prediction, choices_dict)
                    if result:
                        return result.upper() == answer.upper()
            except Exception:
                pass

        return is_equal(prediction, answer)

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
            if hasattr(message, "reasoning") and message.reasoning and not response:
                response = message.reasoning
            if not response and hasattr(message, "model_extra"):
                reasoning = message.model_extra.get("reasoning", "")
                if reasoning:
                    response = reasoning

            if not response:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            use_gpt_extraction = getattr(self, "use_gpt_extraction", True)
            answer = data_item.get("answer", "")

            prefetch_result, prefetch_success = self._prefetch_answer(
                response, data_item
            )

            if prefetch_success and prefetch_result:
                extracted = prefetch_result
                extraction_method = "prefetch"
            elif use_gpt_extraction:
                question = data_item.get("question", "")
                gpt_result = await self._extract_with_gpt(question, response)
                if gpt_result:
                    extracted = gpt_result
                    extraction_method = "gpt"
                else:
                    extracted = self.extract_answer_fallback(response)
                    extraction_method = "fallback"
            else:
                extracted = self.extract_answer_fallback(response)
                extraction_method = "fallback"

            correct = self.score(extracted, answer, data_item)

            sample = {
                "id": data_item.get("id", data_item.get("index", "")),
                "question": data_item.get("question", "")[:200],
                "answer": answer,
                "prediction": extracted,
                "raw_response": response[:500],
                "correct": correct,
                "category": data_item.get("category", ""),
                "extraction_method": extraction_method,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            MathVision,
            split="testmini",
            use_gpt_extraction=True,
            judge_model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=2048,
        )
    )
