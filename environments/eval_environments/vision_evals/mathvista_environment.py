"""MathVista evaluation environment."""

import asyncio
import base64
import io
import os
import re
from typing import Dict, List, Optional, Tuple

import openai
from datasets import load_dataset
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.eval_environments.eval import EvalBase, eval_runner

ICL_EXAMPLES = [
    """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.
Question: Which number is missing?
Model response: The number missing in the sequence is 14.
Extracted answer: 14
""",
    """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.
Extracted answer: 0.6
""",
    """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.
Extracted answer: 1.45
""",
    """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line graph saw its maximum peak?
Model response: The line graph saw its maximum peak between 2007 and 2008.
Extracted answer: [2007, 2008]
""",
    """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5
Model response: The correct answer is (B) 8/11.
Extracted answer: B
""",
]


def build_extraction_prompt(question: str, prediction: str) -> str:
    task_description = """Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.
"""
    prompt = task_description
    for example in ICL_EXAMPLES:
        prompt += example + "\n"
    prompt += question + "\n"
    prompt += "Model response: " + prediction + "\n"
    prompt += "Extracted answer:"
    return prompt


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


class MathVista(EvalBase):
    TASK_TYPES = ["FQA", "GPS", "MWP", "TQA", "VQA"]
    SKILL_TYPES = ["ALG", "ARI", "GEO", "LOG", "NUM", "SCI", "STA"]

    def setup_data(self) -> list:
        split = getattr(self, "split", "testmini")
        dataset = load_dataset("AI4Math/MathVista", split=split)
        print(f"Loaded {len(dataset)} examples from MathVista ({split})")
        return list(dataset)

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        if "decoded_image" in item and item["decoded_image"] is not None:
            return self.encode_image(item["decoded_image"])
        if "image" in item and item["image"] is not None:
            if isinstance(item["image"], Image.Image):
                return self.encode_image(item["image"])
        raise ValueError(f"Could not find image for item {item.get('pid', 'unknown')}")

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)

        use_query = getattr(self, "use_query", True)
        if use_query and "query" in item:
            prompt = item["query"]
        else:
            prompt = self._build_custom_prompt(item)

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

    def _build_custom_prompt(self, item: dict) -> str:
        question = item.get("question", "")
        question_type = item.get("question_type", "free_form")
        answer_type = item.get("answer_type", "text")
        precision = item.get("precision", 2)

        if question_type == "multi_choice":
            choices = item.get("choices", [])
            choices_text = "\n".join(choices) if choices else ""
            hint = (
                "Please answer the question and provide the correct option letter, "
                "e.g., A, B, C, D, at the end."
            )
            return f"Hint: {hint}\nQuestion: {question}\nChoices:\n{choices_text}"

        if answer_type == "integer":
            hint = (
                "Please answer the question requiring an integer answer "
                "and provide the final value, e.g., 1, 2, 3, at the end."
            )
        elif answer_type == "float":
            hint = (
                f"Please answer the question requiring a floating-point number "
                f"with {precision} decimal place(s) and provide the final value at the end."
            )
        elif answer_type == "list":
            hint = (
                "Please answer the question requiring a Python list as an answer "
                "and provide the final list, e.g., [1, 2, 3], at the end."
            )
        else:
            hint = "Please answer the question and provide the final answer at the end."

        return f"Hint: {hint}\nQuestion: {question}"

    def _prefetch_answer(self, response: str, item: dict) -> Tuple[Optional[str], bool]:
        question_type = item.get("question_type", "free_form")
        answer_type = item.get("answer_type", "text")

        if question_type == "multi_choice":
            choices_list = item.get("choices", [])
            if choices_list:
                choices = {chr(65 + i): val for i, val in enumerate(choices_list)}
                result = can_infer(response, choices)
                if result:
                    return result, True

            # Fallback: find last letter
            for char in reversed(response.upper()):
                if char in "ABCDEFGH":
                    return char, True
            return None, False

        response = response.strip()

        if answer_type == "integer":
            numbers = re.findall(r"-?\d+", response)
            if numbers:
                return numbers[-1], True

        elif answer_type == "float":
            numbers = re.findall(r"-?\d+\.?\d*", response)
            if numbers:
                return numbers[-1], True

        elif answer_type == "list":
            match = re.search(r"\[[\d\.,\s-]+\]", response)
            if match:
                return match.group(0), True

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

            prompt = build_extraction_prompt(question, response)

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

    def extract_answer(
        self, response: str, answer_type: str, question_type: str
    ) -> str:
        response = response.strip()

        if question_type == "multi_choice":
            for char in reversed(response.upper()):
                if char in "ABCDEFGH":
                    return char
            return ""

        if answer_type == "integer":
            numbers = re.findall(r"-?\d+", response)
            return numbers[-1] if numbers else ""

        if answer_type == "float":
            numbers = re.findall(r"-?\d+\.?\d*", response)
            return numbers[-1] if numbers else ""

        if answer_type == "list":
            match = re.search(r"\[[\d\.,\s-]+\]", response)
            return match.group(0) if match else ""

        return response

    def score(
        self, prediction: str, answer: str, answer_type: str, precision: int = 0
    ) -> bool:
        pred = prediction.strip()
        ans = answer.strip()

        if not pred:
            return False

        if answer_type == "text":
            return pred.upper() == ans.upper()

        if answer_type == "integer":
            try:
                return int(float(pred)) == int(float(ans))
            except (ValueError, OverflowError):
                return False

        if answer_type == "float":
            try:
                tolerance = 10 ** (-precision) if precision > 0 else 0.01
                return abs(float(pred) - float(ans)) < tolerance
            except ValueError:
                return False

        if answer_type == "list":
            try:
                pred_list = eval(pred)
                ans_list = eval(ans)
                return pred_list == ans_list
            except Exception:
                return False

        return pred.lower() == ans.lower()

    async def run_item(self, server: ServerManager, data_item: dict) -> Tuple[dict, dict]:
        try:
            messages = self.build_messages(data_item)

            completion = await self.chat_completion(server, messages)

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

            answer_type = data_item.get("answer_type", "text")
            question_type = data_item.get("question_type", "free_form")
            precision = data_item.get("precision", 0)

            use_gpt_extraction = getattr(self, "use_gpt_extraction", True)
            prefetch_result, prefetch_success = self._prefetch_answer(
                response, data_item
            )

            if prefetch_success and prefetch_result:
                extracted = prefetch_result
                extraction_method = "prefetch"
            elif use_gpt_extraction:
                question = data_item.get("query", data_item.get("question", ""))
                gpt_result = await self._extract_with_gpt(question, response)
                if gpt_result:
                    extracted = gpt_result
                    extraction_method = "gpt"
                else:
                    extracted = self.extract_answer(
                        response, answer_type, question_type
                    )
                    extraction_method = "regex_fallback"
            else:
                extracted = self.extract_answer(response, answer_type, question_type)
                extraction_method = "regex"

            answer = data_item.get("answer", "")
            correct = self.score(extracted, answer, answer_type, precision)

            sample = {
                "pid": data_item.get("pid", ""),
                "question": data_item.get("question", ""),
                "answer": answer,
                "prediction": extracted,
                "raw_response": response[:500],
                "correct": correct,
                "question_type": question_type,
                "answer_type": answer_type,
                "extraction_method": extraction_method,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            MathVista(
                split="testmini",
                use_query=True,
                use_gpt_extraction=True,
                judge_model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=4096,
            )
        )
    )
