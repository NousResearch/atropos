import asyncio
import base64
import io
import re
from typing import List, Tuple

from datasets import load_dataset
from openai import AsyncOpenAI
from PIL import Image

from environments.eval_environments.eval_base import EvalBase, eval_runner


class MathVista(EvalBase):
    """
    MathVista evaluation environment.

    A benchmark for mathematical reasoning in visual contexts.
    https://mathvista.github.io/
    """

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

            answer_type = data_item.get("answer_type", "text")
            question_type = data_item.get("question_type", "free_form")
            precision = data_item.get("precision", 0)

            extracted = self.extract_answer(response, answer_type, question_type)
            answer = data_item.get("answer", "")
            correct = self.score(extracted, answer, answer_type, precision)

            sample = {
                "pid": data_item.get("pid", ""),
                "question": data_item.get("question", ""),
                "answer": answer,
                "prediction": extracted,
                "correct": correct,
                "question_type": question_type,
                "answer_type": answer_type,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            MathVista,
            split="testmini",
            use_query=True,
            temperature=0.0,
            max_tokens=4096,
        )
    )
