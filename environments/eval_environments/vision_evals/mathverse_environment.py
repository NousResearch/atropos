"""MathVerse evaluation environment."""

import asyncio
import base64
import io
import os
import re
from typing import List, Optional, Tuple

import openai
from datasets import load_dataset
from environments.eval_environments.eval import EvalBase, eval_runner
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager

EXTRACT_ICL_EXAMPLES = [
    "1.\nModel response: 'The perimeter of the sector is approximately (-2, 1)'\n"
    "Extracted Answer: (-2, 1)\n",
    "2.\nModel response: 'The correct option is D. They give the solutions to $f(t)=g(t)$.'\n"
    "Extracted Answer: D\n",
    "3.\nModel response: 'The range is (-4, 1]. Domain: (-3, 3], Range: (-4, 1]'\n"
    "Extracted Answer: Domain: (-3, 3], Range: (-4, 1]\n",
    "4.\nModel response: 'I cannot provide the answer because there is not enough information.'\n"
    "Extracted Answer: null\n",
    "5.\nModel response: 'The distance d between Ned and Bart is approximately 22.3 meters.'\n"
    "Extracted answer: 22.3\n",
    "6.\nModel response: 'The equation for f is f(x) = -x^2 - 2x + 1'\n"
    "Extracted answer: f(x) = -x^2 - 2x + 1\n",
]

SCORE_ICL_EXAMPLES = [
    """[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0
""",
    """[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()
Choices:
A:2
B:2√{3}
C:√{3}
D:2√{2}
[Standard Answer]: C
[Model_answer] : B:2√{3}
Judgement: 0
""",
    """[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0
""",
    """[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()
Choices:
A:2
B:2√{3}
C:√{3}
D:2√{2}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0
""",
]


class MathVerse(EvalBase):
    PROBLEM_VERSIONS = [
        "Text Dominant",
        "Text Lite",
        "Vision Intensive",
        "Vision Dominant",
        "Vision Only",
    ]

    def setup_data(self) -> list:
        config = getattr(self, "config", "testmini")
        dataset = load_dataset("AI4Math/MathVerse", config, split="testmini")
        print(f"Loaded {len(dataset)} examples from MathVerse ({config})")
        return list(dataset)

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        if "image" in item and item["image"] is not None:
            if isinstance(item["image"], Image.Image):
                return self.encode_image(item["image"])
        raise ValueError(
            f"Could not find image for item {item.get('sample_index', 'unknown')}"
        )

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)

        use_cot = getattr(self, "use_cot", False)
        if use_cot and "query_cot" in item:
            question = item["query_cot"]
        elif "question_for_eval" in item:
            question = item["question_for_eval"]
        else:
            question = item.get("question", "")

        prompt = f"""{question}

Please solve the problem step by step and provide your final answer."""

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

    async def _extract_with_gpt(self, prediction: str) -> Optional[str]:
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

            task_description = (
                "I am providing you a response from a model to a math problem, "
                "termed 'Model Response'. You should extract the answer from the "
                "response as 'Extracted Answer'. Directly output the extracted "
                "answer with no explanation.\n\n"
            )
            prompt = task_description
            for example in EXTRACT_ICL_EXAMPLES:
                prompt += example + "\n\n"
            prompt += f"7.\nModel response: '{prediction}'\nExtracted Answer: "

            completion = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )

            result = completion.choices[0].message.content.strip()
            return result if result else None

        except Exception as e:
            print(f"GPT extraction error: {e}")
            return None

    async def _score_with_gpt(
        self, question: str, answer: str, extracted: str
    ) -> Optional[bool]:
        judge_model = getattr(self, "judge_model", "gpt-4o-mini")
        judge_base_url = getattr(self, "judge_base_url", "https://api.openai.com/v1")
        judge_api_key = os.environ.get(
            getattr(self, "judge_api_key_env", "OPENAI_API_KEY"), ""
        )

        if not judge_api_key:
            return None

        if str(extracted).strip() == str(answer).strip():
            return True

        try:
            judge_client = openai.AsyncOpenAI(
                api_key=judge_api_key,
                base_url=judge_base_url,
            )

            task_description = (
                "Below are two answers to a math question. Question is [Question], "
                "[Standard Answer] is the standard answer to the question, and "
                "[Model_answer] is the answer extracted from a model's output to "
                "this question. Determine whether these two answers are consistent.\n"
                "Please note that only when the [Model_answer] completely matches "
                "the [Standard Answer] means they are consistent. For non-MCQ "
                "questions, if the meaning is expressed in the same way, it is also "
                "considered consistent, for example, 0.5m and 50cm.\n"
                "If they are consistent, Judgement is 1; if different, Judgement is 0.\n\n"
            )
            prompt = task_description
            for example in SCORE_ICL_EXAMPLES:
                prompt += example + "\n\n"
            prompt += f"""[Question]: {question}
[Standard Answer]: {answer}
[Model_answer] : {extracted}
Judgement:"""

            completion = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=16,
            )

            result = completion.choices[0].message.content.strip()
            if result in ["0", "1"]:
                return int(result) == 1
            return None

        except Exception as e:
            print(f"GPT scoring error: {e}")
            return None

    def extract_answer_fallback(self, response: str) -> str:
        response = response.strip().upper()

        for char in reversed(response):
            if char in "ABCDE":
                return char

        numbers = re.findall(r"-?\d+\.?\d*", response)
        if numbers:
            return numbers[-1]

        return response[:100]

    def score_fallback(self, prediction: str, answer: str) -> bool:
        pred = prediction.strip().upper()
        ans = answer.strip().upper()

        if pred == ans:
            return True

        try:
            pred_num = float(pred)
            ans_num = float(ans)
            return abs(pred_num - ans_num) < 0.01
        except ValueError:
            pass

        return False

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
            if hasattr(message, "reasoning") and message.reasoning and not response:
                response = message.reasoning
            if not response and hasattr(message, "model_extra"):
                reasoning = message.model_extra.get("reasoning", "")
                if reasoning:
                    response = reasoning

            if not response:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            use_gpt_evaluation = getattr(self, "use_gpt_evaluation", True)
            answer = data_item.get("answer", "")
            question = data_item.get("question_for_eval", data_item.get("question", ""))

            if use_gpt_evaluation:
                extracted = await self._extract_with_gpt(response)
                if not extracted:
                    extracted = self.extract_answer_fallback(response)
                    extraction_method = "fallback"
                else:
                    extraction_method = "gpt"
            else:
                extracted = self.extract_answer_fallback(response)
                extraction_method = "fallback"

            if use_gpt_evaluation:
                score_result = await self._score_with_gpt(question, answer, extracted)
                if score_result is not None:
                    correct = score_result
                    scoring_method = "gpt"
                else:
                    correct = self.score_fallback(extracted, answer)
                    scoring_method = "fallback"
            else:
                correct = self.score_fallback(extracted, answer)
                scoring_method = "fallback"

            metadata = data_item.get("metadata", {})
            sample = {
                "sample_index": data_item.get("sample_index", ""),
                "problem_index": data_item.get("problem_index", ""),
                "problem_version": data_item.get("problem_version", ""),
                "question": question[:200],
                "answer": answer,
                "prediction": extracted,
                "raw_response": response[:500],
                "correct": correct,
                "subject": (
                    metadata.get("subject", "") if isinstance(metadata, dict) else ""
                ),
                "subfield": (
                    metadata.get("subfield", "") if isinstance(metadata, dict) else ""
                ),
                "extraction_method": extraction_method,
                "scoring_method": scoring_method,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            MathVerse(
                split="testmini",
                use_cot=False,
                use_gpt_evaluation=True,
                judge_model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=2048,
            )
        )
    )
