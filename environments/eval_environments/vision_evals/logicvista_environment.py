"""LogicVista evaluation environment."""

import asyncio
import base64
import io
import os
import re
from typing import Dict, List, Optional, Tuple

import openai
from datasets import load_dataset
from environments.eval_environments.eval import EvalBase, eval_runner
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager

EXTRACTION_PROMPT_TEMPLATE = """You are a information extractor that extracts multiple choice letter answer choices \
from a paragraph that contains the answer choice and sometimes explaination of why that \
choice is correct to the given question.
What letter did the following answer choose? If the answer did not select a letter answer choice, \
first try to infer the answer based off the given choices.
If it does not correspond to an answer choice OR there is no selected answer, respond with Z.
Make sure you answer with ONLY the letters chosen.
Example 1:
Question: <start>
What is the main object in image?
Options: A. teddy bear B. rabbit C. cat D. dog
<end>
Answer: <start>
a cute teddy bear
<end>
Your output: A
Example 2:
Question: <start>
What is the main object in image?
Options: A. teddy bear B. rabbit C. cat D. dog
<end>
Answer: <start>
Spider
<end>
Your output: Z
Example 3:
Question: <start>
Which figure is a rotation of the object?
<end>
Answer: <start>
The figure on the right, labeled "D," is a rotation of the object shown in the top left corner.
<end>
Your output: D
Example 4:
Question: <start>
Which of the boxes comes next in the sequence? Select from A-E
<end>
Answer: <start>
The sequence of the boxes is A, B, C, D, E.
<end>
Your output: ABCDE
Example 5:
Question: <start>
{question}
<end>
Answer: <start>
{prediction}
<end>
Your output: """


class LogicVista(EvalBase):
    SKILL_CATEGORIES = [
        "inductive",
        "deductive",
        "numerical",
        "spatial",
        "mechanical",
    ]

    CAPABILITY_CATEGORIES = [
        "diagram",
        "ocr",
        "patterns",
        "graphs",
        "tables",
        "3d shapes",
        "puzzles",
        "sequences",
        "physics",
    ]

    def setup_data(self) -> list:
        split = getattr(self, "split", "test")
        dataset = load_dataset("lscpku/LogicVista", split=split)
        print(f"Loaded {len(dataset)} examples from LogicVista ({split})")
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
            f"Could not find image for item {item.get('question_id', 'unknown')}"
        )

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")

        prompt = f"""{question}

Provide your answer as the letter(s) of the correct choice(s), e.g., A, B, C, D, or multiple letters if applicable."""

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

            prompt = EXTRACTION_PROMPT_TEMPLATE.format(
                question=question, prediction=response
            )

            completion = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )

            result = completion.choices[0].message.content.strip()

            if result and result.isupper() and result.isalpha():
                return result
            return None

        except Exception as e:
            print(f"GPT extraction error: {e}")
            return None

    def extract_answer(self, response: str) -> str:
        response = response.strip().upper()

        letters_with_sep = re.findall(r"[A-E](?:\s*[,\s]\s*[A-E])*", response)
        if letters_with_sep:
            letters = re.findall(r"[A-E]", letters_with_sep[-1])
            return "".join(sorted(set(letters)))

        letters = re.findall(
            r"[A-E]", response[-20:] if len(response) > 20 else response
        )
        if letters:
            return "".join(sorted(set(letters)))

        all_letters = re.findall(r"[A-E]", response)
        if all_letters:
            return "".join(sorted(set(all_letters[-4:])))

        return ""

    def score(self, prediction: str, answer: str) -> bool:
        if not prediction:
            return False

        answer_letters = re.findall(r"[A-Ea-e]", answer)
        answer_normalized = "".join(sorted(set(c.lower() for c in answer_letters)))

        pred_letters = [c.lower() for c in prediction if c.isalpha()]
        pred_normalized = "".join(sorted(set(pred_letters)))

        return pred_normalized == answer_normalized

    async def run_item(
        self, server: ServerManager, data_item: dict
    ) -> Tuple[dict, dict]:
        try:
            messages = self.build_messages(data_item)

            completion = await self.chat_completion(server, messages)

            if not completion.choices:
                return {"accuracy": 0.0, "hit": 0}, {"error": "Empty response"}

            message = completion.choices[0].message
            response = message.content or ""
            if hasattr(message, "reasoning") and message.reasoning and not response:
                response = message.reasoning
            if not response and hasattr(message, "model_extra"):
                reasoning = message.model_extra.get("reasoning", "")
                if reasoning:
                    response = reasoning

            if not response:
                return {"accuracy": 0.0, "hit": 0}, {"error": "Empty response"}

            use_gpt_extraction = getattr(self, "use_gpt_extraction", True)
            extracted = None
            extraction_method = "regex"

            if use_gpt_extraction:
                question = data_item.get("question", "")
                gpt_result = await self._extract_with_gpt(question, response)
                if gpt_result and gpt_result != "Z":
                    extracted = gpt_result
                    extraction_method = "gpt"

            if not extracted:
                extracted = self.extract_answer(response)
                extraction_method = "regex"

            answer = data_item.get("answer", "")
            correct = self.score(extracted, answer)

            sample = {
                "question_id": data_item.get("question_id", data_item.get("index", "")),
                "question": data_item.get("question", ""),
                "answer": answer,
                "prediction": extracted,
                "raw_response": response[:500],
                "hit": 1 if correct else 0,
                "correct": correct,
                "skill": data_item.get("skill", ""),
                "extraction_method": extraction_method,
            }

            return {
                "accuracy": 1.0 if correct else 0.0,
                "hit": 1 if correct else 0,
            }, sample

        except Exception as e:
            return {"accuracy": 0.0, "hit": 0}, {"error": str(e)}


def compute_skill_metrics(samples: List[dict]) -> Dict:
    import pandas as pd

    df = pd.DataFrame(samples)

    if "hit" not in df.columns or "skill" not in df.columns:
        return {"overall_accuracy": df.get("hit", pd.Series([0])).mean()}

    metrics = {}

    # Overall accuracy
    metrics["Overall"] = {
        "total": len(df),
        "correct": int(df["hit"].sum()),
        "accuracy": float(df["hit"].mean() * 100),
    }

    # By skill category
    skill_keywords = ["inductive", "deductive", "numerical", "spatial", "mechanical"]

    for skill in skill_keywords:
        skill_df = df[df["skill"].str.contains(skill, case=False, na=False)]
        if len(skill_df) > 0:
            metrics[skill] = {
                "total": len(skill_df),
                "correct": int(skill_df["hit"].sum()),
                "accuracy": float(skill_df["hit"].mean() * 100),
            }

    return metrics


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            LogicVista(
                split="test",
                use_gpt_extraction=True,
                judge_model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=512,
            )
        )
    )
