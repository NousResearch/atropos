import asyncio
import base64
import io
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import openai
from datasets import load_dataset
from openai import AsyncOpenAI
from PIL import Image

from environments.eval_environments.eval_base import EvalBase, eval_runner


class CharXiv(EvalBase):
    """
    CharXiv evaluation environment.

    A benchmark for chart reasoning from arXiv papers.
    https://charxiv.github.io/
    """

    CATEGORY_NAMES = {
        "descriptive": {
            0: "INEX",
            1: "ENUM",
            2: "PATT",
            3: "CNTG",
            4: "COMP",
        },
        "reasoning": {
            0: "TC",
            1: "TG",
            2: "NC",
            3: "NG",
        },
    }

    def setup_data(self) -> list:
        mode = getattr(self, "mode", "reasoning")
        split = getattr(self, "split", "val")
        split_name = "validation" if split == "val" else "test"

        raw_dataset = load_dataset("princeton-nlp/CharXiv", split=split_name)

        data = []
        for item in raw_dataset:
            if mode == "reasoning":
                data.append(
                    {
                        "image": item["image"],
                        "figure_id": item.get("original_id", ""),
                        "query": item["reasoning_q"],
                        "answer": item["reasoning_a"],
                        "qa_source": item.get("reasoning_q_source", 0),
                        "category": item.get("category", ""),
                    }
                )
            else:
                for i in range(1, 5):
                    q_key = f"descriptive_q{i}"
                    a_key = f"descriptive_a{i}"
                    if item.get(q_key) and item.get(a_key):
                        data.append(
                            {
                                "image": item["image"],
                                "figure_id": item.get("original_id", ""),
                                "query": item[q_key],
                                "answer": item[a_key],
                                "inst_category": i - 1,
                                "category": item.get("category", ""),
                            }
                        )

        print(f"Loaded {len(data)} examples from CharXiv ({mode}, {split})")
        return data

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        images_path: Optional[str] = getattr(self, "images_path", None)
        if images_path:
            figure_id = item.get("figure_id", item.get("id", 0))
            image_path = Path(images_path) / f"{figure_id}.png"
            if not image_path.exists():
                image_path = Path(images_path) / f"{figure_id}.jpg"
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        if "image" in item and item["image"] is not None:
            return self.encode_image(item["image"])

        raise ValueError(
            f"Could not find image for item: {item.get('figure_id', 'unknown')}"
        )

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        query = item.get("query", item.get("question", ""))

        mode = getattr(self, "mode", "reasoning")
        if mode == "descriptive":
            instruction = (
                "Answer the question about this chart. Be concise and specific."
            )
        else:
            instruction = "Analyze this chart and answer the question. Provide your answer directly."

        prompt = f"{instruction}\n\nQuestion: {query}"

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

    def score_reasoning(self, prediction: str, answer: str) -> bool:
        pred = prediction.strip().lower()
        ans = answer.strip().lower()

        if ans in pred:
            return True
        if pred == ans:
            return True

        pred_clean = re.sub(r"[^a-z0-9\s]", "", pred)
        ans_clean = re.sub(r"[^a-z0-9\s]", "", ans)

        return ans_clean in pred_clean or pred_clean == ans_clean

    async def score_descriptive(self, query: str, prediction: str, answer: str) -> bool:
        judge_model = getattr(self, "judge_model", "gpt-4o")
        judge_base_url = getattr(self, "judge_base_url", "https://api.openai.com/v1")
        judge_api_key = os.environ.get(
            getattr(self, "judge_api_key_env", "OPENAI_API_KEY"), ""
        )

        if not judge_api_key:
            return self.score_reasoning(prediction, answer)

        judge_client = openai.AsyncOpenAI(
            api_key=judge_api_key,
            base_url=judge_base_url,
        )

        prompt = f"""Evaluate if the model's response correctly answers the question about the chart.

Question: {query}
Correct Answer: {answer}
Model Response: {prediction}

Does the model's response correctly answer the question? Consider:
- The core information matches
- Minor wording differences are acceptable
- The response addresses what was asked

Output only "1" if the response is correct, or "0" if incorrect."""

        try:
            response = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            result = response.choices[0].message.content.strip()
            return "1" in result
        except Exception as e:
            print(f"Judge error: {e}")
            return False

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

            answer = data_item.get("answer", "")
            query = data_item.get("query", data_item.get("question", ""))

            mode = getattr(self, "mode", "reasoning")
            if mode == "descriptive":
                correct = await self.score_descriptive(query, response, answer)
            else:
                correct = self.score_reasoning(response, answer)

            category_id = data_item.get("inst_category", data_item.get("qa_source", 0))
            category_name = self.CATEGORY_NAMES.get(mode, {}).get(
                category_id, "unknown"
            )

            sample = {
                "figure_id": data_item.get("figure_id", ""),
                "question": query,
                "answer": answer,
                "prediction": response,
                "correct": correct,
                "category": category_name,
            }

            return {"accuracy": 1.0 if correct else 0.0}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            CharXiv,
            mode="reasoning",
            split="val",
            temperature=0.0,
            max_tokens=4096,
        )
    )
