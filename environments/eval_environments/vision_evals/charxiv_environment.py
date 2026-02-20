"""CharXiv evaluation environment."""

import asyncio
import base64
import io
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import openai
from datasets import load_dataset
from environments.eval_environments.eval import EvalBase, eval_runner
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager

DESCRIPTIVE_CATEGORIES = {
    1: "Information Extraction",
    2: "Information Extraction",
    3: "Information Extraction",
    4: "Information Extraction",
    5: "Information Extraction",
    6: "Information Extraction",
    7: "Information Extraction",
    8: "Enumeration",
    9: "Enumeration",
    10: "Counting",
    11: "Pattern Recognition",
    12: "Counting",
    13: "Enumeration",
    14: "Enumeration",
    15: "Enumeration",
    16: "Pattern Recognition",
    17: "Compositionality",
    18: "Pattern Recognition",
    19: "Counting",
}

REASONING_CATEGORIES = {
    1: "Text-in-Chart",
    2: "Text-in-General",
    3: "Number-in-Chart",
    4: "Number-in-General",
}

DESCRIPTIVE_QUESTIONS = {
    1: "What is the title of the chart?",
    2: "What is the label of the x-axis?",
    3: "What is the label of the y-axis?",
    4: "What is the leftmost labeled tick on the x-axis?",
    5: "What is the rightmost labeled tick on the x-axis?",
    6: "What is the spatially lowest labeled tick on the y-axis?",
    7: "What is the spatially highest labeled tick on the y-axis?",
    8: "What are all the labels in the legend?",
    9: "List all the categories in the x-axis.",
    10: "How many distinct bars are there?",
    11: "Does the chart contain a grid?",
    12: "How many lines are there in the chart?",
    13: "Is there a legend in the chart?",
    14: "What are the names of the curves in the chart?",
    15: "Does the chart contain horizontal bars?",
    16: "Do the bars have error bars?",
    17: "Describe the general trend of the chart.",
    18: "Is there any point emphasized/highlighted in the chart?",
    19: "How many sections does the pie chart have?",
}

GRADING_QUERY_TEMPLATE = """You are evaluating a model's answer to a chart understanding question.

Question: {question}
Ground Truth Answer: {answer}
Model's Answer: {prediction}

Please evaluate whether the model's answer is correct or partially correct.
Consider semantic equivalence - different phrasings that mean the same thing should be considered correct.
For numerical answers, exact matches or very close values should be considered correct.
For yes/no questions, the meaning should match the ground truth.
For enumeration questions (listing items), all items should be present regardless of order.

Respond with a JSON object containing:
- "extract_answer": The key answer extracted from the model's response
- "score": A float from 0.0 to 1.0 indicating correctness (0.0 = wrong, 0.5 = partial, 1.0 = correct)

Example response: {{"extract_answer": "60", "score": 1.0}}"""


class CharXiv(EvalBase):
    MODES = ["descriptive", "reasoning"]

    def setup_data(self) -> list:
        mode = getattr(self, "mode", "descriptive")
        split = getattr(self, "split", "validation")

        dataset = load_dataset("princeton-nlp/CharXiv", "default", split=split)

        data = []
        for item in dataset:
            if mode == "descriptive":
                for i in range(1, 5):
                    q_key = f"descriptive_q{i}"
                    a_key = f"descriptive_a{i}"
                    if a_key in item and item[a_key]:
                        template_id = item.get(q_key, i)
                        if (
                            isinstance(template_id, int)
                            and template_id in DESCRIPTIVE_QUESTIONS
                        ):
                            question = DESCRIPTIVE_QUESTIONS[template_id]
                        else:
                            question = (
                                str(template_id)
                                if template_id
                                else f"Descriptive question {i}"
                            )

                        data.append(
                            {
                                "image": item["image"],
                                "question": question,
                                "answer": item[a_key],
                                "qid": (
                                    template_id if isinstance(template_id, int) else i
                                ),
                                "category": item.get("category", ""),
                                "grading_query": item.get("grading_query", ""),
                            }
                        )
            elif mode == "reasoning":
                if "reasoning_q" in item and item.get("reasoning_a"):
                    data.append(
                        {
                            "image": item["image"],
                            "question": item["reasoning_q"],
                            "answer": item["reasoning_a"],
                            "inst_category": item.get("category", 1),
                            "grading_query": item.get("grading_query", ""),
                        }
                    )
            else:
                raise ValueError(
                    f"Invalid mode: {mode}. Must be 'descriptive' or 'reasoning'."
                )

        print(f"Loaded {len(data)} examples from CharXiv ({mode}, {split})")
        return data

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        if "image" in item and item["image"] is not None:
            if isinstance(item["image"], Image.Image):
                return self.encode_image(item["image"])
        raise ValueError("Could not find image for item")

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

    async def _judge_with_gpt(
        self, question: str, answer: str, prediction: str, grading_query: str = ""
    ) -> Tuple[Optional[str], float]:
        judge_model = getattr(self, "judge_model", "gpt-4o-mini")
        judge_base_url = getattr(self, "judge_base_url", "https://api.openai.com/v1")
        judge_api_key = os.environ.get(
            getattr(self, "judge_api_key_env", "OPENAI_API_KEY"), ""
        )

        if not judge_api_key:
            return None, 0.0

        if grading_query:
            prompt = grading_query.replace("{PREDICTION}", prediction)
        else:
            prompt = GRADING_QUERY_TEMPLATE.format(
                question=question, answer=answer, prediction=prediction
            )

        try:
            judge_client = openai.AsyncOpenAI(
                api_key=judge_api_key,
                base_url=judge_base_url,
            )

            completion = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )

            response = completion.choices[0].message.content.strip()

            try:
                result = json.loads(response)
                if isinstance(result, dict):
                    extract_answer = result.get("extract_answer", "")
                    score = float(result.get("score", 0.0))
                    return extract_answer, score
            except json.JSONDecodeError:
                json_match = re.search(r"\{[^}]+\}", response)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        extract_answer = result.get("extract_answer", "")
                        score = float(result.get("score", 0.0))
                        return extract_answer, score
                    except (json.JSONDecodeError, ValueError):
                        pass

            return None, 0.0

        except Exception as e:
            print(f"GPT judge error: {e}")
            return None, 0.0

    def _fallback_score(
        self, prediction: str, answer: str, mode: str
    ) -> Tuple[str, float]:
        prediction = prediction.strip().lower()
        answer = answer.strip().lower()

        if not prediction:
            return "", 0.0

        if mode == "reasoning":
            if answer in prediction:
                return prediction, 1.0
            try:
                pred_nums = re.findall(r"-?\d+\.?\d*", prediction)
                ans_nums = re.findall(r"-?\d+\.?\d*", answer)
                if pred_nums and ans_nums:
                    for p in pred_nums:
                        for a in ans_nums:
                            if abs(float(p) - float(a)) < 0.01:
                                return prediction, 1.0
            except ValueError:
                pass
            return prediction, 0.0

        else:
            pred_words = set(prediction.split())
            ans_words = set(answer.split())
            if not ans_words:
                return prediction, 0.0
            overlap = len(pred_words & ans_words) / len(ans_words)
            return prediction, min(overlap, 1.0)

    def get_category(self, item: dict, mode: str) -> str:
        if mode == "descriptive":
            qid = item.get("qid", 1)
            return DESCRIPTIVE_CATEGORIES.get(qid, "Information Extraction")
        else:
            inst_category = item.get("inst_category", 1)
            return REASONING_CATEGORIES.get(inst_category, "Text-in-Chart")

    async def run_item(
        self, server: ServerManager, data_item: dict
    ) -> Tuple[dict, dict]:
        try:
            messages = self.build_messages(data_item)
            mode = getattr(self, "mode", "descriptive")

            completion = await self.chat_completion(server, messages)

            if not completion.choices:
                return {"accuracy": 0.0, "score": 0.0}, {"error": "Empty response"}

            message = completion.choices[0].message
            response = message.content or ""
            if hasattr(message, "reasoning") and message.reasoning and not response:
                response = message.reasoning
            if not response and hasattr(message, "model_extra"):
                reasoning = message.model_extra.get("reasoning", "")
                if reasoning:
                    response = reasoning

            if not response:
                return {"accuracy": 0.0, "score": 0.0}, {"error": "Empty response"}

            use_gpt_judge = getattr(self, "use_gpt_judge", True)
            grading_query = data_item.get("grading_query", "")
            answer = data_item.get("answer", "")
            question = data_item.get("question", "")

            if use_gpt_judge:
                extracted, score = await self._judge_with_gpt(
                    question, answer, response, grading_query
                )
                evaluation_method = "gpt_judge"
            else:
                extracted, score = self._fallback_score(response, answer, mode)
                evaluation_method = "fallback"

            if extracted is None:
                extracted, score = self._fallback_score(response, answer, mode)
                evaluation_method = "fallback"

            category = self.get_category(data_item, mode)

            sample = {
                "question": data_item.get("question", ""),
                "answer": answer,
                "prediction": response[:500],
                "extract_answer": extracted,
                "score": score,
                "category": category,
                "mode": mode,
                "qid": data_item.get("qid", data_item.get("inst_category", "")),
                "evaluation_method": evaluation_method,
            }

            return {"accuracy": score, "score": score}, sample

        except Exception as e:
            return {"accuracy": 0.0, "score": 0.0}, {"error": str(e)}


def compute_category_metrics(samples: List[dict]) -> Dict:
    from collections import defaultdict

    scores_by_category = defaultdict(list)

    for sample in samples:
        if "error" in sample:
            continue
        category = sample.get("category", "Unknown")
        score = sample.get("score", 0.0)
        scores_by_category[category].append(score)

    metrics = {}
    total_score = 0.0
    total_count = 0

    for category, scores in scores_by_category.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            metrics[category] = {
                "count": len(scores),
                "average_score": avg_score,
            }
            total_score += sum(scores)
            total_count += len(scores)

    if total_count > 0:
        metrics["Overall"] = {
            "count": total_count,
            "average_score": total_score / total_count,
        }

    return metrics


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            CharXiv(
                mode="descriptive",  # or "reasoning"
                split="validation",
                use_gpt_judge=True,
                judge_model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=1024,
            )
        )
    )
