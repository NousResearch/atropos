import asyncio
import base64
import io
import re
from typing import List, Tuple

from datasets import load_dataset
from openai import AsyncOpenAI
from PIL import Image

from environments.eval_environments.eval_base import EvalBase, eval_runner


class InfoVQA(EvalBase):
    def setup_data(self) -> list:
        split = getattr(self, "split", "validation")
        dataset = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split=split)
        print(f"Loaded {len(dataset)} examples from InfoVQA ({split})")
        return list(dataset)

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        if "image" in item and item["image"] is not None:
            if isinstance(item["image"], Image.Image):
                return self.encode_image(item["image"])
        raise ValueError(f"Could not find image for item {item.get('id', 'unknown')}")

    def build_messages(self, item: dict) -> List[dict]:
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")

        prompt = f"""Look at the infographic and answer the question.

Question: {question}

Provide only the answer, as concisely as possible."""

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

    def extract_answer(self, response: str) -> str:
        response = response.strip()

        patterns = [
            r"answer[:\s]+(.+?)(?:\.|$)",
            r"\"([^\"]+)\"",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        lines = response.split("\n")
        if lines:
            return lines[-1].strip()

        return response

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(text.split())
        return text

    def anls_score(
        self, prediction: str, answers: List[str], threshold: float = 0.5
    ) -> float:
        """
        Calculate Average Normalized Levenshtein Similarity (ANLS).
        This is the standard metric for InfoVQA.
        """
        pred_norm = self.normalize_text(prediction)

        if not pred_norm:
            return 0.0

        max_score = 0.0
        for answer in answers:
            ans_norm = self.normalize_text(answer)
            if not ans_norm:
                continue

            if pred_norm == ans_norm:
                max_score = 1.0
                break

            distance = self._levenshtein_distance(pred_norm, ans_norm)
            max_len = max(len(pred_norm), len(ans_norm))
            nls = 1 - distance / max_len if max_len > 0 else 0

            if nls >= threshold:
                max_score = max(max_score, nls)

        return max_score

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

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
                return {"accuracy": 0.0, "anls": 0.0}, {"error": "Empty response"}

            message = completion.choices[0].message
            response = message.content or ""
            if hasattr(message, "reasoning") and message.reasoning and not response:
                response = message.reasoning
            if not response and hasattr(message, "model_extra"):
                reasoning = message.model_extra.get("reasoning", "")
                if reasoning:
                    response = reasoning

            if not response:
                return {"accuracy": 0.0, "anls": 0.0}, {"error": "Empty response"}

            extracted = self.extract_answer(response)
            answers = data_item.get("answer", [])
            if isinstance(answers, str):
                answers = [answers]

            anls = self.anls_score(extracted, answers)
            correct = anls >= 0.5

            sample = {
                "id": data_item.get("id", ""),
                "question": data_item.get("question", ""),
                "answers": answers,
                "prediction": extracted,
                "anls": anls,
                "correct": correct,
            }

            return {"accuracy": 1.0 if correct else 0.0, "anls": anls}, sample

        except Exception as e:
            return {"accuracy": 0.0, "anls": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            InfoVQA,
            split="test",
            temperature=0.0,
            max_tokens=256,
        )
    )
