"""MMVet evaluation environment."""

import asyncio
import base64
import io
import os
from typing import List, Optional, Tuple

import openai
from datasets import load_dataset
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.eval_environments.eval import EvalBase, eval_runner


class MMVet(EvalBase):
    """MMVet evaluation - comprehensive VLM capability benchmark with GPT-based scoring."""

    def setup_data(self) -> list:
        split = getattr(self, "split", "test")

        try:
            dataset = load_dataset("lmms-lab/MMVet", split=split)
            print(f"Loaded {len(dataset)} examples from MMVet ({split})")
            return list(dataset)
        except Exception as e:
            print(f"Warning: Could not load MMVet: {e}")
            try:
                dataset = load_dataset("whyu/MM-Vet", split=split)
                print(f"Loaded {len(dataset)} examples from MMVet ({split})")
                return list(dataset)
            except Exception:
                raise ValueError(f"Could not load MMVet dataset: {e}")

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

        content = []
        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )
        content.append({"type": "text", "text": question})

        return [{"role": "user", "content": content}]

    async def gpt_score(self, question: str, answer: str, prediction: str) -> float:
        """Use GPT to score the prediction against the ground truth answer."""
        judge_model = getattr(self, "judge_model", "gpt-4o-mini")
        judge_base_url = getattr(self, "judge_base_url", "https://api.openai.com/v1")
        judge_api_key = os.environ.get(
            getattr(self, "judge_api_key_env", "OPENAI_API_KEY"), ""
        )

        if not judge_api_key:
            pred_lower = prediction.lower().strip()
            ans_lower = answer.lower().strip()
            if pred_lower == ans_lower:
                return 1.0
            elif ans_lower in pred_lower or pred_lower in ans_lower:
                return 0.5
            return 0.0

        try:
            judge_client = openai.AsyncOpenAI(
                api_key=judge_api_key,
                base_url=judge_base_url,
            )

            prompt = f"""You are evaluating the quality of a model's answer compared to a reference answer.

Question: {question}

Reference Answer: {answer}

Model's Answer: {prediction}

Score the model's answer on a scale from 0 to 1:
- 1.0: Completely correct and matches the reference
- 0.5-0.9: Partially correct or captures the main idea
- 0.1-0.4: Somewhat related but mostly incorrect
- 0.0: Completely wrong or irrelevant

Output ONLY a single number between 0 and 1."""

            completion = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            result = completion.choices[0].message.content.strip()
            try:
                score = float(result)
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.0

        except Exception as e:
            print(f"GPT scoring error: {e}")
            pred_lower = prediction.lower().strip()
            ans_lower = answer.lower().strip()
            if pred_lower == ans_lower:
                return 1.0
            elif ans_lower in pred_lower:
                return 0.5
            return 0.0

    async def run_item(self, server: ServerManager, data_item: dict) -> Tuple[dict, dict]:
        try:
            messages = self.build_messages(data_item)

            completion = await self.chat_completion(server, messages)

            if not completion.choices:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            message = completion.choices[0].message
            response = message.content or ""

            if not response:
                return {"accuracy": 0.0}, {"error": "Empty response"}

            question = data_item.get("question", "")
            answer = data_item.get("answer", "")

            use_gpt_scoring = getattr(self, "use_gpt_scoring", True)
            if use_gpt_scoring:
                score = await self.gpt_score(question, answer, response)
            else:
                pred_lower = response.lower().strip()
                ans_lower = answer.lower().strip()
                if pred_lower == ans_lower:
                    score = 1.0
                elif ans_lower in pred_lower:
                    score = 0.5
                else:
                    score = 0.0

            sample = {
                "id": data_item.get("index", data_item.get("id", "")),
                "question": question[:200],
                "category": data_item.get("capability", data_item.get("category", "")),
                "answer": answer[:200],
                "prediction": response[:500],
                "score": score,
            }

            return {"accuracy": score}, sample

        except Exception as e:
            return {"accuracy": 0.0}, {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            MMVet(
                split="test",
                use_gpt_scoring=True,
                judge_model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=512,
            )
        )
    )
