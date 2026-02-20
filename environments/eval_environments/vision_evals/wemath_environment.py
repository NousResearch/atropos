"""We-Math evaluation environment."""

import asyncio
import base64
import io
import re
import string
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from datasets import load_dataset
from environments.eval_environments.eval import EvalBase, eval_runner
from PIL import Image

from atroposlib.envs.server_handling.server_manager import ServerManager


class WeMath(EvalBase):
    """
    We-Math evaluation environment.

    A benchmark for visual mathematical reasoning with multi-step problems
    and 4-dimensional evaluation metrics (IK, IG, CM, RM).
    """

    def setup_data(self) -> list:
        split = getattr(self, "split", "testmini")
        dataset = load_dataset("We-Math/We-Math", split=split)
        print(f"Loaded {len(dataset)} examples from We-Math ({split})")
        return list(dataset)

    def encode_image(self, pil_image: Image.Image) -> str:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_image_base64(self, item: dict) -> str:
        img = item.get("image_path") or item.get("image")
        if img is not None:
            if isinstance(img, Image.Image):
                return self.encode_image(img)
            elif isinstance(img, bytes):
                return base64.b64encode(img).decode("utf-8")
        raise ValueError(
            f"Could not find image for item {item.get('ID', item.get('problem_id', 'unknown'))}"
        )

    def build_messages(self, item: dict) -> List[dict]:
        """Build prompt with question, options, and optional hint (MCQ format)."""
        image_base64 = self.get_image_base64(item)
        question = item.get("question", "")

        # Build options from A-H if present
        options = {}
        for letter in string.ascii_uppercase[:8]:  # A-H
            if (
                letter in item
                and item[letter] is not None
                and not pd.isna(item.get(letter, float("nan")))
            ):
                options[letter] = item[letter]

        # Build prompt
        prompt_parts = []

        # Add hint if present
        hint = item.get("hint", "")
        if hint and not pd.isna(hint):
            prompt_parts.append(f"Hint: {hint}")

        prompt_parts.append(f"Question: {question}")

        # Add options if present
        if options:
            options_text = "Options:\n"
            for letter, value in options.items():
                options_text += f"{letter}. {value}\n"
            prompt_parts.append(options_text)

        # Add COT requirement if dataset is WeMath_COT
        use_cot = getattr(self, "use_cot", False)
        requirement = item.get("requirement", "")
        if use_cot and requirement and not pd.isna(requirement):
            prompt_parts.append(requirement)
        else:
            prompt_parts.append(
                "Answer with the option's letter from the given choices directly."
            )

        prompt = "\n".join(prompt_parts)

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
        """
        Extract MCQ answer letter from response.

        Following VLMEvalKit logic: look for letter after "Answer" keyword,
        or extract first valid letter (A-H).
        """
        response = str(response).strip()

        # Try to find answer after "Answer" keyword
        answer_match = re.search(r"Answer[:\s]*([A-Ha-h])", response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()

        # Clean response and look for first valid letter
        cleaned = re.sub(r"[>><<:.\s]", "", response).strip()
        if cleaned and cleaned[0].upper() in "ABCDEFGH":
            return cleaned[0].upper()

        # Fallback: find any letter A-H in the response
        for char in response.upper():
            if char in "ABCDEFGH":
                return char

        return ""

    def score(self, prediction: str, answer: str) -> bool:
        """Check if prediction matches answer (case-insensitive)."""
        if not prediction:
            return False
        return prediction.upper() == answer.upper()

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

            extracted = self.extract_answer(response)
            answer = data_item.get("answer", "")
            correct = self.score(extracted, answer)

            # Get problem metadata for 4-dimensional analysis
            problem_id = data_item.get("ID", data_item.get("problem_id", ""))
            key = data_item.get("key", "")  # e.g., "2steps_1", "2steps_multi", etc.

            sample = {
                "ID": problem_id,
                "key": key,
                "question": data_item.get("question", ""),
                "answer": answer,
                "prediction": extracted,
                "raw_response": response[:500],  # Truncate for logging
                "hit": 1 if correct else 0,
                "joker": correct,  # VLMEvalKit naming convention
            }

            return {
                "accuracy": 1.0 if correct else 0.0,
                "hit": 1 if correct else 0,
            }, sample

        except Exception as e:
            return {"accuracy": 0.0, "hit": 0}, {"error": str(e)}


def compute_4d_metrics(samples: List[dict]) -> Dict:
    """
    Compute We-Math 4-dimensional metrics from evaluation samples.

    This implements the evaluation logic from VLMEvalKit's wemath.py.

    Returns metrics for:
    - IK (Insufficient Knowledge): Steps wrong AND multi wrong
    - IG (Inadequate Generalization): Steps right BUT multi wrong
    - CM (Complete Mastery): Steps right AND multi right
    - RM (Rote Memorization): Steps wrong BUT multi right
    """
    # Convert samples to DataFrame
    df = pd.DataFrame(samples)

    if "key" not in df.columns or df["key"].isna().all():
        # Dataset doesn't have step structure, return basic accuracy
        return {
            "overall_accuracy": df["hit"].mean() if "hit" in df.columns else 0.0,
            "note": "Dataset lacks step structure for 4D metrics",
        }

    # Separate by step type
    data_2steps = df[df["key"].str.contains("2steps", na=False)]
    data_3steps = df[df["key"].str.contains("3steps", na=False)]

    metrics = {}

    # Process 2-step problems
    if len(data_2steps) > 0:
        merged_2steps = _process_steps_data(data_2steps, 2)
        if merged_2steps is not None:
            metrics["2step"] = _calculate_step_metrics(merged_2steps, 2)

    # Process 3-step problems
    if len(data_3steps) > 0:
        merged_3steps = _process_steps_data(data_3steps, 3)
        if merged_3steps is not None:
            metrics["3step"] = _calculate_step_metrics(merged_3steps, 3)

    # Compute overall 4D metrics
    if "2step" in metrics or "3step" in metrics:
        total_counts = _compute_total_counts(metrics)
        total_count = 525  # Standard We-Math total

        # Compute rates and final scores
        final_metrics = _compute_final_scores(total_counts, total_count)
        metrics["overall"] = final_metrics

    # Basic accuracy
    metrics["step_accuracy"] = df["hit"].mean() if "hit" in df.columns else 0.0

    return metrics


def _process_steps_data(df: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Process step data and merge by problem ID."""
    try:
        steps_data = {}
        for i in range(1, steps + 1):
            key = f"{steps}steps_{i}"
            step_df = df[df["key"] == key].copy()
            if len(step_df) == 0:
                return None
            step_df.columns = [f"{col}_{i}" for col in step_df.columns]
            steps_data[i] = step_df

        # Get multi-step data
        multi_key = f"{steps}steps_multi"
        multi_df = df[df["key"] == multi_key].copy()
        if len(multi_df) == 0:
            return None
        multi_df.columns = [f"{col}_multi" for col in multi_df.columns]

        # Merge all steps
        merged = steps_data[1]
        for i in range(2, steps + 1):
            merged = pd.merge(
                merged,
                steps_data[i],
                left_on="ID_1",
                right_on=f"ID_{i}",
                how="left",
            )
        merged = pd.merge(
            merged, multi_df, left_on="ID_1", right_on="ID_multi", how="left"
        )

        return merged
    except Exception:
        return None


def _calculate_step_metrics(merged: pd.DataFrame, steps: int) -> Dict:
    """Calculate metrics for a step type (2-step or 3-step)."""
    try:
        # Get joker columns
        joker_cols = [f"joker_{i}" for i in range(1, steps + 1)]
        joker_multi = "joker_multi"

        # Check if columns exist
        for col in joker_cols + [joker_multi]:
            if col not in merged.columns:
                return {}

        # Calculate conditions
        all_steps_correct = merged[joker_cols].all(axis=1)
        any_step_correct = merged[joker_cols].any(axis=1)
        all_steps_wrong = ~merged[joker_cols].any(axis=1)
        any_step_wrong = ~merged[joker_cols].all(axis=1)
        multi_correct = merged[joker_multi] == True  # noqa: E712

        return {
            # Strict: ALL steps must be correct
            "CM_strict": int((all_steps_correct & multi_correct).sum()),
            "IG": int((all_steps_correct & ~multi_correct).sum()),
            "RM_strict": int((any_step_wrong & multi_correct).sum()),
            "IK": int((any_step_wrong & ~multi_correct).sum()),
            # Loose: ANY step correct
            "CM_loose": int((any_step_correct & multi_correct).sum()),
            "RM_loose": int((all_steps_wrong & multi_correct).sum()),
            "total": len(merged),
        }
    except Exception:
        return {}


def _compute_total_counts(metrics: Dict) -> Dict:
    """Aggregate counts across step types."""
    totals = defaultdict(int)

    for step_type in ["2step", "3step"]:
        if step_type in metrics:
            for key in ["CM_strict", "CM_loose", "IG", "RM_strict", "RM_loose", "IK"]:
                if key in metrics[step_type]:
                    totals[key] += metrics[step_type][key]

    return dict(totals)


def _compute_final_scores(total_counts: Dict, total_count: int = 525) -> Dict:
    """Compute final 4D scores and rates."""
    results = {}

    # Calculate rates
    for key in ["IK", "IG", "CM_strict", "CM_loose", "RM_strict", "RM_loose"]:
        count = total_counts.get(key, 0)
        results[f"{key}_count"] = count
        results[f"{key}_rate"] = count / total_count if total_count > 0 else 0.0

    # Calculate RM rates (relative to CM + RM)
    cm_rm_strict = total_counts.get("CM_strict", 0) + total_counts.get("RM_strict", 0)
    cm_rm_loose = total_counts.get("CM_loose", 0) + total_counts.get("RM_loose", 0)

    results["RM_strict_relative"] = (
        total_counts.get("RM_strict", 0) / cm_rm_strict if cm_rm_strict > 0 else 0.0
    )
    results["RM_loose_relative"] = (
        total_counts.get("RM_loose", 0) / cm_rm_loose if cm_rm_loose > 0 else 0.0
    )

    # Final scores (VLMEvalKit formula)
    results["score_strict"] = (
        total_count
        - 0.5 * total_counts.get("IG", 0)
        - total_counts.get("RM_strict", 0)
        - total_counts.get("IK", 0)
    ) / total_count

    results["score_loose"] = (
        total_count
        - 0.5 * total_counts.get("IG", 0)
        - total_counts.get("RM_loose", 0)
        - total_counts.get("IK", 0)
    ) / total_count

    return results


if __name__ == "__main__":
    asyncio.run(
        eval_runner(
            WeMath(split="testmini", use_cot=False, temperature=0.0, max_tokens=512)
        )
    )
