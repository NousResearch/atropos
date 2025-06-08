"""
Reward functions for the BLEUBERI environment.
"""

from typing import List

import evaluate
from bert_score import BERTScorer

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = BERTScorer(model_type="distilbert-base-uncased")


def extract_answer(completion: str) -> str:
    """Extract the answer from a completion with potential thinking tags."""
    if "<answer>" in completion and "</answer>" in completion:
        return completion.split("<answer>")[1].split("</answer>")[0].strip()
    elif "<answer>" in completion:
        return completion.split("<answer>")[1].strip()
    else:
        return completion.strip()


def bleu_reward(completions: List[str], references: List[List[str]]) -> List[float]:
    """
    Calculate BLEU scores for completions against references.

    Args:
        completions: List of model completions
        references: List of lists of reference completions (one or more references per completion)

    Returns:
        List of BLEU scores
    """
    scores = []
    for completion, refs in zip(completions, references):
        if isinstance(completion, dict) and "content" in completion:
            completion = completion["content"]
        elif (
            isinstance(completion, list)
            and len(completion) > 0
            and isinstance(completion[0], dict)
            and "role" in completion[0]
        ):
            completion = next(msg for msg in completion if msg["role"] == "assistant")[
                "content"
            ]

        # Handle invalid completions
        if isinstance(completion, float) or (
            isinstance(completion, str) and len(completion.strip()) == 0
        ):
            scores.append(0)
            continue

        # Extract answer from completion if it has thinking tags
        completion = extract_answer(completion)
        if len(completion.strip()) == 0:
            scores.append(0)
            continue

        # Calculate BLEU score
        bleu_score = bleu.compute(
            predictions=[completion], references=[refs], smooth=True
        )
        scores.append(bleu_score["bleu"])

    return scores


def rouge_reward(completions: List[str], references: List[List[str]]) -> List[float]:
    """
    Calculate ROUGE scores for completions against references.

    Args:
        completions: List of model completions
        references: List of lists of reference completions (one reference per completion)

    Returns:
        List of ROUGE-L scores
    """
    scores = []
    for completion, refs in zip(completions, references):
        if isinstance(completion, dict) and "content" in completion:
            completion = completion["content"]
        elif (
            isinstance(completion, list)
            and len(completion) > 0
            and isinstance(completion[0], dict)
            and "role" in completion[0]
        ):
            completion = next(msg for msg in completion if msg["role"] == "assistant")[
                "content"
            ]

        # Handle invalid completions
        if isinstance(completion, float) or (
            isinstance(completion, str) and len(completion.strip()) == 0
        ):
            scores.append(0)
            continue

        # Extract answer from completion if it has thinking tags
        completion = extract_answer(completion)
        if len(completion.strip()) == 0:
            scores.append(0)
            continue

        # Take first reference for ROUGE calculation
        ref = refs[0] if refs else ""
        rouge_score = rouge.compute(predictions=[completion], references=[ref])
        scores.append(rouge_score["rougeL"])

    return scores


def bertscore_reward(
    completions: List[str], references: List[List[str]]
) -> List[float]:
    """
    Calculate BERTScore for completions against references.

    Args:
        completions: List of model completions
        references: List of lists of reference completions (one reference per completion)

    Returns:
        List of BERTScore F1 scores
    """
    all_completions = []
    all_refs = []

    for completion, refs in zip(completions, references):
        if isinstance(completion, dict) and "content" in completion:
            completion = completion["content"]
        elif (
            isinstance(completion, list)
            and len(completion) > 0
            and isinstance(completion[0], dict)
            and "role" in completion[0]
        ):
            completion = next(msg for msg in completion if msg["role"] == "assistant")[
                "content"
            ]

        # Extract answer from completion if it has thinking tags
        completion = extract_answer(completion)

        # Take first reference for BERTScore calculation
        ref = refs[0] if refs else ""

        all_completions.append(completion)
        all_refs.append(ref)

    # Calculate BERTScore
    P, R, F1 = bertscore.score(all_completions, all_refs)
    scores = F1.tolist()

    return scores


def bleu_rouge_f1_reward(
    completions: List[str], references: List[List[str]]
) -> List[float]:
    """
    Calculate F1 of BLEU and ROUGE scores.

    Args:
        completions: List of model completions
        references: List of lists of reference completions (one reference per completion)

    Returns:
        List of F1(BLEU, ROUGE) scores
    """
    bleu_scores = bleu_reward(completions, references)
    rouge_scores = rouge_reward(completions, references)

    combined_scores = [
        2 * bleu * rouge / (bleu + rouge) if (bleu + rouge) > 0 else 0.0
        for bleu, rouge in zip(bleu_scores, rouge_scores)
    ]

    return combined_scores


# Registry of reward functions
REWARD_FUNCTIONS = {
    "bleu": bleu_reward,
    "rouge": rouge_reward,
    "bertscore": bertscore_reward,
    "bleu_rouge_f1": bleu_rouge_f1_reward,
}
