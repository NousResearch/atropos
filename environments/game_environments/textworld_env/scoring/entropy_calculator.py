"""Entropy-based confidence scoring for LLM responses.

Implements entropy and varentropy calculations based on token-level logprobs
to determine model confidence in generated responses.
"""

import math
from typing import Any, Dict, List, Optional


def calculate_token_entropy(token_logprobs: List[Dict[str, Any]]) -> float:
    """Calculate Shannon entropy for a single token's top logprobs.

    Args:
        token_logprobs: List of {"token": str, "logprob": float} dicts

    Returns:
        Shannon entropy (higher = more uncertain)
    """
    if not token_logprobs:
        return 0.0

    # Convert logprobs to probabilities
    logprobs = [item["logprob"] for item in token_logprobs]
    probs = [math.exp(lp) for lp in logprobs]

    # Calculate Shannon entropy: -sum(p * log(p))
    entropy = 0.0
    for prob in probs:
        if prob > 0:
            entropy -= prob * math.log(prob)

    return entropy


def calculate_sequence_entropy(logprobs_data: List[Dict[str, Any]]) -> float:
    """Calculate mean entropy across a sequence of tokens.

    Args:
        logprobs_data: List of token logprobs data from completion response

    Returns:
        Mean entropy across all tokens
    """
    if not logprobs_data:
        return 0.0

    total_entropy = 0.0
    valid_tokens = 0

    for token_data in logprobs_data:
        if "top_logprobs" in token_data and token_data["top_logprobs"]:
            token_entropy = calculate_token_entropy(token_data["top_logprobs"])
            total_entropy += token_entropy
            valid_tokens += 1

    return total_entropy / valid_tokens if valid_tokens > 0 else 0.0


def calculate_varentropy(logprobs_data: List[Dict[str, Any]]) -> float:
    """Calculate variance of entropy across token sequence (varentropy).

    Varentropy measures "uncertainty about uncertainty" - how much the
    model's confidence varies across the sequence.

    Args:
        logprobs_data: List of token logprobs data from completion response

    Returns:
        Variance of entropy across tokens
    """
    if not logprobs_data or len(logprobs_data) < 2:
        return 0.0

    # Calculate entropy for each token
    entropies = []
    for token_data in logprobs_data:
        if "top_logprobs" in token_data and token_data["top_logprobs"]:
            token_entropy = calculate_token_entropy(token_data["top_logprobs"])
            entropies.append(token_entropy)

    if len(entropies) < 2:
        return 0.0

    # Calculate variance
    mean_entropy = sum(entropies) / len(entropies)
    variance = sum((e - mean_entropy) ** 2 for e in entropies) / len(entropies)

    return variance


def confidence_score(
    logprobs_data: Optional[List[Dict[str, Any]]],
    entropy_weight: float = 0.7,
    varentropy_weight: float = 0.3,
) -> float:
    """Calculate combined confidence score from entropy and varentropy.

    Lower entropy + lower varentropy = higher confidence
    Returns inverted score so higher values indicate higher confidence.

    Args:
        logprobs_data: Token logprobs data from completion response
        entropy_weight: Weight for entropy component
        varentropy_weight: Weight for varentropy component

    Returns:
        Confidence score (higher = more confident, range roughly 0-1)
    """
    if not logprobs_data:
        return 0.0

    entropy = calculate_sequence_entropy(logprobs_data)
    varentropy = calculate_varentropy(logprobs_data)

    # Combine entropy and varentropy (both are "bad" so we want lower values)
    combined_uncertainty = entropy_weight * entropy + varentropy_weight * varentropy

    # Convert to confidence score (invert and normalize)
    # Use sigmoid-like function to map to 0-1 range
    confidence = 1.0 / (1.0 + combined_uncertainty)

    return confidence


def classify_confidence(entropy: float, varentropy: float) -> str:
    """Classify response based on entropy/varentropy combination.

    Based on Entropix sampler approach:
    - Low entropy + low varentropy: "confident flow"
    - High entropy + low varentropy: "uncertain but consistent"
    - Low entropy + high varentropy: "confident exploration"
    - High entropy + high varentropy: "uncertain noise"

    Args:
        entropy: Sequence entropy
        varentropy: Variance of entropy

    Returns:
        Classification string
    """
    entropy_threshold = 2.0
    varentropy_threshold = 1.0

    high_entropy = entropy > entropy_threshold
    high_varentropy = varentropy > varentropy_threshold

    if not high_entropy and not high_varentropy:
        return "confident_flow"
    elif high_entropy and not high_varentropy:
        return "uncertain_consistent"
    elif not high_entropy and high_varentropy:
        return "confident_exploration"
    else:
        return "uncertain_noise"
