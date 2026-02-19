"""
Data processing utilities for GRPO trainer.

Handles data retrieval from Atropos API, padding, batching,
and advantage normalization.

Also extracts inference logprobs for proper GRPO loss computation:
- Inference logprobs serve as π_old (reference policy) for importance sampling
- They are batched and padded to align token-by-token with training labels

Also supports optional on-policy distillation arrays:
- distill_token_ids[seq][pos][top_k]
- distill_logprobs[seq][pos][top_k]
"""

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from .api import get_batch


def pad_data_to_good_offset(
    data: dict,
    batch_size: int,
    extract_inference_logprobs: bool = True,
) -> Tuple[
    List[torch.Tensor],  # token_batches
    List[torch.Tensor],  # label_batches
    List[torch.Tensor],  # advantage_batches
    List[torch.Tensor],  # temperature_batches
    Optional[List[torch.Tensor]],  # inference_logprob_batches (aligned with labels)
    Optional[List[list]],  # distill_token_id_batches (nested ragged arrays)
    Optional[List[list]],  # distill_logprob_batches (nested ragged arrays)
]:
    """
    Pad and batch data from the Atropos API.

    Processes raw batch data into properly padded tensors suitable for training:
    - Pads token sequences to nearest multiple of 64
    - Normalizes advantage scores
    - Extracts temperature values
    - Extracts and pads inference logprobs for proper GRPO loss computation

    Args:
        data: Raw batch data from Atropos API
        batch_size: Size of each training batch
        extract_inference_logprobs: Whether to extract inference logprobs

    Returns:
        Tuple of (
            token_batches,
            label_batches,
            advantage_batches,
            temperature_batches,
            inference_logprob_batches,
            distill_token_id_batches,
            distill_logprob_batches,
        )
        inference_logprob_batches is None if extract_inference_logprobs=False or no logprobs in data

    Note:
        inference_logprob_batches are padded with 0.0 at positions where labels == -100.
        This allows token-by-token alignment during GRPO loss computation.
    """
    max_token_len = max(
        [max([len(x) for x in item["tokens"]]) for item in data["batch"]]
    )

    # Pad to nearest multiple of 64 for GPU efficiency
    good_multiple = 64
    if (max_token_len - 1) % (good_multiple) != 0:
        max_token_len = math.ceil((max_token_len - 1) / (good_multiple)) * good_multiple
        token_setup_len = max_token_len + 1  # +1 for causal shift
    else:
        token_setup_len = max_token_len
        max_token_len = max_token_len - 1  # -1 for causal shift

    # Process all items
    input_ids = []
    labels = []
    advantages = []
    lengths = []
    temperatures = []
    inference_logprobs_padded: List[np.ndarray] = []  # Padded to match labels shape
    has_any_logprobs = False
    distill_token_ids_padded: List[list] = []
    distill_logprobs_padded: List[list] = []
    has_any_distill = False

    for item in data["batch"]:
        # Normalize advantage scores
        scores = np.array(item["scores"])
        if len(scores) > 1:
            scores = scores - scores.mean()
            scores = scores / max(scores.std(), 1e-8)
        item["scores"] = scores

        # Handle score overrides
        if item["overrides"] is not None:
            for i in range(len(item["overrides"])):
                if item["overrides"][i].get("set_advantage_to_zero", False):
                    item["scores"][i] = 0

        # Process each sample in the item
        for i in range(len(item["tokens"])):
            seq_len = len(item["tokens"][i])
            lengths.append(math.ceil((seq_len - 1) / good_multiple) * good_multiple)

            # Create labels with padding (-100 for masked positions)
            label_item = np.concatenate(
                [
                    np.array(item["masks"][i]),
                    np.full(
                        max(0, token_setup_len - seq_len),
                        -100,
                        dtype=np.int32,
                    ),
                ]
            )

            # Pad tokens
            item["tokens"][i] = np.concatenate(
                [
                    np.array(item["tokens"][i]),
                    np.zeros(
                        max(0, token_setup_len - seq_len),
                        dtype=np.int32,
                    ),
                ]
            )

            input_ids.append(item["tokens"][i][:-1])  # Remove last for causal
            labels.append(label_item[1:])  # Shift by 1 for causal
            advantages.append(item["scores"][i])

            # Extract and pad inference logprobs to match labels shape
            # IMPORTANT: inference_logprobs is ALREADY ALIGNED with tokens/masks:
            # - 1.0 for prompt tokens (masked positions)
            # - actual negative logprobs for generated tokens
            # We just need to pad to match the sequence length
            if extract_inference_logprobs and "inference_logprobs" in item:
                if i < len(item["inference_logprobs"]):
                    raw_logprobs = np.array(
                        item["inference_logprobs"][i], dtype=np.float32
                    )
                    has_any_logprobs = True

                    # Create padded logprobs array matching token_setup_len
                    # Fill with 1.0 (the masked token placeholder value) for padding
                    padded_logprobs = np.full(token_setup_len, 1.0, dtype=np.float32)

                    # Copy raw_logprobs directly - they're already aligned with tokens
                    n_to_copy = min(len(raw_logprobs), token_setup_len)
                    padded_logprobs[:n_to_copy] = raw_logprobs[:n_to_copy]

                    # Shift by 1 to match causal label shift
                    inference_logprobs_padded.append(padded_logprobs[1:])
                else:
                    # No logprobs for this sample, use 1.0
                    inference_logprobs_padded.append(
                        np.full(token_setup_len - 1, 1.0, dtype=np.float32)
                    )
            elif extract_inference_logprobs:
                # No inference_logprobs in item, use 1.0
                inference_logprobs_padded.append(
                    np.full(token_setup_len - 1, 1.0, dtype=np.float32)
                )

            # Extract optional distillation arrays.
            # Format:
            #   distill_token_ids[seq][pos][top_k], distill_logprobs[seq][pos][top_k]
            seq_token_ids = None
            seq_logprobs = None
            if (
                isinstance(item.get("distill_token_ids"), list)
                and isinstance(item.get("distill_logprobs"), list)
                and i < len(item["distill_token_ids"])
                and i < len(item["distill_logprobs"])
            ):
                seq_token_ids = item["distill_token_ids"][i]
                seq_logprobs = item["distill_logprobs"][i]

            if seq_token_ids is not None and seq_logprobs is not None:
                has_any_distill = True
                seq_target_len = token_setup_len - 1
                padded_ids = list(seq_token_ids[:seq_target_len]) + [
                    [] for _ in range(max(0, seq_target_len - len(seq_token_ids)))
                ]
                padded_lps = list(seq_logprobs[:seq_target_len]) + [
                    [] for _ in range(max(0, seq_target_len - len(seq_logprobs)))
                ]
                distill_token_ids_padded.append(padded_ids)
                distill_logprobs_padded.append(padded_lps)
            else:
                seq_target_len = token_setup_len - 1
                distill_token_ids_padded.append([[] for _ in range(seq_target_len)])
                distill_logprobs_padded.append([[] for _ in range(seq_target_len)])

            # Extract temperature (priority: override > generation_params > group_overrides > 1.0)
            t = 1.0
            if (
                item.get("overrides")
                and i < len(item["overrides"])
                and isinstance(item["overrides"][i], dict)
                and ("temperature" in item["overrides"][i])
            ):
                t = float(item["overrides"][i]["temperature"])
            elif item.get("generation_params") and (
                "temperature" in item["generation_params"]
            ):
                t = float(item["generation_params"]["temperature"])
            elif item.get("group_overrides") and (
                "temperature" in item["group_overrides"]
            ):
                t = float(item["group_overrides"]["temperature"])
            temperatures.append(t)

    # Batch the data
    token_batches = []
    label_batches = []
    advantage_batches = []
    temperature_batches = []
    inference_logprob_batches = []
    distill_token_id_batches = []
    distill_logprob_batches = []

    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))

        token_batches.append(torch.tensor(np.stack(input_ids[start:end], axis=0)))
        label_batches.append(torch.tensor(np.stack(labels[start:end], axis=0)))
        advantage_batches.append(
            torch.tensor(np.stack(advantages[start:end], axis=0)).view(-1, 1)
        )
        temperature_batches.append(
            torch.tensor(np.array(temperatures[start:end], dtype=np.float32)).view(
                -1, 1, 1
            )
        )

        # Batch inference logprobs (same shape as labels)
        if extract_inference_logprobs and inference_logprobs_padded:
            inference_logprob_batches.append(
                torch.tensor(np.stack(inference_logprobs_padded[start:end], axis=0))
            )
        if distill_token_ids_padded:
            distill_token_id_batches.append(distill_token_ids_padded[start:end])
            distill_logprob_batches.append(distill_logprobs_padded[start:end])

    # Return inference logprob batches if we have any real logprobs
    final_logprob_batches = (
        inference_logprob_batches
        if (has_any_logprobs and inference_logprob_batches)
        else None
    )
    final_distill_token_id_batches = (
        distill_token_id_batches if (has_any_distill and distill_token_id_batches) else None
    )
    final_distill_logprob_batches = (
        distill_logprob_batches if (has_any_distill and distill_logprob_batches) else None
    )

    return (
        token_batches,
        label_batches,
        advantage_batches,
        temperature_batches,
        final_logprob_batches,
        final_distill_token_id_batches,
        final_distill_logprob_batches,
    )


def get_data(
    batch_size: int,
    seq_len: int,
    atropos_url: str = "http://localhost:8000",
    extract_inference_logprobs: bool = True,
) -> Tuple[
    List[
        Tuple[
            List[torch.Tensor],  # token_batches
            List[torch.Tensor],  # label_batches
            List[torch.Tensor],  # advantage_batches
            List[torch.Tensor],  # temperature_batches
            Optional[List[torch.Tensor]],  # inference_logprob_batches
            Optional[List[list]],  # distill_token_id_batches
            Optional[List[list]],  # distill_logprob_batches
        ]
    ],
    None,  # Legacy return (no longer used)
]:
    """
    Fetch and process training data from the Atropos API.

    Continuously polls the API until data is available, then processes
    all available batches.

    Args:
        batch_size: Size of each training batch
        seq_len: Maximum sequence length (for reference, not used directly)
        atropos_url: URL of the Atropos API server
        extract_inference_logprobs: Whether to extract inference logprobs for GRPO loss

    Returns:
        Tuple of (batches, None)
        - batches: List of processed batch tuples, each containing:
          (
              token_batches,
              label_batches,
              advantage_batches,
              temperature_batches,
              inference_logprob_batches,
              distill_token_id_batches,
              distill_logprob_batches,
          )
        - inference_logprob_batches are aligned with labels for proper GRPO loss computation
    """
    batches = []
    _logged_logprob_warning = False

    while True:
        data = get_batch(url=atropos_url)

        if data["batch"] is not None:
            # DEBUG: Check if inference_logprobs exists in the data
            if not _logged_logprob_warning:
                has_logprobs = any(
                    "inference_logprobs" in item for item in data["batch"]
                )
                if has_logprobs:
                    # Check if they're non-empty
                    sample_item = next(
                        (
                            item
                            for item in data["batch"]
                            if "inference_logprobs" in item
                        ),
                        None,
                    )
                    if sample_item and sample_item.get("inference_logprobs"):
                        sample_lp = (
                            sample_item["inference_logprobs"][0]
                            if sample_item["inference_logprobs"]
                            else []
                        )
                        print(
                            f"    [Data] ✓ inference_logprobs found in batch (sample len: {len(sample_lp)})"
                        )
                    else:
                        print(
                            "    [Data] ⚠ inference_logprobs key exists but is empty!"
                        )
                else:
                    print("    [Data] ⚠ NO inference_logprobs in batch data!")
                    print(
                        f"    [Data] Keys in first item: {list(data['batch'][0].keys())}"
                    )
                _logged_logprob_warning = True

            # Process and accumulate batches (now includes batched inference logprobs)
            (
                token_batches,
                label_batches,
                adv_batches,
                temp_batches,
                inf_logprob_batches,
                distill_token_id_batches,
                distill_logprob_batches,
            ) = pad_data_to_good_offset(data, batch_size, extract_inference_logprobs)

            # Include inference logprob batches in the tuple
            batches.append(
                (
                    token_batches,
                    label_batches,
                    adv_batches,
                    temp_batches,
                    inf_logprob_batches,
                    distill_token_id_batches,
                    distill_logprob_batches,
                )
            )

        elif len(batches) > 0:
            # Return accumulated batches when no more data
            return batches, None
        else:
            # Wait for data
            time.sleep(1)
