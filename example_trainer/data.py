"""
Data processing utilities for GRPO trainer.

Handles data retrieval from Atropos API, padding, batching,
and advantage normalization.

Also extracts inference logprobs for alignment validation with training logprobs.
"""

import json
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
    List[torch.Tensor], 
    List[torch.Tensor], 
    List[torch.Tensor], 
    List[torch.Tensor],
    Optional[List[np.ndarray]],
]:
    """
    Pad and batch data from the Atropos API.
    
    Processes raw batch data into properly padded tensors suitable for training:
    - Pads token sequences to nearest multiple of 64
    - Normalizes advantage scores
    - Extracts temperature values
    - Optionally extracts inference logprobs for alignment validation
    
    Args:
        data: Raw batch data from Atropos API
        batch_size: Size of each training batch
        extract_inference_logprobs: Whether to extract inference logprobs
        
    Returns:
        Tuple of (token_batches, label_batches, advantage_batches, temperature_batches, inference_logprobs)
        inference_logprobs is None if extract_inference_logprobs=False or no logprobs in data
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
    inference_logprobs_list: List[np.ndarray] = []
    
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
            lengths.append(
                math.ceil((len(item["tokens"][i]) - 1) / good_multiple) * good_multiple
            )
            
            # Create labels with padding
            label_item = np.concatenate([
                np.array(item["masks"][i]),
                np.full(
                    max(0, token_setup_len - len(item["tokens"][i])),
                    -100,
                    dtype=np.int32,
                ),
            ])
            
            # Pad tokens
            item["tokens"][i] = np.concatenate([
                np.array(item["tokens"][i]),
                np.zeros(
                    max(0, token_setup_len - len(item["tokens"][i])),
                    dtype=np.int32,
                ),
            ])
            
            input_ids.append(item["tokens"][i][:-1])  # Remove last for causal
            labels.append(label_item[1:])  # Shift by 1 for causal
            advantages.append(item["scores"][i])
            
            # Extract inference logprobs for alignment validation
            # These come from vLLM during rollout generation
            if extract_inference_logprobs and "inference_logprobs" in item:
                if i < len(item["inference_logprobs"]):
                    inference_logprobs_list.append(
                        np.array(item["inference_logprobs"][i], dtype=np.float32)
                    )
            
            # Extract temperature (priority: override > generation_params > group_overrides > 1.0)
            t = 1.0
            if (
                item.get("overrides")
                and i < len(item["overrides"])
                and isinstance(item["overrides"][i], dict)
                and ("temperature" in item["overrides"][i])
            ):
                t = float(item["overrides"][i]["temperature"])
            elif item.get("generation_params") and ("temperature" in item["generation_params"]):
                t = float(item["generation_params"]["temperature"])
            elif item.get("group_overrides") and ("temperature" in item["group_overrides"]):
                t = float(item["group_overrides"]["temperature"])
            temperatures.append(t)
    
    # Batch the data
    token_batches = []
    label_batches = []
    advantage_batches = []
    temperature_batches = []
    
    for i in range(len(input_ids) // batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        
        token_batches.append(
            torch.tensor(np.stack(input_ids[start:end], axis=0))
        )
        label_batches.append(
            torch.tensor(np.stack(labels[start:end], axis=0))
        )
        advantage_batches.append(
            torch.tensor(np.stack(advantages[start:end], axis=0)).view(-1, 1)
        )
        temperature_batches.append(
            torch.tensor(
                np.array(temperatures[start:end], dtype=np.float32)
            ).view(-1, 1, 1)
        )
    
    # Return inference logprobs if available
    inference_logprobs = inference_logprobs_list if inference_logprobs_list else None
    
    return token_batches, label_batches, advantage_batches, temperature_batches, inference_logprobs


def get_data(
    batch_size: int,
    seq_len: int,
    atropos_url: str = "http://localhost:8000",
    extract_inference_logprobs: bool = True,
) -> Tuple[
    List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]],
    Optional[List[np.ndarray]],
]:
    """
    Fetch and process training data from the Atropos API.
    
    Continuously polls the API until data is available, then processes
    all available batches.
    
    Args:
        batch_size: Size of each training batch
        seq_len: Maximum sequence length (for reference, not used directly)
        atropos_url: URL of the Atropos API server
        extract_inference_logprobs: Whether to extract inference logprobs for alignment
        
    Returns:
        Tuple of (batches, all_inference_logprobs)
        - batches: List of processed batch tuples
        - all_inference_logprobs: List of inference logprob arrays for alignment validation
    """
    batches = []
    all_inference_logprobs: List[np.ndarray] = []
    
    while True:
        data = get_batch(url=atropos_url)
        
        if data["batch"] is not None:
            # Save batch for debugging
            with open("temp.json", "w", encoding="utf-8") as f:
                json.dump(data, f)
            
            # Process and accumulate batches
            token_batches, label_batches, adv_batches, temp_batches, inf_logprobs = \
                pad_data_to_good_offset(data, batch_size, extract_inference_logprobs)
            
            batches.append((token_batches, label_batches, adv_batches, temp_batches))
            
            if inf_logprobs:
                all_inference_logprobs.extend(inf_logprobs)
                
        elif len(batches) > 0:
            # Return accumulated batches when no more data
            return batches, all_inference_logprobs if all_inference_logprobs else None
        else:
            # Wait for data
            time.sleep(1)

