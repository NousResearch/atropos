"""
Data processing utilities for GRPO trainer.

Handles data retrieval from Atropos API, padding, batching,
and advantage normalization.
"""

import json
import math
import time
from typing import List, Tuple

import numpy as np
import torch

from .api import get_batch


def pad_data_to_good_offset(data: dict, batch_size: int) -> Tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
]:
    """
    Pad and batch data from the Atropos API.
    
    Processes raw batch data into properly padded tensors suitable for training:
    - Pads token sequences to nearest multiple of 64
    - Normalizes advantage scores
    - Extracts temperature values
    
    Args:
        data: Raw batch data from Atropos API
        batch_size: Size of each training batch
        
    Returns:
        Tuple of (token_batches, label_batches, advantage_batches, temperature_batches)
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
    
    return token_batches, label_batches, advantage_batches, temperature_batches


def get_data(
    batch_size: int,
    seq_len: int,
    atropos_url: str = "http://localhost:8000",
) -> List[Tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
]]:
    """
    Fetch and process training data from the Atropos API.
    
    Continuously polls the API until data is available, then processes
    all available batches.
    
    Args:
        batch_size: Size of each training batch
        seq_len: Maximum sequence length (for reference, not used directly)
        atropos_url: URL of the Atropos API server
        
    Returns:
        List of processed batch tuples
    """
    batches = []
    
    while True:
        data = get_batch(url=atropos_url)
        
        if data["batch"] is not None:
            # Save batch for debugging
            with open("temp.json", "w", encoding="utf-8") as f:
                json.dump(data, f)
            
            # Process and accumulate batches
            batches.append(pad_data_to_good_offset(data, batch_size))
        elif len(batches) > 0:
            # Return accumulated batches when no more data
            return batches
        else:
            # Wait for data
            time.sleep(1)

