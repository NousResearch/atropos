"""
Dataset utilities for the BLEUBERI environment.
"""

import logging
import random
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


def load_tulu_dataset(
    dataset_name: str = "allenai/tulu-3-sft-mixture",
    dataset_split: str = "train",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """
    Load the Tulu dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset on Hugging Face
        dataset_split: Dataset split to load (train, validation, test)
        cache_dir: Directory to cache the dataset
        streaming: Whether to stream the dataset
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling

    Returns:
        Loaded dataset
    """
    try:
        if dataset_name.startswith("allenai/") or "/" in dataset_name:
            ds = load_dataset(
                dataset_name,
                split=dataset_split,
                cache_dir=cache_dir,
                streaming=streaming,
            )
        else:
            # Assume it's a local path
            loaded_ds = load_dataset(dataset_name, cache_dir=cache_dir)
            if isinstance(loaded_ds, DatasetDict):
                ds = loaded_ds[dataset_split]
            else:
                ds = loaded_ds

        logger.info(f"Loaded dataset with {len(ds)} examples")

        if shuffle and not streaming:
            ds = ds.shuffle(seed=seed)

        return ds

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Create a small dummy dataset for testing
        dummy_data = []
        for i in range(10):
            dummy_data.append(
                {
                    "id": i,
                    "messages": [
                        {"role": "user", "content": f"Sample prompt {i}"},
                        {"role": "assistant", "content": f"Sample response {i}"},
                    ],
                    "source": "dummy",
                }
            )
        return Dataset.from_list(dummy_data)


def get_user_prompt_from_messages(
    messages: List[Dict[str, str]], example_id: Any = None
) -> Optional[str]:
    """Extract the user prompt from a list of messages."""
    if not messages:
        if example_id:
            logger.warning(f"Messages list is empty for example {example_id}.")
        return None

    for item in messages:
        if item.get("role") == "user":
            return item.get("content")

    if example_id:
        logger.warning(f"No user prompt found in messages for example {example_id}.")
    return None


def get_assistant_response_from_messages(
    messages: List[Dict[str, str]], example_id: Any = None
) -> Optional[str]:
    """Extract the assistant response from a list of messages."""
    if not messages:
        if example_id:
            logger.warning(f"Messages list is empty for example {example_id}.")
        return None

    for item in messages:
        if item.get("role") == "assistant":
            return item.get("content")

    if example_id:
        logger.warning(
            f"No assistant response found in messages for example {example_id}."
        )
    return None


def aggregate_references(
    dataset: Dataset,
    ref_models: List[str] = ["gold"],
) -> List[Dict[str, Any]]:
    """
    Aggregate references from the dataset based on specified reference models.

    Args:
        dataset: Input dataset
        ref_models: List of reference model names (or "gold" for ground truth)

    Returns:
        List of dictionaries with references aggregated
    """
    logger.info(f"Aggregating data from reference models: {ref_models}")

    # Check if "gold" (ground truth) is included in ref_models
    use_gold = "gold" in ref_models
    models_to_use = ref_models.copy()
    if use_gold:
        models_to_use.remove("gold")

    # Create mapping from model names to column names
    model_column_mapping = {}
    available_ref_columns = [
        col for col in dataset.column_names if col.startswith("ref_output_")
    ]

    logger.info(f"Available reference output columns: {available_ref_columns}")

    for model in models_to_use:
        expected_column = f"ref_output_{model}"

        if expected_column in available_ref_columns:
            model_column_mapping[model] = expected_column
            logger.info(f"Mapped model '{model}' to column '{expected_column}'")
        else:
            logger.warning(
                f"Could not find column '{expected_column}' for model '{model}'"
            )

    # Only keep models that have corresponding columns
    models_to_use = [model for model in models_to_use if model in model_column_mapping]
    if not models_to_use and not use_gold:
        raise ValueError("No reference models could be mapped to dataset columns")

    # Filter examples that have all references
    def has_all_references(example):
        if use_gold and (
            example.get("ref_output_gold") is None
            or pd.isna(example.get("ref_output_gold"))
            or str(example.get("ref_output_gold")).strip() == ""
        ):
            return False

        for model in models_to_use:
            col_name = model_column_mapping[model]
            if (
                example.get(col_name) is None
                or pd.isna(example.get(col_name))
                or str(example.get(col_name)).strip() == ""
            ):
                return False

        return True

    dataset_filtered = dataset.filter(has_all_references)
    logger.info(
        f"After filtering for complete references: {len(dataset_filtered)} examples"
    )

    # Aggregate data
    aggregated_data = []
    for example in dataset_filtered:
        example_id = example.get("id", "unknown_id")

        # Get prompt from "prompt" field or messages
        if "prompt" in example and example["prompt"] is not None:
            prompt = example["prompt"]
        else:
            prompt = get_user_prompt_from_messages(example.get("messages"), example_id)

        # Get ground truth from ref_output_gold or messages
        if "ref_output_gold" in example and example["ref_output_gold"] is not None:
            ground_truth = example["ref_output_gold"]
        else:
            ground_truth = get_assistant_response_from_messages(
                example.get("messages"), example_id
            )

        # Collect references
        references = []
        if use_gold:
            references.append(ground_truth)

        for model in models_to_use:
            col_name = model_column_mapping[model]
            references.append(example[col_name])

        # Create aggregated example
        aggregated_example = {
            "id": example_id,
            "source": example.get("source", "unknown"),
            "messages": example.get("messages", []),
            "prompt": prompt,
            "ground_truth": ground_truth,
            "references": references,
        }

        aggregated_data.append(aggregated_example)

    logger.info(f"Aggregated {len(aggregated_data)} examples with specified references")
    return aggregated_data


def select_examples(
    data: List[Dict[str, Any]],
    selection_mode: str = "random",
    num_examples: Optional[int] = None,
    score_field: Optional[str] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Select examples based on the specified mode.

    Args:
        data: List of examples with scores
        selection_mode: Mode for selection (random, easy, medium, hard)
        num_examples: Number of examples to select (if None, selects all)
        score_field: Field name for scores (needed for easy, medium, hard modes)
        seed: Random seed for selection

    Returns:
        List of selected examples
    """
    if not data:
        return []

    if selection_mode == "random":
        if num_examples and len(data) > num_examples:
            random.seed(seed)
            indices = random.sample(range(len(data)), num_examples)
            return [data[i] for i in indices]
        else:
            return data

    elif selection_mode in ["easy", "medium", "hard"]:
        if not score_field:
            logger.warning(
                f"Score field not provided for {selection_mode} selection mode. Using random selection."
            )
            return select_examples(data, "random", num_examples, None, seed)

        # Sort based on scores
        if all(score_field in example for example in data):
            sorted_data = sorted(
                data, key=lambda x: x[score_field], reverse=(selection_mode == "easy")
            )

            if num_examples and len(sorted_data) > num_examples:
                if selection_mode == "medium":
                    # Select from the middle
                    start_idx = (len(sorted_data) - num_examples) // 2
                    return sorted_data[start_idx : start_idx + num_examples]
                else:
                    # Select from the beginning (for both easy and hard, just sorted differently)
                    return sorted_data[:num_examples]
            else:
                return sorted_data
        else:
            logger.warning(
                f"Not all examples have score field '{score_field}'. Using random selection."
            )
            return select_examples(data, "random", num_examples, None, seed)

    else:
        logger.warning(
            f"Unknown selection mode: {selection_mode}. Using random selection."
        )
        return select_examples(data, "random", num_examples, None, seed)
