"""
NL2Bash Data Loader

Loads the NL2SH-ALFA dataset from HuggingFace for training LLMs
to translate natural language to Bash commands.

Dataset: westenfelder/NL2SH-ALFA (NAACL 2025)
Paper: "LLM-Supported Natural Language to Bash Translation"
"""

from typing import Any, Dict, List, Optional

from datasets import load_dataset


def load_nl2bash_split(
    split: str = "train",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load a split of the NL2SH-ALFA dataset.

    Args:
        split: One of 'train' or 'test'
        limit: Optional limit on number of examples to load

    Returns:
        List of dictionaries with:
        - nl: Natural language instruction
        - bash: Gold bash command
        - bash2: Alternative bash command (test only)
        - difficulty: Difficulty level 0-2 (test only)

    Note: NL2SH-ALFA uses the config parameter (not split parameter) to select
    train vs test data. Both configs use split="train" internally.
    """
    if split not in ("train", "test"):
        raise ValueError(f"Split must be 'train' or 'test', got: {split}")

    # Load dataset - config parameter selects train/test, split is always "train"
    print(f"Loading NL2SH-ALFA {split} data from HuggingFace...")
    dataset = load_dataset("westenfelder/NL2SH-ALFA", split, split="train")

    # Convert to list of dicts
    data = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break

        entry = {
            "nl": item["nl"],
            "bash": item["bash"],
        }

        # Test set has additional fields
        if split == "test":
            entry["bash2"] = item.get("bash2")
            entry["difficulty"] = item.get("difficulty", 1)

        data.append(entry)

    print(f"Loaded {len(data)} {split} examples")
    return data


def load_nl2bash() -> Dict[str, List[Dict[str, Any]]]:
    """
    Load the full NL2SH-ALFA dataset.

    Returns:
        Dictionary with 'train' and 'test' splits
    """
    return {
        "train": load_nl2bash_split("train"),
        "test": load_nl2bash_split("test"),
    }


if __name__ == "__main__":
    # Test the loader
    print("Testing NL2Bash loader...")

    print("\n--- Training Set ---")
    train = load_nl2bash_split("train", limit=3)
    for i, item in enumerate(train):
        print(f"\nExample {i+1}:")
        print(f"  NL: {item['nl']}")
        print(f"  Bash: {item['bash']}")

    print("\n--- Test Set ---")
    test = load_nl2bash_split("test", limit=3)
    for i, item in enumerate(test):
        print(f"\nExample {i+1}:")
        print(f"  NL: {item['nl']}")
        print(f"  Bash: {item['bash']}")
        print(f"  Bash2: {item.get('bash2', 'N/A')}")
        print(f"  Difficulty: {item.get('difficulty', 'N/A')}")
