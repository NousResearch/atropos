"""
Dataset loader for the Solidity Smart Contract Vulnerability Dataset.

Loads and preprocesses the darkknight25/Smart_Contract_Vulnerability_Dataset
from HuggingFace for use in the Solidity audit RL environment.

The upstream JSONL file contains some malformed rows, so we download the raw
file and parse line-by-line, skipping invalid entries.
"""

import json
import random
from typing import Dict, List, Tuple

from huggingface_hub import hf_hub_download

DATASET_REPO = "darkknight25/Smart_Contract_Vulnerability_Dataset"
DATASET_FILENAME = "smartcontract_vuleablities _dataset.jsonl"

REQUIRED_KEYS = {"code_snippet", "category"}


def normalize_category(category: str) -> str:
    """Normalize vulnerability category to lowercase with underscores."""
    return category.strip().lower().replace(" ", "_").replace("-", "_")


def preprocess_entry(entry: Dict) -> Dict:
    """Normalize a single dataset entry to a consistent format.

    Returns:
        Dict with keys: code_snippet, category, description, severity, vulnerable
    """
    return {
        "code_snippet": entry["code_snippet"].strip(),
        "category": normalize_category(entry["category"]),
        "description": entry.get("description", "").strip(),
        "severity": entry.get("severity", "unknown").strip().lower(),
        "vulnerable": bool(entry.get("vulnerable", True)),
    }


def _load_jsonl_robust(filepath: str) -> List[Dict]:
    """Load a JSONL file, skipping malformed rows."""
    entries = []
    skipped = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and REQUIRED_KEYS.issubset(obj.keys()):
                    entries.append(obj)
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"Warning: skipped {skipped} malformed rows in dataset")
    return entries


def load_vulnerability_dataset(
    seed: int = 42,
    test_ratio: float = 0.2,
) -> Tuple[List[Dict], List[Dict]]:
    """Load and split the vulnerability dataset into train and test sets.

    Downloads the raw JSONL from HuggingFace Hub and parses line-by-line
    to handle malformed rows in the upstream dataset.

    Args:
        seed: Random seed for reproducible splitting.
        test_ratio: Fraction of data to use for testing.

    Returns:
        Tuple of (train_data, test_data) where each is a list of preprocessed dicts.
    """
    filepath = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILENAME,
        repo_type="dataset",
    )

    raw_entries = _load_jsonl_robust(filepath)
    print(f"Loaded {len(raw_entries)} valid entries from dataset")

    data = [preprocess_entry(e) for e in raw_entries]

    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    return train_data, test_data
