"""
T1 dataset loader — downloads and parses the capitalone/T1 HuggingFace dataset.

Conversations are returned in the format expected by t1_core:
  [{"Role": "assistant"|"user", "Filled_Template": str, "Filled_Plan": str}, ...]
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_tree

logger = logging.getLogger(__name__)

REPO_ID = "capitalone/T1"

# Single-domain only for now (simpler tool defs, shorter conversations)
SINGLE_DOMAINS = ["hotel", "flight", "restaurant", "attraction"]
MULTI_DOMAINS = [
    "flighthotel",
    "hotelrestaurant",
    "hotelattraction",
    "flighthotelrestaurant",
    "flighthotelattraction",
]
ALL_DOMAINS = SINGLE_DOMAINS + MULTI_DOMAINS


def _parse_role(filled_template: str) -> Tuple[str, str]:
    """Extract role and content from 'role: content' format."""
    if filled_template.startswith("assistant:"):
        return "assistant", filled_template[len("assistant:") :].strip()
    elif filled_template.startswith("user:"):
        return "user", filled_template[len("user:") :].strip()
    else:
        # Fallback — try to guess
        return "assistant", filled_template.strip()


def _csv_to_conversations(path: str) -> Dict[int, List[dict]]:
    """Parse a T1 CSV file into a dict of conversation_id → turns."""
    df = pd.read_csv(path)
    conversations = {}

    for conv_id, group in df.groupby("ID"):
        turns = []
        for _, row in group.iterrows():
            template = row["Filled_Template"]
            plan = row["Filled_Plan"]

            # Skip NaN/empty templates
            if pd.isna(template) or str(template).strip() in ("", "nan"):
                continue

            template = str(template)
            role, content = _parse_role(template)

            # Skip empty content after role extraction
            if not content.strip():
                continue

            # NaN plans become empty string
            if pd.isna(plan) or str(plan).strip() == "nan":
                plan = ""
            else:
                plan = str(plan)

            turns.append(
                {
                    "Role": role,
                    "Filled_Template": content,
                    "Filled_Plan": plan,
                }
            )

        # Only keep conversations with at least one user turn
        if turns and any(t["Role"] == "user" for t in turns):
            conversations[int(conv_id)] = turns

    return conversations


def load_t1_dataset(
    domains: Optional[List[str]] = None,
    split: str = "train",
    max_files_per_domain: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[List[dict]]:
    """Load T1 conversations from HuggingFace.

    Args:
        domains: List of domains to load (default: single-domain only)
        split: "train", "test", or "validation"
        max_files_per_domain: Limit files per domain (each has 25, ~15 convos each)
        cache_dir: HF cache directory

    Returns:
        List of conversations, each a list of turn dicts
    """
    if domains is None:
        domains = SINGLE_DOMAINS

    all_conversations = []

    # List all CSV files for the requested domains/split
    repo_files = list(list_repo_tree(REPO_ID, repo_type="dataset", recursive=True))
    csv_files = [
        f.path for f in repo_files if hasattr(f, "size") and f.path.endswith(".csv")
    ]

    for domain in domains:
        prefix = f"{domain}/{split}/"
        domain_files = sorted([f for f in csv_files if f.startswith(prefix)])

        if max_files_per_domain:
            domain_files = domain_files[:max_files_per_domain]

        logger.info(f"Loading {len(domain_files)} files from {domain}/{split}")

        for file_path in domain_files:
            kwargs = {}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir

            local_path = hf_hub_download(
                REPO_ID, file_path, repo_type="dataset", **kwargs
            )
            convos = _csv_to_conversations(local_path)
            all_conversations.extend(convos.values())

    logger.info(f"Loaded {len(all_conversations)} conversations total")
    return all_conversations


def load_t1_split(
    domains: Optional[List[str]] = None,
    max_files_per_domain: Optional[int] = None,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[List[dict]], List[List[dict]]]:
    """Load T1 train conversations and split into train/eval.

    Args:
        domains: Domains to load
        max_files_per_domain: Limit files per domain
        eval_ratio: Fraction of conversations for eval
        seed: Random seed for split

    Returns:
        (train_conversations, eval_conversations)
    """
    conversations = load_t1_dataset(
        domains=domains,
        split="train",
        max_files_per_domain=max_files_per_domain,
    )

    rng = random.Random(seed)
    rng.shuffle(conversations)

    n_eval = max(1, int(len(conversations) * eval_ratio))
    eval_convos = conversations[:n_eval]
    train_convos = conversations[n_eval:]

    logger.info(f"Split: {len(train_convos)} train, {len(eval_convos)} eval")
    return train_convos, eval_convos
