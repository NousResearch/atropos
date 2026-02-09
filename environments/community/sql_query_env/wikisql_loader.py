"""
WikiSQL Data Loader

Downloads and loads the WikiSQL dataset directly from the Salesforce GitHub repository.
This module provides functionality to fetch the dataset, extract it, and load it
in a format compatible with the SQL Query Environment.
"""

import json
import os
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

# WikiSQL data source
WIKISQL_DATA_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"

# SQL operators from WikiSQL lib/query.py
AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


def get_cache_dir() -> Path:
    """Get the cache directory for WikiSQL data."""
    cache_dir = Path(
        os.environ.get("WIKISQL_CACHE_DIR", Path.home() / ".cache" / "wikisql")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_wikisql(cache_dir: Optional[Path] = None, force: bool = False) -> Path:
    """
    Download the WikiSQL dataset from GitHub.

    Args:
        cache_dir: Directory to cache the downloaded data
        force: Force re-download even if already cached

    Returns:
        Path to the extracted data directory
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    data_dir = cache_dir / "data"
    tar_file = cache_dir / "data.tar.bz2"

    # Check if already extracted
    if data_dir.exists() and not force:
        expected_files = [
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "train.db",
            "dev.db",
            "test.db",
        ]
        if all((data_dir / f).exists() for f in expected_files):
            return data_dir

    # Download if needed
    if not tar_file.exists() or force:
        print(f"Downloading WikiSQL dataset from {WIKISQL_DATA_URL}...")
        urlretrieve(WIKISQL_DATA_URL, tar_file)
        print(f"Downloaded to {tar_file}")

    # Extract
    print(f"Extracting to {cache_dir}...")
    with tarfile.open(tar_file, "r:bz2") as tar:
        tar.extractall(cache_dir)
    print("Extraction complete.")

    return data_dir


def build_human_readable_sql(
    table_header: List[str],
    sel: int,
    agg: int,
    conds: List[Tuple[int, int, str]],
) -> str:
    """
    Build a human-readable SQL query from WikiSQL format.

    Args:
        table_header: List of column names
        sel: Index of selected column
        agg: Index of aggregation operator
        conds: List of (column_index, operator_index, condition) tuples

    Returns:
        Human-readable SQL query string
    """
    # Build SELECT clause
    sel_col = table_header[sel] if sel < len(table_header) else f"col{sel}"
    if agg > 0:
        select_clause = f"{AGG_OPS[agg]}({sel_col})"
    else:
        select_clause = sel_col

    sql = f"SELECT {select_clause} FROM data"

    # Build WHERE clause
    if conds:
        where_parts = []
        for col_idx, op_idx, condition in conds:
            col_name = (
                table_header[col_idx]
                if col_idx < len(table_header)
                else f"col{col_idx}"
            )
            op = COND_OPS[op_idx] if op_idx < len(COND_OPS) else "="
            # Quote string conditions
            if isinstance(condition, str):
                where_parts.append(f"{col_name} {op} '{condition}'")
            else:
                where_parts.append(f"{col_name} {op} {condition}")
        sql += " WHERE " + " AND ".join(where_parts)

    return sql


def load_wikisql_split(
    split: str = "train",
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load a WikiSQL dataset split.

    Args:
        split: One of 'train', 'dev', or 'test'
        cache_dir: Directory containing the WikiSQL data

    Returns:
        List of dictionaries with question, table info, and gold SQL
    """
    if cache_dir is None:
        data_dir = download_wikisql()
    else:
        data_dir = cache_dir

    jsonl_file = data_dir / f"{split}.jsonl"
    tables_file = data_dir / f"{split}.tables.jsonl"

    if not jsonl_file.exists():
        raise FileNotFoundError(f"WikiSQL {split} JSONL file not found: {jsonl_file}")
    if not tables_file.exists():
        raise FileNotFoundError(f"WikiSQL {split} tables file not found: {tables_file}")

    # Load all tables from tables.jsonl (has real column headers)
    tables_cache = {}
    with open(tables_file, "r", encoding="utf-8") as f:
        for line in f:
            table = json.loads(line.strip())
            tables_cache[table["id"]] = {
                "header": table["header"],
                "rows": table["rows"],
                "types": table.get("types", ["text"] * len(table["header"])),
            }

    print(f"Loaded {len(tables_cache)} tables for {split} split")

    # Load questions and build dataset
    dataset = []
    skipped = 0
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())

            table_info = tables_cache.get(item["table_id"])
            if not table_info or not table_info["header"]:
                skipped += 1
                continue

            # Build human-readable SQL
            conds = [(c[0], c[1], c[2]) for c in item["sql"]["conds"]]
            gold_sql = build_human_readable_sql(
                table_info["header"],
                item["sql"]["sel"],
                item["sql"]["agg"],
                conds,
            )

            dataset.append(
                {
                    "question": item["question"],
                    "header": table_info["header"],
                    "rows": table_info["rows"],
                    "types": table_info["types"],
                    "gold_sql": gold_sql,
                    "table_id": item["table_id"],
                }
            )

    if skipped > 0:
        print(f"Skipped {skipped} examples with missing tables")

    return dataset


def load_wikisql(cache_dir: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load the full WikiSQL dataset.

    Returns:
        Dictionary with 'train', 'dev', and 'test' splits
    """
    return {
        "train": load_wikisql_split("train", cache_dir),
        "dev": load_wikisql_split("dev", cache_dir),
        "test": load_wikisql_split("test", cache_dir),
    }


if __name__ == "__main__":
    # Test the loader
    print("Testing WikiSQL loader...")
    train = load_wikisql_split("train")
    print(f"Loaded {len(train)} training examples")
    if train:
        print("\nFirst example:")
        print(f"  Question: {train[0]['question']}")
        print(f"  Header: {train[0]['header']}")
        print(f"  Gold SQL: {train[0]['gold_sql']}")
        print(f"  Rows (first 2): {train[0]['rows'][:2]}")
