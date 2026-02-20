"""
RedCode Data Loader

Loads the RedCode-Exec dataset for training LLMs to recognize and refuse
risky code execution. Dataset from NeurIPS'24 paper.

Dataset: AI-secure/RedCode (GitHub)
Paper: "RedCode: Risky Code Execution and Generation Benchmark for Code Agents"
"""

import json
import random
import ssl
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Cache directory for downloaded data
CACHE_DIR = Path(__file__).parent / ".cache"

# Create SSL context that handles certificate issues on some systems (esp. macOS)
try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    # If certifi is not installed, use unverified context as fallback
    SSL_CONTEXT = ssl._create_unverified_context()

# RedCode GitHub base URL for raw files
REDCODE_BASE_URL = "https://raw.githubusercontent.com/AI-secure/RedCode/main"

# Available RedCode-Exec dataset files by category
# Files are organized by category index (vulnerability type)
PYTHON_FILES = [
    f"dataset/RedCode-Exec/py2text_dataset_json/index{i}_30_codes_full.json"
    for i in range(1, 26)
    if i != 24  # index24 seems to be missing based on common patterns
]

BASH_FILES = [
    f"dataset/RedCode-Exec/bash2text_dataset_json/index{i}_30_codes_full_upd.json"
    for i in range(1, 26)
    if i not in [10, 11, 17, 19, 24]  # Some indices may be missing
]

# Vulnerability categories from RedCode paper (mapped to indices)
VULNERABILITY_CATEGORIES = {
    1: "data_exfiltration",
    2: "privilege_escalation",
    3: "file_system_manipulation",
    4: "network_attack",
    5: "process_manipulation",
    6: "crypto_attack",
    7: "system_modification",
    8: "misc",
}


def download_file(url: str, cache_path: Path) -> str:
    """Download a file from URL and cache it locally."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Atropos RedCode Env)"}
        )
        with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as response:
            content = response.read().decode("utf-8")

        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)

        return content
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def load_redcode_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a specific RedCode-Exec dataset file."""
    url = f"{REDCODE_BASE_URL}/{file_path}"
    cache_path = CACHE_DIR / file_path.replace("/", "_")

    try:
        content = download_file(url, cache_path)
        data = json.loads(content)
    except Exception:
        # Return empty list if file can't be loaded
        return []

    # Normalize data structure
    result = []
    for item in data:
        normalized = {
            "prompt": item.get("Code", item.get("code", "")),
            "text_summary": item.get("Text_summary", ""),
            "text_details": item.get("Text_details", ""),
            "language": item.get("Language", "unknown").lower(),
            "index": item.get("Index", ""),
            "is_unsafe": True,  # All RedCode-Exec items are unsafe code
            "source_file": file_path,
        }
        # Extract category from index (e.g., "1_2" -> category 1)
        if normalized["index"] and "_" in str(normalized["index"]):
            try:
                cat_idx = int(str(normalized["index"]).split("_")[0])
                normalized["category"] = VULNERABILITY_CATEGORIES.get(
                    cat_idx, f"category_{cat_idx}"
                )
            except ValueError:
                normalized["category"] = "misc"
        else:
            normalized["category"] = "misc"

        result.append(normalized)

    return result


def load_redcode_exec(
    languages: Optional[List[Literal["python", "bash"]]] = None,
    limit: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load RedCode-Exec dataset with filtering options.

    Args:
        languages: Languages to include. Default: ["python", "bash"]
        limit: Maximum number of examples to return
        seed: Random seed for shuffling
        verbose: Print loading progress

    Returns:
        List of test case dictionaries with fields:
        - prompt: The risky code
        - text_summary: Summary of what the code does
        - text_details: Detailed description
        - is_unsafe: Always True (all RedCode-Exec items are unsafe)
        - language: python or bash
        - category: Vulnerability category
    """
    if languages is None:
        languages = ["python", "bash"]

    data = []
    total_files = 0
    loaded_files = 0

    # Load Python files
    if "python" in languages:
        for file_path in PYTHON_FILES:
            total_files += 1
            try:
                items = load_redcode_file(file_path)
                if items:
                    data.extend(items)
                    loaded_files += 1
                    if verbose:
                        print(f"Loaded {len(items)} Python examples from {file_path}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load {file_path}: {e}")

    # Load Bash files
    if "bash" in languages:
        for file_path in BASH_FILES:
            total_files += 1
            try:
                items = load_redcode_file(file_path)
                if items:
                    data.extend(items)
                    loaded_files += 1
                    if verbose:
                        print(f"Loaded {len(items)} Bash examples from {file_path}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load {file_path}: {e}")

    if verbose:
        print(
            f"Loaded {len(data)} total examples from {loaded_files}/{total_files} files"
        )

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(data)

    if limit:
        data = data[:limit]

    return data


def load_redcode_split(
    split: Literal["train", "test"] = "train",
    train_ratio: float = 0.9,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Load RedCode-Exec with train/test split.

    Since RedCode doesn't have official splits, we create them deterministically.
    """
    all_data = load_redcode_exec(**kwargs)

    # Deterministic split based on hash of code
    train_data = []
    test_data = []

    for item in all_data:
        code_hash = hash(item.get("prompt", str(item)))
        if (code_hash % 100) < (train_ratio * 100):
            train_data.append(item)
        else:
            test_data.append(item)

    if split == "train":
        print(f"Train split: {len(train_data)} examples")
        return train_data
    else:
        print(f"Test split: {len(test_data)} examples")
        return test_data


def get_dataset_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics about the loaded dataset."""
    stats = {
        "total": len(data),
        "by_language": {},
        "by_category": {},
    }

    for item in data:
        lang = item.get("language", "unknown")
        cat = item.get("category", "unknown")

        if lang not in stats["by_language"]:
            stats["by_language"][lang] = 0
        stats["by_language"][lang] += 1

        if cat not in stats["by_category"]:
            stats["by_category"][cat] = 0
        stats["by_category"][cat] += 1

    return stats


if __name__ == "__main__":
    print("Testing RedCode loader...")
    print()

    print("--- Loading examples (limit=50) ---")
    data = load_redcode_exec(limit=50, verbose=True)

    if data:
        print()
        print("--- Sample examples ---")
        for i, item in enumerate(data[:3]):
            print(f"\nExample {i + 1}:")
            print(f"  Language: {item.get('language', 'unknown')}")
            print(f"  Category: {item.get('category', 'unknown')}")
            print(f"  Index: {item.get('index', 'unknown')}")
            prompt = str(item.get("prompt", ""))[:100].replace("\n", " ")
            print(f"  Prompt: {prompt}...")
            summary = str(item.get("text_summary", ""))[:100]
            print(f"  Summary: {summary}...")

        print()
        print("--- Dataset Stats ---")
        stats = get_dataset_stats(data)
        print(f"  Total: {stats['total']}")
        print(f"  By language: {stats['by_language']}")
        print(f"  By category: {stats['by_category']}")
    else:
        print("No data loaded. Check network connection and file paths.")
