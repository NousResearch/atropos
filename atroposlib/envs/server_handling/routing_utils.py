import hashlib
from typing import List, Optional

def get_prefix_hash(input_ids: List[int], prefix_cutoff: int = 100) -> str:
    """
    Generate a stable hash for a sequence of tokens to use in session pinning.
    
    This mimics the SGLang Model Gateway (SMG) approach which hashes the start
    of a conversation to ensure identical prefixes route to the same backend
    worker to maximize KV cache hits.
    
    Args:
        input_ids: Full sequence of token IDs.
        prefix_cutoff: How many tokens to include in the hash. 100 is usually 
                       enough to capture the unique start of a system prompt.
    Returns:
        A deterministic MD5 hash string of the prefix.
    """
    if not input_ids:
        return "empty_prefix"
        
    cutoff = min(len(input_ids), prefix_cutoff)
    prefix_tokens = input_ids[:cutoff]
    
    # Convert token list to bytes
    prefix_bytes = b",".join(str(t).encode('utf-8') for t in prefix_tokens)
    return hashlib.md5(prefix_bytes).hexdigest()

def get_consistent_worker_index(prefix_hash: str, num_workers: int) -> int:
    """
    Map a hash string to a worker index using the hash integer value.
    This provides basic consistent hashing.
    
    Args:
        prefix_hash: The MD5 hash string.
        num_workers: Total number of active workers.
    Returns:
        Index of the target worker in the server pool.
    """
    if num_workers <= 0:
        return 0
    # Convert hex string to integer and modulo by worker count
    hash_int = int(prefix_hash, 16)
    return hash_int % num_workers
