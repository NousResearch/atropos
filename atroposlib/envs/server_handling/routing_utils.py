import hashlib
from typing import List, Optional

def get_prefix_hash(input_ids: List[int], prefix_cutoff: int = 100) -> str:
    """
    Generate a stable MD5 hash for a sequence of tokens.
    Used for consistent session routing to maximize KV cache hits.
    """
    if not input_ids:
        return "empty_prefix"
        
    cutoff = min(len(input_ids), prefix_cutoff)
    prefix_tokens = input_ids[:cutoff]
    
    prefix_bytes = b",".join(str(t).encode('utf-8') for t in prefix_tokens)
    return hashlib.md5(prefix_bytes).hexdigest()

def get_consistent_worker_index(prefix_hash: str, num_workers: int) -> int:
    """Map a hash string to a worker index."""
    if num_workers <= 0:
        return 0
    return int(prefix_hash, 16) % num_workers

