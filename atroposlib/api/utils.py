from typing import Dict, List, Optional, Tuple


def find_groups_summing_to_target(buffer: List[Dict], target_size: int) -> List[int]:
    """
    Find indices of groups in buffer that sum exactly to target_size.
    Prioritizes FIFO order.

    :param buffer: Buffer of groups from same env
    :param target_size: Target sum of group sizes
    :return: List of indices that sum to target_size, or empty list if impossible
    """
    if not buffer:
        return []

    # First try simple FIFO
    current_sum = 0
    indices = []

    for i, group in enumerate(buffer):
        size = len(group["tokens"])
        if current_sum + size <= target_size:
            indices.append(i)
            current_sum += size
            if current_sum == target_size:
                return indices

    # If FIFO doesn't work exactly, try dynamic programming
    # to find any valid combination (still preferring earlier indices)
    n = len(buffer)
    sizes = [len(g["tokens"]) for g in buffer]

    # dp[i][j] = can we make sum j using first i groups
    dp = [[False] * (target_size + 1) for _ in range(n + 1)]
    dp[0][0] = True

    for i in range(1, n + 1):
        for j in range(target_size + 1):
            # Don't take group i-1
            dp[i][j] = dp[i - 1][j]
            # Take group i-1 if possible
            if j >= sizes[i - 1]:
                dp[i][j] = dp[i][j] or dp[i - 1][j - sizes[i - 1]]

    if not dp[n][target_size]:
        return []

    # Backtrack to find indices, preferring earlier ones
    indices = []
    j = target_size
    for i in range(n, 0, -1):
        if j >= sizes[i - 1] and dp[i - 1][j - sizes[i - 1]]:
            indices.append(i - 1)
            j -= sizes[i - 1]

    return sorted(indices)  # Return in FIFO order


def grab_exact_from_heterogeneous_queue(
    queue: List[Dict[str, List]], batch_size: int
) -> Tuple[Optional[List], List]:
    """
    Grabs a batch of exactly batch_size sequences from a queue of items with different group sizes.

    Each item in the queue has a 'tokens' field containing a list of sequences.
    e.g. queue = [{"tokens": [[1, 2, 3],[4, 5, 6, 7, 8]]}, {"tokens": [[9, 10]]}]
    where the first item has 2 sequences and the second has 1 sequence.

    This function returns a batch containing exactly batch_size sequences total, and the remaining queue.

    Because all groups are a common denominator of the batchsize, and all groups are a power of 2,
    we can simplify a bit by assuming we can grab groups of groups to be equal to the maximum group size.
    Note that we cannot split items, so we must take the entire item with all its sequences.

    There may be a more efficient clearing mechanism by grouping these smaller groups heterogeneously, but
    forcing them all into powers of two groups is a simple way to ensure we can grab a batch of the correct size.

    :param queue: List of items, each with a 'tokens' field containing sequences
    :param batch_size: Target number of sequences for the batch
    :return: batch, new_queue
    """

    # Pass 1: precompute group sizes, total sequences and early exit if not enough sequences.
    total_groups = len(queue)
    if total_groups == 0:
        return None, queue

    group_sizes = []
    lengths = []
    total_sequences = 0
    max_group_size = 0

    for item in queue:
        length = len(item["tokens"])  # Number of sequences in this group
        lengths.append(length)
        group_sizes.append(length)
        total_sequences += length
        if length > max_group_size:
            max_group_size = length

    if total_sequences < batch_size:
        return None, queue

    group_sizes_set = set(group_sizes)
    group_batching_storage = {size: [] for size in group_sizes_set}

    # Index into the queue and batch related indices into "packs"
    potential_batch_indices = []
    for i, group_size in enumerate(group_sizes):
        group_batching_storage[group_size].append(i)
        if len(group_batching_storage[group_size]) * group_size == max_group_size:
            potential_batch_indices.extend(group_batching_storage[group_size])
            group_batching_storage[group_size].clear()  # much faster than = []

    # Calculate total sequences in potential batch only once (avoid repeated sums)
    potential_batch_sequences_total = sum(lengths[i] for i in potential_batch_indices)
    if potential_batch_sequences_total < batch_size:
        return None, queue

    # Batch selection
    batch = []
    batch_indices = []
    running_seqs = 0
    for idx in potential_batch_indices:
        group = queue[idx]
        batch.append(group)
        batch_indices.append(idx)
        running_seqs += lengths[idx]
        if running_seqs == batch_size:
            break
        elif running_seqs > batch_size:
            # Should never happen due to problem constraints, but sanity check
            return None, queue

    if running_seqs != batch_size:
        return None, queue

    # Construct new_queue with a single pass, using a set for O(1) lookup
    batch_indices_set = set(batch_indices)
    new_queue = [item for i, item in enumerate(queue) if i not in batch_indices_set]
    return batch, new_queue


def grab_batch_with_minimum_allocations(
    queue: List[Dict[str, any]], batch_size: int, env_configs: List[Dict[str, any]]
) -> Tuple[Optional[List], List]:
    """
    Grabs a batch from the queue while respecting minimum allocation requirements for environments.
    This function works with groups where each group contains multiple sequences.

    :param queue: List of groups with env_id field and sequences (stored in 'tokens' field)
    :param batch_size: Target batch size in sequences
    :param env_configs: List of environment configs with min_batch_allocation field
    :return: batch, new_queue
    """
    if not queue:
        return None, queue

    # Build env_id to min allocation mapping
    env_min_allocations = {}
    for env in env_configs:
        if env.get("connected", False) and env.get("min_batch_allocation") is not None:
            env_min_allocations[env["registered_id"]] = env["min_batch_allocation"]

    # If no minimum allocations, fall back to original function
    if not env_min_allocations:
        return grab_exact_from_heterogeneous_queue(queue, batch_size)

    # First, find the maximum group size across all items
    max_group_size = 0
    for item in queue:
        group_size = len(item.get("tokens", []))
        if group_size > max_group_size:
            max_group_size = group_size

    # Group queue items by env_id and calculate which can form complete packs
    items_by_env = {}
    packable_items_by_env = {}

    for i, item in enumerate(queue):
        env_id = item.get("env_id")
        group_size = len(item.get("tokens", []))

        if env_id is not None:
            if env_id not in items_by_env:
                items_by_env[env_id] = {}
                packable_items_by_env[env_id] = []

            if group_size not in items_by_env[env_id]:
                items_by_env[env_id][group_size] = []

            items_by_env[env_id][group_size].append((i, item, group_size))

            # Check if we can form a complete pack
            items_of_size = items_by_env[env_id][group_size]
            if len(items_of_size) * group_size == max_group_size:
                # We have a complete pack!
                packable_items_by_env[env_id].extend(items_of_size)
                items_by_env[env_id][group_size] = []

    # Calculate minimum sequences needed per env
    min_sequences_per_env = {}
    total_min_sequences = 0
    for env_id, min_proportion in env_min_allocations.items():
        min_sequences = int(batch_size * min_proportion)
        if min_sequences > 0:
            # Check if this env has any items in the queue at all
            if env_id not in items_by_env:
                # This env has a minimum but no items - can't satisfy minimum
                return None, queue
            # Check if this env has any packable items
            if env_id not in packable_items_by_env or not packable_items_by_env[env_id]:
                # This env has items but no packable items - can't satisfy minimum
                return None, queue
            min_sequences_per_env[env_id] = min_sequences
            total_min_sequences += min_sequences

    # If minimums exceed batch size, scale them down proportionally
    if total_min_sequences > batch_size:
        scale_factor = batch_size / total_min_sequences
        for env_id in min_sequences_per_env:
            # Ensure at least one pack from each env with minimum
            if packable_items_by_env.get(env_id):
                min_group_size = min(g[2] for g in packable_items_by_env[env_id])
                min_sequences_per_env[env_id] = max(
                    min_group_size,
                    int(min_sequences_per_env[env_id] * scale_factor),
                )

    # Build batch ensuring minimums are met
    batch = []
    batch_indices = []
    sequences_taken_per_env = {env_id: 0 for env_id in packable_items_by_env}
    total_sequences = 0

    # First pass: satisfy minimum requirements using packable items
    for env_id, min_sequences in min_sequences_per_env.items():
        if env_id in packable_items_by_env:
            # Take packable items in order (FIFO)
            for idx, item, group_size in packable_items_by_env[env_id]:
                if sequences_taken_per_env[env_id] >= min_sequences:
                    break
                if total_sequences + group_size <= batch_size:
                    batch.append(item)
                    batch_indices.append(idx)
                    sequences_taken_per_env[env_id] += group_size
                    total_sequences += group_size

    # Second pass: fill remaining slots with packable items from any env
    if total_sequences < batch_size:
        # Collect all remaining packable items in queue order
        all_packable = []
        for i, item in enumerate(queue):
            if i not in batch_indices:
                # Check if this item is in any env's packable list
                env_id = item.get("env_id")
                if env_id in packable_items_by_env:
                    for idx, packable_item, size in packable_items_by_env[env_id]:
                        if idx == i:
                            all_packable.append((i, item, size))
                            break

        # Take packable items in queue order
        for idx, item, group_size in all_packable:
            if total_sequences + group_size <= batch_size:
                batch.append(item)
                batch_indices.append(idx)
                total_sequences += group_size
            if total_sequences == batch_size:
                break

    # If we couldn't form a full batch, return None
    if total_sequences != batch_size:
        return None, queue

    # Construct new queue
    batch_indices_set = set(batch_indices)
    new_queue = [item for i, item in enumerate(queue) if i not in batch_indices_set]
    return batch, new_queue
