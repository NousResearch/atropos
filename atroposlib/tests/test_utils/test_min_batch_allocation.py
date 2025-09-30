"""Tests for minimum batch allocation functionality."""

import random

from atroposlib.api.utils import grab_batch_with_minimum_allocations


class TestMinBatchAllocation:
    """Test cases for minimum batch allocation feature."""

    def test_basic_minimum_allocation(self):
        """Test that basic minimum allocations are respected."""
        # Each item represents a group with multiple token sequences
        queue = [
            {
                "tokens": [[1, 2], [3, 4]],
                "env_id": 0,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.5, 0.6],
            },  # 2 groups
            {
                "tokens": [[5, 6], [7, 8]],
                "env_id": 1,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.7, 0.8],
            },  # 2 groups
            {
                "tokens": [[9, 10], [11, 12]],
                "env_id": 0,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.5, 0.6],
            },  # 2 groups
            {
                "tokens": [[13, 14], [15, 16]],
                "env_id": 1,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.7, 0.8],
            },  # 2 groups
        ]

        env_configs = [
            {
                "registered_id": 0,
                "connected": True,
                "min_batch_allocation": 0.25,
            },  # 25% = 2 groups min
            {
                "registered_id": 1,
                "connected": True,
                "min_batch_allocation": 0.5,
            },  # 50% = 4 groups min
        ]

        batch_size = 8  # 8 token groups total
        batch, new_queue = grab_batch_with_minimum_allocations(
            queue, batch_size, env_configs
        )

        assert batch is not None

        # Count groups (not items) per environment
        env_groups = {}
        total_groups = 0
        for item in batch:
            env_id = item["env_id"]
            groups = len(item["tokens"])
            env_groups[env_id] = env_groups.get(env_id, 0) + groups
            total_groups += groups

        assert total_groups == batch_size

        # Env 1 should have at least 50% (4 groups)
        assert env_groups.get(1, 0) >= 4
        # Env 0 should have at least 25% (2 groups)
        assert env_groups.get(0, 0) >= 2

    def test_no_minimum_allocation_fallback(self):
        """Test fallback to original function when no minimums specified."""
        queue = [
            {"tokens": [[1, 2]], "env_id": 0, "masks": [[1, 1]], "scores": [0.5, 0.6]},
            {"tokens": [[3, 4]], "env_id": 1, "masks": [[1, 1]], "scores": [0.7, 0.8]},
            {"tokens": [[5, 6]], "env_id": 0, "masks": [[1, 1]], "scores": [0.5, 0.6]},
            {"tokens": [[7, 8]], "env_id": 1, "masks": [[1, 1]], "scores": [0.7, 0.8]},
        ]

        env_configs = [
            {"registered_id": 0, "connected": True},
            {"registered_id": 1, "connected": True},
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 4, env_configs)

        # Should still form a batch using original logic
        assert batch is not None
        assert len(new_queue) < len(queue)

    def test_conflicting_minimums_scale_down(self):
        """Test that conflicting minimums > 100% are scaled down."""
        queue = [
            {
                "tokens": [[1, 2], [3, 4]],
                "env_id": 0,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.5, 0.6],
            },  # 2 groups
            {
                "tokens": [[5, 6], [7, 8]],
                "env_id": 1,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.7, 0.8],
            },  # 2 groups
        ]

        env_configs = [
            {"registered_id": 0, "connected": True, "min_batch_allocation": 0.7},  # 70%
            {
                "registered_id": 1,
                "connected": True,
                "min_batch_allocation": 0.6,
            },  # 60% = 130% total
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 4, env_configs)

        # Should still form a batch with scaled allocations
        assert batch is not None
        assert len(batch) == 2  # Both items needed to form batch of 4 groups

    def test_missing_env_in_queue(self):
        """Test handling when an env has minimum but no items in queue."""
        queue = [
            {
                "tokens": [[1, 2], [3, 4]],
                "env_id": 0,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.5, 0.6],
            },  # 2 groups
            {
                "tokens": [[5, 6], [7, 8]],
                "env_id": 0,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.5, 0.6],
            },  # 2 groups
        ]

        env_configs = [
            {"registered_id": 0, "connected": True, "min_batch_allocation": 0.3},
            {
                "registered_id": 1,
                "connected": True,
                "min_batch_allocation": 0.5,
            },  # No items!
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 4, env_configs)

        # Should return None because env 1 has minimum allocation but no items
        assert batch is None

    def test_disconnected_env_ignored(self):
        """Test that disconnected environments are ignored."""
        queue = [
            {"tokens": [[1, 2]], "env_id": 0, "masks": [[1, 1]], "scores": [0.5, 0.6]},
            {"tokens": [[3, 4]], "env_id": 1, "masks": [[1, 1]], "scores": [0.7, 0.8]},
        ]

        env_configs = [
            {"registered_id": 0, "connected": True, "min_batch_allocation": 0.25},
            {
                "registered_id": 1,
                "connected": False,
                "min_batch_allocation": 0.75,
            },  # Disconnected!
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 2, env_configs)

        # Should only consider connected env
        assert batch is not None
        # May include env 1 items but won't enforce its minimum

    def test_mixed_group_sizes(self):
        """Test handling of different group sizes."""
        queue = [
            {"tokens": [[1]], "env_id": 0, "masks": [[1]], "scores": [0.5]},  # size 1
            {
                "tokens": [[2, 3, 4, 5]],
                "env_id": 0,
                "masks": [[1, 1, 1, 1]],
                "scores": [0.6, 0.7, 0.8, 0.9],
            },  # size 4
            {
                "tokens": [[6, 7]],
                "env_id": 1,
                "masks": [[1, 1]],
                "scores": [0.5, 0.6],
            },  # size 2
        ]

        env_configs = [
            {"registered_id": 0, "connected": True, "min_batch_allocation": 0.5},
            {"registered_id": 1, "connected": True, "min_batch_allocation": 0.25},
        ]

        # Try to form batch of size 7 (which would need all items)
        batch, new_queue = grab_batch_with_minimum_allocations(queue, 7, env_configs)

        if batch is not None:
            total_tokens = sum(len(item["tokens"]) for item in batch)
            assert total_tokens == 7

    def test_empty_queue(self):
        """Test handling of empty queue."""
        queue = []
        env_configs = [
            {"registered_id": 0, "connected": True, "min_batch_allocation": 0.5},
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 4, env_configs)

        assert batch is None
        assert new_queue == []

    def test_insufficient_items_for_batch(self):
        """Test when there aren't enough items to form a full batch."""
        queue = [
            {"tokens": [[1, 2]], "env_id": 0, "masks": [[1, 1]], "scores": [0.5, 0.6]},
        ]

        env_configs = [
            {"registered_id": 0, "connected": True, "min_batch_allocation": 0.5},
        ]

        # Request batch size 4 but only have 2 tokens
        batch, new_queue = grab_batch_with_minimum_allocations(queue, 4, env_configs)

        assert batch is None
        assert len(new_queue) == 1  # Original queue unchanged

    def test_heterogeneous_envs(self):
        """Test envs with individual group sizes."""
        # Env 0: all groups have size 2
        # Env 1: all groups have size 4
        # Env 2: all groups have size 8
        queue = []

        # Add items for env 0 (group size 2)
        for i in range(1):
            queue.append(
                {
                    "tokens": [[i * 2, i * 2 + 1] for _ in range(2)],
                    "env_id": 0,
                    "masks": [[1, 1] for _ in range(2)],
                    "scores": [0.5, 0.6],
                }
            )
        # for i in range(1):
        #     queue.append(
        #         {
        #             "tokens": [[i * 2, i * 2 + 1] for _ in range(2)],
        #             "env_id": 1,
        #             "masks": [[1, 1] for _ in range(2)],
        #             "scores": [0.5, 0.6],
        #         }
        #     )
        # Add 3 items of group size 2 to show why greedy packing doesn't work
        for i in range(3):
            queue.append(
                {
                    "tokens": [[i * 2, i * 2 + 1] for _ in range(2)],
                    "env_id": 6,
                    "masks": [[1, 1] for _ in range(2)],
                    "scores": [0.5, 0.6],
                }
            )

        # Add items for env 1 (group size 4)
        for i in range(5):
            queue.append(
                {
                    "tokens": [
                        [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3] for _ in range(4)
                    ],
                    "env_id": 2,
                    "masks": [[1, 1, 1, 1] for _ in range(4)],
                    "scores": [0.7, 0.8, 0.9, 1.0],
                }
            )

        # Add items for env 2 (group size 8)
        for i in range(3):
            queue.append(
                {
                    "tokens": [[i * 8 + j] for j in range(8)],
                    "env_id": 3,
                    "masks": [[1] for _ in range(8)],
                    "scores": [0.5] * 8,
                }
            )

        env_configs = [
            {
                "registered_id": 0,
                "connected": True,
                "min_batch_allocation": 0.1,
            },  # min 2 sequences
            # {
            #     "registered_id": 1,
            #     "connected": True,
            #     "min_batch_allocation": 0.1,
            # },  # min 2 sequences
            # {
            #     "registered_id": 2,
            #     "connected": True,
            #     "min_batch_allocation": 0.25,
            # },  # min 4 sequences
            {
                "registered_id": 3,
                "connected": True,
                "min_batch_allocation": 0.5,
            },  # min 8 sequences
        ]

        batch_size = 16
        batch, new_queue = grab_batch_with_minimum_allocations(
            queue, batch_size, env_configs
        )

        # Since env 0 has min allocation of 10% but can't form any complete packs
        # (has 1 item of size 2, needs 4 to make pack of 8), the function should
        # return None as it cannot satisfy the minimum allocation requirement
        assert batch is None

        # Queue should be unchanged
        assert len(new_queue) == len(queue)

    def test_packing_constraint_enforcement(self):
        """Test that packing to max group size is properly enforced."""
        # Create queue with items that can't form complete packs
        queue = [
            {
                "tokens": [[1, 2]],
                "env_id": 0,
                "masks": [[1, 1]],
                "scores": [0.5],
            },  # size 1
            {
                "tokens": [[3, 4]],
                "env_id": 0,
                "masks": [[1, 1]],
                "scores": [0.5],
            },  # size 1
            {
                "tokens": [[5, 6]],
                "env_id": 0,
                "masks": [[1, 1]],
                "scores": [0.5],
            },  # size 1
            # Need 4 items of size 1 to make a pack of 4, only have 3
            {
                "tokens": [[7, 8], [9, 10], [11, 12], [13, 14]],
                "env_id": 1,
                "masks": [[1, 1]] * 4,
                "scores": [0.7] * 4,
            },  # size 4
        ]

        env_configs = [
            {"registered_id": 0, "connected": True, "min_batch_allocation": 0.25},
            {"registered_id": 1, "connected": True, "min_batch_allocation": None},
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 4, env_configs)

        # Should return None because env 0 can't form complete packs
        assert batch is None

    def test_fifo_order_preservation(self):
        """Test that FIFO order is preserved when forming batches."""
        queue = []
        # Add items with sequential scores to track order
        for i in range(8):
            queue.append(
                {
                    "tokens": [[i, i + 1]],
                    "env_id": 0,
                    "masks": [[1, 1]],
                    "scores": [float(i)],  # Use score to track original order
                }
            )

        env_configs = [
            {"registered_id": 0, "connected": True, "min_batch_allocation": None},
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 4, env_configs)

        if batch is not None:
            # Check that we got the first 4 items (scores 0-3)
            batch_scores = [item["scores"][0] for item in batch]
            assert sorted(batch_scores) == [0.0, 1.0, 2.0, 3.0]

    def test_exact_minimum_boundary(self):
        """Test behavior at exact minimum allocation boundaries."""
        queue = [
            {
                "tokens": [[1, 2], [3, 4]],
                "env_id": 0,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.5, 0.6],
            },
            {
                "tokens": [[5, 6], [7, 8]],
                "env_id": 0,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.5, 0.6],
            },
            {
                "tokens": [[9, 10], [11, 12]],
                "env_id": 1,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.7, 0.8],
            },
            {
                "tokens": [[13, 14], [15, 16]],
                "env_id": 1,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.7, 0.8],
            },
        ]

        env_configs = [
            {
                "registered_id": 0,
                "connected": True,
                "min_batch_allocation": 0.5,
            },  # Exactly 50%
            {
                "registered_id": 1,
                "connected": True,
                "min_batch_allocation": 0.5,
            },  # Exactly 50%
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 8, env_configs)

        assert batch is not None
        env_counts = {}
        for item in batch:
            env_id = item["env_id"]
            count = len(item["tokens"])
            env_counts[env_id] = env_counts.get(env_id, 0) + count

        # Both envs should get exactly 4 sequences (50%)
        assert env_counts[0] == 4
        assert env_counts[1] == 4

    def test_zero_minimum_allocation(self):
        """Test that zero minimum allocation is handled correctly."""
        queue = [
            {"tokens": [[1, 2]], "env_id": 0, "masks": [[1, 1]], "scores": [0.5]},
            {"tokens": [[3, 4]], "env_id": 1, "masks": [[1, 1]], "scores": [0.7]},
        ]

        env_configs = [
            {"registered_id": 0, "connected": True, "min_batch_allocation": 0.0},  # 0%
            {"registered_id": 1, "connected": True, "min_batch_allocation": 0.5},  # 50%
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 2, env_configs)

        # Should work fine - env 0 has no minimum requirement
        assert batch is not None

    def test_multiple_complete_packs(self):
        """Test forming multiple complete packs from same environment."""
        queue = []
        # Add 16 items of size 1 from env 0 (can form 4 complete packs of 4)
        for i in range(16):
            queue.append(
                {
                    "tokens": [[i]],
                    "env_id": 0,
                    "masks": [[1]],
                    "scores": [0.5],
                }
            )

        # Add 2 items of size 4 from env 1
        for i in range(2):
            queue.append(
                {
                    "tokens": [[100 + i * 4, 101 + i * 4, 102 + i * 4, 103 + i * 4]],
                    "env_id": 1,
                    "masks": [[1, 1, 1, 1]],
                    "scores": [0.7],
                }
            )

        env_configs = [
            {
                "registered_id": 0,
                "connected": True,
                "min_batch_allocation": 0.75,
            },  # 12 sequences
            {"registered_id": 1, "connected": True, "min_batch_allocation": None},
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 16, env_configs)

        assert batch is not None
        env_counts = {}
        for item in batch:
            env_id = item["env_id"]
            count = len(item["tokens"])
            env_counts[env_id] = env_counts.get(env_id, 0) + count

        # Env 0 should get at least 12 sequences
        assert env_counts.get(0, 0) >= 12
        assert sum(env_counts.values()) == 16

    def test_no_packable_items(self):
        """Test when no items can form complete packs."""
        queue = [
            {
                "tokens": [[1, 2], [3, 4]],
                "env_id": 0,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.5, 0.6],
            },  # size 2
            {
                "tokens": [[5, 6], [7, 8]],
                "env_id": 0,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.5, 0.6],
            },  # size 2
            {
                "tokens": [[9, 10], [11, 12]],
                "env_id": 0,
                "masks": [[1, 1], [1, 1]],
                "scores": [0.5, 0.6],
            },  # size 2
            # Only 3 items of size 2, need 4 to make complete pack of 8
            {
                "tokens": [
                    [13, 14, 15, 16],
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32],
                    [33, 34, 35, 36],
                    [37, 38, 39, 40],
                    [41, 42, 43, 44],
                ],
                "env_id": 1,
                "masks": [[1, 1, 1, 1]] * 8,
                "scores": [0.7] * 8,
            },  # size 8
        ]

        env_configs = [
            {
                "registered_id": 0,
                "connected": True,
                "min_batch_allocation": 0.25,
            },  # Can't form complete packs
            {"registered_id": 1, "connected": True, "min_batch_allocation": None},
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 8, env_configs)

        # Env 0 can't form complete packs (has 3 items, needs 4)
        assert batch is None

    def test_env_without_items(self):
        """Test env config without any items in queue."""
        queue = [
            {"tokens": [[1, 2]], "env_id": 0, "masks": [[1, 1]], "scores": [0.5]},
            {"tokens": [[3, 4]], "env_id": 0, "masks": [[1, 1]], "scores": [0.5]},
        ]

        env_configs = [
            {"registered_id": 0, "connected": True, "min_batch_allocation": 0.5},
            {
                "registered_id": 1,
                "connected": True,
                "min_batch_allocation": None,
            },  # No items
            {
                "registered_id": 2,
                "connected": True,
                "min_batch_allocation": 0.3,
            },  # No items
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 2, env_configs)

        # Should work - env 2 has no items so its minimum is ignored
        assert batch is not None

    def test_scaling_with_single_env(self):
        """Test scaling behavior with only one env having minimum."""
        queue = []
        for i in range(8):
            queue.append(
                {
                    "tokens": [[i, i + 1]],
                    "env_id": 0,
                    "masks": [[1, 1]],
                    "scores": [0.5],
                }
            )

        env_configs = [
            {
                "registered_id": 0,
                "connected": True,
                "min_batch_allocation": 1.5,
            },  # 150% - impossible
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 4, env_configs)

        # Should scale down to 100% and work
        assert batch is not None
        assert len(batch) == 4

    def test_mixed_null_and_set_minimums(self):
        """Test mix of environments with and without minimum allocations."""
        queue = []
        # Env 0: 4 items of size 2
        for i in range(4):
            queue.append(
                {
                    "tokens": [[i * 2, i * 2 + 1], [i * 2 + 10, i * 2 + 11]],
                    "env_id": 0,
                    "masks": [[1, 1], [1, 1]],
                    "scores": [0.5, 0.6],
                }
            )
        # Env 1: 2 items of size 2
        for i in range(2):
            queue.append(
                {
                    "tokens": [[i * 2 + 20, i * 2 + 21], [i * 2 + 30, i * 2 + 31]],
                    "env_id": 1,
                    "masks": [[1, 1], [1, 1]],
                    "scores": [0.7, 0.8],
                }
            )
        # Env 2: 2 items of size 2 (no minimum)
        for i in range(2):
            queue.append(
                {
                    "tokens": [[i * 2 + 40, i * 2 + 41], [i * 2 + 50, i * 2 + 51]],
                    "env_id": 2,
                    "masks": [[1, 1], [1, 1]],
                    "scores": [0.9, 1.0],
                }
            )

        env_configs = [
            {
                "registered_id": 0,
                "connected": True,
                "min_batch_allocation": 0.4,
            },  # 40% = 6.4 ≈ 6
            {
                "registered_id": 1,
                "connected": True,
                "min_batch_allocation": 0.2,
            },  # 20% = 3.2 ≈ 3
            {
                "registered_id": 2,
                "connected": True,
                "min_batch_allocation": None,
            },  # No minimum
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 16, env_configs)

        assert batch is not None
        env_counts = {}
        for item in batch:
            env_id = item["env_id"]
            count = len(item["tokens"])
            env_counts[env_id] = env_counts.get(env_id, 0) + count

        # Check minimums are satisfied
        assert env_counts.get(0, 0) >= 6  # At least 40% of 16
        assert env_counts.get(1, 0) >= 3  # At least 20% of 16
        assert sum(env_counts.values()) == 16

    def test_random_consistent_group_sizes(self):
        """Random test where each env has a consistent power-of-2 group size."""
        for _ in range(100):
            batch_size = 64 * random.randint(1, 4)
            num_envs = random.randint(2, 4)

            # Assign each env a consistent group size
            env_group_sizes = {}
            for env_id in range(num_envs):
                env_group_sizes[env_id] = 2 ** random.randint(0, 3)  # 1, 2, 4, or 8

            # Create queue
            queue = []
            for env_id in range(num_envs):
                group_size = env_group_sizes[env_id]
                num_items = random.randint(5, 20)
                for i in range(num_items):
                    queue.append(
                        {
                            "tokens": [
                                [env_id * 1000 + i * 10 + j] for j in range(group_size)
                            ],
                            "env_id": env_id,
                            "masks": [[1] for _ in range(group_size)],
                            "scores": [0.5 + env_id * 0.1] * group_size,
                        }
                    )

            # Random minimum allocations that sum to less than 1.0
            env_configs = []
            remaining = 0.9
            for env_id in range(num_envs):
                if env_id == num_envs - 1:
                    min_alloc = remaining
                else:
                    min_alloc = random.uniform(0.1, min(0.4, remaining))
                remaining -= min_alloc

                env_configs.append(
                    {
                        "registered_id": env_id,
                        "connected": True,
                        "min_batch_allocation": min_alloc,
                    }
                )

            batch, new_queue = grab_batch_with_minimum_allocations(
                queue, batch_size, env_configs
            )

            if batch is not None:
                # Verify batch size
                total_sequences = sum(len(item["tokens"]) for item in batch)
                assert total_sequences == batch_size

                # Verify all items from same env have same group size
                env_group_sizes_seen = {}
                for item in batch:
                    env_id = item["env_id"]
                    group_size = len(item["tokens"])
                    if env_id in env_group_sizes_seen:
                        assert group_size == env_group_sizes_seen[env_id]
                    else:
                        env_group_sizes_seen[env_id] = group_size

    def test_queue_dominated_by_one_env(self):
        """Test minimum allocation when one env dominates the queue."""
        queue = []

        # Only env 1 items in queue
        for i in range(100):
            queue.append(
                {
                    "tokens": [[1000 + i, 1001 + i]],
                    "env_id": 1,
                    "masks": [[1, 1]],
                    "scores": [0.7],
                }
            )

        env_configs = [
            {
                "registered_id": 0,
                "connected": True,
                "min_batch_allocation": 0.5,
            },  # 50% but no items!
            {"registered_id": 1, "connected": True, "min_batch_allocation": 0.3},  # 30%
        ]

        batch, new_queue = grab_batch_with_minimum_allocations(queue, 10, env_configs)

        # Should return None because env 0 has minimum allocation but no items
        assert batch is None

        # Test with env 0 having no minimum - should work
        env_configs[0]["min_batch_allocation"] = None
        batch, new_queue = grab_batch_with_minimum_allocations(queue, 10, env_configs)

        assert batch is not None
        env_counts = {}
        for item in batch:
            env_id = item["env_id"]
            count = len(item["tokens"])
            env_counts[env_id] = env_counts.get(env_id, 0) + count

        # Should all be from env 1
        assert env_counts.get(1, 0) == 10
        assert sum(env_counts.values()) == 10


if __name__ == "__main__":
    test = TestMinBatchAllocation()
    test.test_queue_dominated_by_one_env()
