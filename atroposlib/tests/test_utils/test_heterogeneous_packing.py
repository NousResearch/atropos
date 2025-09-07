"""Tests for heterogeneous group packing utility."""

import pytest

from atroposlib.api.utils import find_groups_summing_to_target


class TestHeterogeneousPacking:
    """Test cases for finding groups that sum to target size."""

    def test_simple_fifo_exact_match(self):
        """Test when FIFO order gives exact match."""
        buffer = [
            {"tokens": [[1, 2]], "scores": [0.5]},  # size 1
            {"tokens": [[3, 4], [5, 6]], "scores": [0.6, 0.7]},  # size 2
            {"tokens": [[7, 8]], "scores": [0.8]},  # size 1
        ]

        indices = find_groups_summing_to_target(buffer, 4)
        assert indices == [0, 1, 2]

    def test_fifo_partial_match(self):
        """Test when FIFO can match with subset."""
        buffer = [
            {"tokens": [[1, 2], [3, 4]], "scores": [0.5, 0.6]},  # size 2
            {"tokens": [[5, 6], [7, 8]], "scores": [0.7, 0.8]},  # size 2
            {
                "tokens": [[9, 10], [11, 12], [13, 14], [15, 16]],
                "scores": [0.9, 1.0, 1.1, 1.2],
            },  # size 4
        ]

        indices = find_groups_summing_to_target(buffer, 4)
        assert indices == [0, 1]  # First two groups sum to 4

    def test_need_dynamic_programming(self):
        """Test when FIFO doesn't work but other combinations do."""
        buffer = [
            {"tokens": [[1, 2], [3, 4], [5, 6]], "scores": [0.5, 0.6, 0.7]},  # size 3
            {"tokens": [[7, 8]], "scores": [0.8]},  # size 1
            {
                "tokens": [[9, 10], [11, 12], [13, 14], [15, 16]],
                "scores": [0.9, 1.0, 1.1, 1.2],
            },  # size 4
        ]

        indices = find_groups_summing_to_target(buffer, 5)
        assert indices == [1, 2]  # Groups at index 1 (size 1) and 2 (size 4)

    def test_impossible_target(self):
        """Test when no combination can reach target."""
        buffer = [
            {"tokens": [[1, 2], [3, 4]], "scores": [0.5, 0.6]},  # size 2
            {
                "tokens": [[5, 6], [7, 8], [9, 10], [11, 12]],
                "scores": [0.7, 0.8, 0.9, 1.0],
            },  # size 4
        ]

        indices = find_groups_summing_to_target(buffer, 3)
        assert indices == []  # Can't make 3 from groups of size 2 and 4

    def test_empty_buffer(self):
        """Test with empty buffer."""
        indices = find_groups_summing_to_target([], 4)
        assert indices == []

    def test_single_group_exact(self):
        """Test when single group matches exactly."""
        buffer = [
            {
                "tokens": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "scores": [0.5, 0.6, 0.7, 0.8],
            },  # size 4
        ]

        indices = find_groups_summing_to_target(buffer, 4)
        assert indices == [0]

    def test_bradley_terry_pairs(self):
        """Test RLAIF use case with Bradley-Terry pairs."""
        buffer = [
            {"tokens": [[1, 2], [3, 4]], "scores": [0.7, 0.3]},  # size 2 (BT pair)
            {"tokens": [[5, 6], [7, 8]], "scores": [0.6, 0.4]},  # size 2 (BT pair)
            {"tokens": [[9, 10], [11, 12]], "scores": [0.8, 0.2]},  # size 2 (BT pair)
            {"tokens": [[13, 14], [15, 16]], "scores": [0.5, 0.5]},  # size 2 (BT pair)
        ]

        indices = find_groups_summing_to_target(buffer, 8)
        assert indices == [0, 1, 2, 3]  # All 4 pairs

    def test_mixed_sizes_complex(self):
        """Test with various power-of-2 sizes."""
        buffer = [
            {"tokens": [[1]], "scores": [0.5]},  # size 1
            {"tokens": [[2], [3]], "scores": [0.6, 0.7]},  # size 2
            {"tokens": [[4]], "scores": [0.8]},  # size 1
            {"tokens": [[5], [6], [7], [8]], "scores": [0.9, 1.0, 1.1, 1.2]},  # size 4
            {"tokens": [[9], [10]], "scores": [1.3, 1.4]},  # size 2
        ]

        # Target 8: should find combination that sums to 8
        indices = find_groups_summing_to_target(buffer, 8)
        assert len(indices) > 0
        assert sum(len(buffer[i]["tokens"]) for i in indices) == 8

    def test_large_groups(self):
        """Test with larger group sizes."""
        buffer = [
            {"tokens": [[i] for i in range(16)], "scores": [0.5] * 16},  # size 16
            {"tokens": [[i] for i in range(8)], "scores": [0.6] * 8},  # size 8
            {"tokens": [[i] for i in range(8)], "scores": [0.7] * 8},  # size 8
        ]

        indices = find_groups_summing_to_target(buffer, 32)
        assert indices == [0, 1, 2]  # All groups needed

    def test_prefer_earlier_indices(self):
        """Test that algorithm prefers earlier indices when multiple solutions exist."""
        buffer = [
            {"tokens": [[1], [2]], "scores": [0.5, 0.6]},  # size 2
            {"tokens": [[3], [4]], "scores": [0.7, 0.8]},  # size 2
            {"tokens": [[5], [6], [7], [8]], "scores": [0.9, 1.0, 1.1, 1.2]},  # size 4
            {"tokens": [[9], [10]], "scores": [1.3, 1.4]},  # size 2
            {"tokens": [[11], [12]], "scores": [1.5, 1.6]},  # size 2
        ]

        indices = find_groups_summing_to_target(buffer, 4)
        assert indices == [0, 1]  # Should prefer first two groups over later ones

    def test_exact_fit_with_remainder(self):
        """Test when we can form exact target but have leftover groups."""
        buffer = [
            {"tokens": [[1], [2]], "scores": [0.5, 0.6]},  # size 2
            {"tokens": [[3], [4], [5], [6]], "scores": [0.7, 0.8, 0.9, 1.0]},  # size 4
            {"tokens": [[7], [8]], "scores": [1.1, 1.2]},  # size 2
            {"tokens": [[9]], "scores": [1.3]},  # size 1
        ]

        indices = find_groups_summing_to_target(buffer, 6)
        assert sorted(indices) == [0, 1]  # First two groups sum to 6

    def test_stress_many_small_groups(self):
        """Test with many small groups."""
        # Create 16 groups of size 1
        buffer = [{"tokens": [[i]], "scores": [i * 0.1]} for i in range(16)]

        indices = find_groups_summing_to_target(buffer, 8)
        assert len(indices) == 8
        assert indices == list(range(8))  # Should take first 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
