"""
BLEUBERI: BLEU-based environment for instruction following.

This environment uses BLEU scores as a reward function for training
models to follow instructions. Based on the paper:
"BLEUBERI: BLEU is a surprisingly effective reward for instruction following"
https://arxiv.org/abs/2505.11080
"""

__all__ = ["BLEUBERIEnv"]

from .bleuberi_env import BLEUBERIEnv  # noqa
