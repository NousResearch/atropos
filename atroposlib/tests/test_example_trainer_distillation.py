"""Tests for example_trainer on-policy distillation support."""

import torch

from example_trainer.data import pad_data_to_good_offset
from example_trainer.training import compute_distillation_loss, compute_grpo_loss


class _DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class _DummyModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)

    def forward(self, tokens):
        batch = tokens.shape[0]
        return _DummyOutput(self.logits[:batch])


def test_pad_data_extracts_distillation_targets():
    data = {
        "batch": [
            {
                "tokens": [[11, 12, 13, 14]],
                "masks": [[-100, -100, 13, 14]],
                "scores": [1.0],
                "overrides": None,
                "generation_params": {"temperature": 0.7},
                "group_overrides": None,
                "inference_logprobs": [[1.0, 1.0, -0.2, -0.3]],
                "distill_token_ids": [[[101, 102], [103], [104], [105, 106]]],
                "distill_logprobs": [
                    [[-0.1, -1.2], [-0.3], [-0.4], [-0.2, -0.5]]
                ],
            }
        ]
    }

    (
        token_batches,
        label_batches,
        advantage_batches,
        temperature_batches,
        inference_logprob_batches,
        distill_token_id_batches,
        distill_logprob_batches,
    ) = pad_data_to_good_offset(
        data,
        batch_size=1,
        extract_inference_logprobs=True,
        extract_distillation_targets=True,
    )

    assert len(token_batches) == 1
    assert len(label_batches) == 1
    assert len(advantage_batches) == 1
    assert len(temperature_batches) == 1
    assert inference_logprob_batches is not None
    assert distill_token_id_batches is not None
    assert distill_logprob_batches is not None

    # Causal shift drops the first position, so position 0 in the trainer aligns
    # with the second original token position.
    assert distill_token_id_batches[0][0][0] == [101, 102]
    assert distill_token_id_batches[0][0][1] == [103]
    assert distill_logprob_batches[0][0][1] == [-0.3]


def test_compute_grpo_loss_supports_distillation_only():
    logits = torch.tensor(
        [
            [
                [0.0, 3.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    model = _DummyModel(logits)

    tokens = torch.tensor([[7, 8]], dtype=torch.long)
    labels = torch.tensor([[1, -100]], dtype=torch.long)
    advantages = torch.tensor([[1.0]], dtype=torch.float32)
    temperatures = torch.tensor([[[1.0]]], dtype=torch.float32)

    loss, metrics = compute_grpo_loss(
        model=model,
        tokens=tokens,
        labels=labels,
        advantages=advantages,
        temperatures=temperatures,
        gradient_accumulation_steps=1,
        inference_logprobs=None,
        distill_token_ids=[[[1, 2], []]],
        distill_logprobs=[[[-0.1, -1.5], []]],
        distill_enabled=True,
        distill_coef=0.5,
        distill_temperature=1.0,
        distill_loss_type="kl",
        distill_only=True,
    )

    assert loss.item() > 0
    assert metrics["distill_positions"] == 1
    assert metrics["distill_loss"] > 0
    assert metrics["mean_ratio"] == 1.0


def test_compute_distillation_loss_supports_jsd():
    logits = torch.tensor(
        [
            [
                [0.0, 2.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[1, -100]], dtype=torch.long)

    loss, metrics = compute_distillation_loss(
        logits=logits,
        labels=labels,
        distill_token_ids=[[[1, 2], []]],
        distill_logprobs=[[[-0.2, -1.7], []]],
        loss_type="jsd",
        temperature=1.0,
        jsd_beta=0.1,
    )

    assert loss.item() >= 0
    assert metrics["distill_positions"] == 1
