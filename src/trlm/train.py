"""Shared byte-level training utilities."""

from __future__ import annotations

import random

import torch


def sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    generator: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.numel() - sequence_length - 1
    if max_start < 0:
        raise ValueError("corpus must contain more bytes than sequence_length")
    starts = [generator.randint(0, max_start) for _ in range(batch_size)]
    inputs = torch.stack([tokens[start : start + sequence_length] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + sequence_length + 1] for start in starts])
    return inputs.to(device), targets.to(device)
