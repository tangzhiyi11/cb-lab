"""Helpers to build causal masks."""

import torch


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
