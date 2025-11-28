"""Ragged attention that relies on an explicit mask."""

import torch

from cb_lab.core.constants import EPS


def ragged_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(0, 1)) / (q.size(-1) ** 0.5)
    scores = scores.masked_fill(~mask, -1e9)
    probs = torch.softmax(scores, dim=-1)
    probs = probs.clamp(EPS, 1.0)
    return torch.matmul(probs, v)
