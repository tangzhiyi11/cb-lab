"""Standard dense causal attention."""

import torch

from cb_lab.core.constants import EPS
from cb_lab.attention.mask_utils import causal_mask


def dense_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    length = q.size(0)
    mask = causal_mask(length, q.device)
    scores = torch.matmul(q, k.transpose(0, 1)) / (q.size(-1) ** 0.5)
    scores = scores.masked_fill(~mask, -1e9)
    probs = torch.softmax(scores, dim=-1)
    probs = probs.clamp(EPS, 1.0)
    return torch.matmul(probs, v)
