"""Simplified decode attention over a KV cache (dense or paged)."""

import torch

from cb_lab.core.constants import EPS
from cb_lab.core.kv_cache import BaseKVCache


def paged_decode_attention(q: torch.Tensor, kv_cache: BaseKVCache) -> torch.Tensor:
    k_all, v_all = kv_cache.get_kv()
    if k_all.numel() == 0:
        # No history; just return q transformed by identity attention.
        return q
    scores = torch.matmul(q, k_all.transpose(0, 1)) / (q.size(-1) ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    probs = probs.clamp(EPS, 1.0)
    attended = torch.matmul(probs, v_all)
    return attended
