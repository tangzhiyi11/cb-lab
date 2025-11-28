"""A minimal single-layer attention model."""

from typing import Tuple

import torch
from torch import nn

from cb_lab.attention.dense_attention import dense_causal_attention
from cb_lab.attention.paged_attention import paged_decode_attention
from cb_lab.attention.ragged_attention import ragged_attention
from cb_lab.core.kv_cache import BaseKVCache


class TinyLLM(nn.Module):
    """Tiny single-head attention block used for all demos."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)

    def forward_prefill_ragged(
        self, tokens: torch.Tensor, ragged_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill path: ragged attention over concatenated prompt chunks."""
        q = self.wq(tokens)
        k = self.wk(tokens)
        v = self.wv(tokens)
        attn_out = ragged_attention(q, k, v, ragged_mask)
        out = self.wo(attn_out)
        return out, k, v

    def forward_prefill_dense(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alternative dense prefill (single sequence)."""
        q = self.wq(tokens)
        k = self.wk(tokens)
        v = self.wv(tokens)
        attn_out = dense_causal_attention(q, k, v)
        out = self.wo(attn_out)
        return out, k, v

    def forward_decode(
        self, new_token: torch.Tensor, kv_cache: BaseKVCache
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode path: attend a single new token against the KV cache."""
        token = new_token.unsqueeze(0) if new_token.dim() == 1 else new_token
        q = self.wq(token)
        k_new = self.wk(token)
        v_new = self.wv(token)
        attn_out = paged_decode_attention(q, kv_cache)
        out = self.wo(attn_out)
        return out, k_new, v_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for basic usage."""
        # Simple dense attention for single sequence
        out, _, _ = self.forward_prefill_dense(x)
        return out
