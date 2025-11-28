"""Standard dense causal attention."""

import torch

from cb_lab.core.constants import EPS
from cb_lab.attention.mask_utils import causal_mask


def dense_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # Handle both 2D (seq_len, dim) and 3D (batch, seq_len, dim) inputs
    if q.dim() == 3:
        # Batch processing: keep batch dimension
        batch_size, seq_len, dim = q.shape

        # Create causal mask for each batch
        mask = causal_mask(seq_len, q.device)  # (seq_len, seq_len)

        # Compute attention scores for each item in batch
        output = []
        for i in range(batch_size):
            q_item = q[i]  # (seq_len, dim)
            k_item = k[i]  # (seq_len, dim)
            v_item = v[i]  # (seq_len, dim)

            scores = torch.matmul(q_item, k_item.transpose(0, 1)) / (dim ** 0.5)
            scores = scores.masked_fill(~mask, -1e9)
            probs = torch.softmax(scores, dim=-1)
            probs = probs.clamp(EPS, 1.0)
            output_item = torch.matmul(probs, v_item)
            output.append(output_item)

        return torch.stack(output)  # (batch_size, seq_len, dim)
    else:
        # Single sequence processing
        length = q.size(0)
        mask = causal_mask(length, q.device)
        scores = torch.matmul(q, k.transpose(0, 1)) / (q.size(-1) ** 0.5)
        scores = scores.masked_fill(~mask, -1e9)
        probs = torch.softmax(scores, dim=-1)
        probs = probs.clamp(EPS, 1.0)
        return torch.matmul(probs, v)
