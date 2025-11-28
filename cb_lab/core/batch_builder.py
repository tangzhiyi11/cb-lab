"""Ragged token table + causal mask builder."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch


@dataclass
class TokenMeta:
    req_id: str
    pos_in_seq: int
    global_idx: int


def build_ragged_batch(
    chunks: Sequence[Tuple[str, int, torch.Tensor]],
) -> Tuple[torch.Tensor, List[TokenMeta], torch.Tensor]:
    """Concatenate chunks and build ragged causal mask.

    Args:
        chunks: List of tuples (req_id, start_pos, tokens_chunk).

    Returns:
        tokens_cat: [N_total, d]
        token_table: metadata per token
        ragged_mask: [N_total, N_total] bool mask
    """
    if not chunks:
        return (
            torch.empty(0),
            [],
            torch.empty(0),
        )

    tokens_cat = torch.cat([c[2] for c in chunks], dim=0)
    token_table: List[TokenMeta] = []
    for req_id, start_pos, chunk in chunks:
        for i in range(chunk.size(0)):
            token_table.append(
                TokenMeta(
                    req_id=req_id,
                    pos_in_seq=start_pos + i,
                    global_idx=len(token_table),
                )
            )

    total = len(token_table)
    ragged_mask = torch.zeros(total, total, dtype=torch.bool, device=tokens_cat.device)
    for i, meta_i in enumerate(token_table):
        for j, meta_j in enumerate(token_table):
            if meta_i.req_id != meta_j.req_id:
                continue
            if meta_j.pos_in_seq <= meta_i.pos_in_seq:
                ragged_mask[i, j] = True
    return tokens_cat, token_table, ragged_mask
