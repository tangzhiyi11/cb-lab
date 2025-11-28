"""Ragged token table + causal mask builder for variable-length sequences."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from cb_lab.exceptions import ValidationError, ConfigurationError


@dataclass
class TokenMeta:
    """Metadata for a single token in the ragged batch.

    Attributes:
        req_id: Request identifier this token belongs to
        pos_in_seq: Position of token within its original sequence
        global_idx: Global index in the concatenated batch
    """

    req_id: str
    pos_in_seq: int
    global_idx: int


def build_ragged_batch(
    chunks: Sequence[Tuple[str, int, torch.Tensor]],
) -> Tuple[torch.Tensor, List[TokenMeta], torch.Tensor]:
    """Concatenate chunks and build ragged causal mask for variable-length sequences.

    This function builds a batch from different request chunks while maintaining
    causal attention constraints within each request but allowing cross-request
    attention for efficient processing.

    Args:
        chunks: Sequence of tuples (req_id, start_pos, tokens_chunk)
                where start_pos is the position in the original sequence

    Returns:
        tokens_cat: Concatenated tokens tensor of shape [N_total, d]
        token_table: List of TokenMeta objects with metadata per token
        ragged_mask: Boolean causal mask of shape [N_total, N_total]

    Raises:
        ValidationError: If input chunks are invalid
        ConfigurationError: If batch construction fails
    """
    try:
        return _build_ragged_batch_impl(chunks)
    except Exception as e:
        if isinstance(e, (ValidationError, ConfigurationError)):
            raise
        raise ConfigurationError(f"Failed to build ragged batch: {e}") from e


def _build_ragged_batch_impl(
    chunks: Sequence[Tuple[str, int, torch.Tensor]],
) -> Tuple[torch.Tensor, List[TokenMeta], torch.Tensor]:
    """Implementation of ragged batch construction."""
    # Handle empty input
    if not chunks:
        return (
            torch.empty(0, 0, dtype=torch.float32),
            [],
            torch.empty(0, 0, dtype=torch.bool),
        )

    # Validate input chunks
    _validate_chunks(chunks)

    # Concatenate all token chunks
    tokens_list = [chunk for _, _, chunk in chunks]
    tokens_cat = torch.cat(tokens_list, dim=0)

    # Build token metadata table
    token_table = _build_token_table(chunks)

    # Build ragged causal mask
    ragged_mask = _build_ragged_mask(token_table, tokens_cat.device)

    return tokens_cat, token_table, ragged_mask


def _validate_chunks(chunks: Sequence[Tuple[str, int, torch.Tensor]]) -> None:
    """Validate input chunks for ragged batch construction."""
    if not isinstance(chunks, (list, tuple)):
        raise ValidationError(f"chunks must be list or tuple, got {type(chunks)}")

    if len(chunks) == 0:
        return  # Empty chunks are handled separately

    # Check device consistency
    first_device = (
        chunks[0][2].device if chunks[0][2].numel() > 0 else torch.device("cpu")
    )
    expected_dim = chunks[0][2].size(1) if chunks[0][2].numel() > 0 else None

    for i, (req_id, start_pos, chunk) in enumerate(chunks):
        # Validate chunk tuple structure
        if not isinstance(req_id, str):
            raise ValidationError(
                f"req_id must be str, got {type(req_id)} at index {i}"
            )

        if not isinstance(start_pos, int) or start_pos < 0:
            raise ValidationError(
                f"start_pos must be non-negative int, got {start_pos} at index {i}"
            )

        if not isinstance(chunk, torch.Tensor):
            raise ValidationError(
                f"chunk must be torch.Tensor, got {type(chunk)} at index {i}"
            )

        if chunk.dim() != 2:
            raise ValidationError(
                f"chunk must be 2D tensor [seq_len, dim], got {chunk.dim()}D at index {i}"
            )

        if chunk.size(0) == 0:
            continue  # Skip empty chunks

        # Check device consistency
        if chunk.device != first_device:
            raise ValidationError(
                f"All chunks must be on same device. Expected {first_device}, "
                f"got {chunk.device} at index {i}"
            )

        # Check feature dimension consistency
        if expected_dim is not None and chunk.size(1) != expected_dim:
            raise ValidationError(
                f"All chunks must have same feature dimension. Expected {expected_dim}, "
                f"got {chunk.size(1)} at index {i}"
            )


def _build_token_table(
    chunks: Sequence[Tuple[str, int, torch.Tensor]],
) -> List[TokenMeta]:
    """Build metadata table for all tokens in the batch."""
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

    return token_table


def _build_ragged_mask(
    token_table: List[TokenMeta], device: torch.device
) -> torch.Tensor:
    """Build ragged causal mask for the token batch."""
    total_tokens = len(token_table)

    if total_tokens == 0:
        return torch.empty(0, 0, dtype=torch.bool, device=device)

    # Initialize mask with all False (no attention allowed)
    ragged_mask = torch.zeros(
        total_tokens, total_tokens, dtype=torch.bool, device=device
    )

    # Build causal attention mask within each request
    # token_i can attend to token_j if they are in the same request
    # and j comes before or at the same position as i (causal constraint)
    for i, meta_i in enumerate(token_table):
        for j, meta_j in enumerate(token_table):
            # Only allow attention within the same request
            if meta_i.req_id != meta_j.req_id:
                continue

            # Enforce causal constraint: can attend to current and previous positions
            if meta_j.pos_in_seq <= meta_i.pos_in_seq:
                ragged_mask[i, j] = True

    return ragged_mask
