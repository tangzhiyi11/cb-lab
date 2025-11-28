import torch

from cb_lab.core.batch_builder import build_ragged_batch


def test_ragged_mask_causal_and_isolated():
    dim = 4
    chunk_a = ("A", 0, torch.randn(2, dim))
    chunk_b = ("B", 0, torch.randn(3, dim))
    tokens, table, mask = build_ragged_batch([chunk_a, chunk_b])

    assert tokens.shape[0] == 5
    # Same sequence causal
    assert mask[1, 0]  # A sees earlier A
    assert not mask[0, 1]  # causal
    # Cross sequence blocked
    assert not mask[0, 3]
    assert not mask[4, 1]
