import torch

from cb_lab.core.batch_builder import build_ragged_batch


def test_token_table_records_positions():
    dim = 3
    chunk_a = ("A", 0, torch.ones(2, dim))
    chunk_a_2 = ("A", 2, torch.ones(1, dim))
    chunk_b = ("B", 0, torch.ones(1, dim))
    _, table, _ = build_ragged_batch([chunk_a, chunk_b, chunk_a_2])

    assert table[0].req_id == "A"
    assert table[0].pos_in_seq == 0
    assert table[2].req_id == "B"
    assert table[2].pos_in_seq == 0
    assert table[3].pos_in_seq == 2


def test_empty_ragged_batch_returns_empty_structures():
    tokens, table, mask = build_ragged_batch([])
    assert tokens.numel() == 0
    assert table == []
    assert mask.numel() == 0
