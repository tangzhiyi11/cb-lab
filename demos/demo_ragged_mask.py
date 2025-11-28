import torch

from cb_lab.core.batch_builder import build_ragged_batch


def main() -> None:
    dim = 4
    chunk_a = ("A", 0, torch.randn(3, dim))
    chunk_b = ("B", 0, torch.randn(2, dim))
    chunk_a_2 = ("A", 3, torch.randn(2, dim))

    tokens_cat, table, mask = build_ragged_batch([chunk_a, chunk_b, chunk_a_2])
    print("tokens_cat:", tokens_cat.shape)
    print("token_table:")
    for meta in table:
        print(f"  idx={meta.global_idx} req={meta.req_id} pos={meta.pos_in_seq}")
    print("ragged mask (True=allowed):")
    print(mask.int())


if __name__ == "__main__":
    main()
