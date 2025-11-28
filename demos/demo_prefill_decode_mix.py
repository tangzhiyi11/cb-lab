import torch

from cb_lab.core.kv_cache import DenseKVCache
from cb_lab.core.request import Request
from cb_lab.core.scheduler import Scheduler
from cb_lab.model.tiny_llm import TinyLLM


def main() -> None:
    dim = 8
    model = TinyLLM(dim)
    scheduler = Scheduler(model, max_tokens_per_step=6, prefill_chunk_size=3)

    req1 = Request(
        req_id="req-1",
        prompt=torch.randn(5, dim),
        max_new_tokens=3,
        kv_cache=DenseKVCache(torch.device("cpu")),
    )
    req2 = Request(
        req_id="req-2",
        prompt=torch.randn(4, dim),
        max_new_tokens=2,
        kv_cache=DenseKVCache(torch.device("cpu")),
    )

    scheduler.add_request(req1)
    scheduler.add_request(req2)

    while scheduler.active:
        scheduler.step()

    print("\nGeneration finished.")
    for req in [req1, req2]:
        print(
            f"{req.req_id}: prefill_len={req.prefill_pos}, generated={len(req.generated_tokens)}"
        )


if __name__ == "__main__":
    main()
