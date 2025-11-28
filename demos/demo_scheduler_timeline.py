import torch

from cb_lab.core.kv_cache import DenseKVCache
from cb_lab.core.request import Request
from cb_lab.core.scheduler import Scheduler
from cb_lab.model.tiny_llm import TinyLLM


def main() -> None:
    dim = 6
    model = TinyLLM(dim)
    scheduler = Scheduler(model, max_tokens_per_step=5, prefill_chunk_size=2)

    reqs = [
        Request(
            req_id="timeline-1",
            prompt=torch.randn(3, dim),
            max_new_tokens=2,
            kv_cache=DenseKVCache(torch.device("cpu")),
        ),
        Request(
            req_id="timeline-2",
            prompt=torch.randn(5, dim),
            max_new_tokens=3,
            kv_cache=DenseKVCache(torch.device("cpu")),
        ),
    ]
    for r in reqs:
        scheduler.add_request(r)

    step = 0
    while scheduler.active and step < 10:
        scheduler.step()
        step += 1

    print("\nScheduler timeline done.")
    for r in reqs:
        print(f"{r.req_id} generated={len(r.generated_tokens)} finished={r.finished}")


if __name__ == "__main__":
    main()
