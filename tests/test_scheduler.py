import torch

from cb_lab.core.kv_cache import DenseKVCache
from cb_lab.core.request import Request
from cb_lab.core.scheduler import Scheduler
from cb_lab.model.tiny_llm import TinyLLM


def test_scheduler_finishes_requests():
    dim = 5
    model = TinyLLM(dim)
    scheduler = Scheduler(model, max_tokens_per_step=4, prefill_chunk_size=2)

    req1 = Request(
        req_id="a",
        prompt=torch.randn(3, dim),
        max_new_tokens=2,
        kv_cache=DenseKVCache(torch.device("cpu")),
    )
    req2 = Request(
        req_id="b",
        prompt=torch.randn(2, dim),
        max_new_tokens=1,
        kv_cache=DenseKVCache(torch.device("cpu")),
    )
    scheduler.add_request(req1)
    scheduler.add_request(req2)

    steps = 0
    while scheduler.active and steps < 10:
        scheduler.step()
        steps += 1

    assert not scheduler.active
    assert req1.finished
    assert req2.finished


def test_scheduler_respects_decode_budget_cap():
    dim = 4
    model = TinyLLM(dim)
    scheduler = Scheduler(model, max_tokens_per_step=1, prefill_chunk_size=2)

    req_a = Request(
        req_id="a",
        prompt=torch.randn(1, dim),
        max_new_tokens=2,
        kv_cache=DenseKVCache(torch.device("cpu")),
    )
    req_b = Request(
        req_id="b",
        prompt=torch.randn(1, dim),
        max_new_tokens=2,
        kv_cache=DenseKVCache(torch.device("cpu")),
    )
    # Force both into decode state.
    req_a.prefill_pos = req_a.prompt.size(0)
    req_b.prefill_pos = req_b.prompt.size(0)

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)

    scheduler.step()
    decoded_counts = [len(req_a.generated_tokens), len(req_b.generated_tokens)]
    assert sum(decoded_counts) == 1  # only one decode allowed per step at budget=1
