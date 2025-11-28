import torch

from cb_lab.core.kv_cache import PagedKVCache
from cb_lab.model.tiny_llm import TinyLLM


def test_decode_over_paged_cache():
    dim = 6
    model = TinyLLM(dim)
    cache = PagedKVCache(block_size=2, device=torch.device("cpu"))
    prompt = torch.randn(3, dim)
    _, k, v = model.forward_prefill_dense(prompt)
    cache.append(k, v)

    new_token = torch.randn(dim)
    out, k_new, v_new = model.forward_decode(new_token, cache)
    assert out.shape == (1, dim)
    cache.append(k_new, v_new)
    assert len(cache) == 4
