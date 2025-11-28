import torch

from cb_lab.core.kv_cache import PagedKVCache
from cb_lab.core.request import Request
from cb_lab.model.tiny_llm import TinyLLM


def main() -> None:
    dim = 8
    model = TinyLLM(dim)
    cache = PagedKVCache(block_size=2, device=torch.device("cpu"))
    req = Request(
        req_id="paged-demo",
        prompt=torch.randn(4, dim),
        max_new_tokens=2,
        kv_cache=cache,
    )

    # Prefill dense to populate cache.
    out, k, v = model.forward_prefill_dense(req.prompt)
    cache.append(k, v)
    print("KV cache length after prefill:", len(cache))

    # Decode one token using paged attention.
    out_decode, k_new, v_new = model.forward_decode(req.decode_seed, cache)
    cache.append(k_new, v_new)
    print("Decode output shape:", out_decode.shape)
    print("KV cache length after decode:", len(cache))


if __name__ == "__main__":
    main()
