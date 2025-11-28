import torch

from cb_lab.core.kv_cache import DenseKVCache
from cb_lab.model.tiny_llm import TinyLLM


def main() -> None:
    dim = 8
    tokens = torch.randn(6, dim)
    cache = DenseKVCache(tokens.device)
    model = TinyLLM(dim)

    out, k, v = model.forward_prefill_dense(tokens)
    cache.append(k, v)

    print("prefill tokens:", tokens.shape)
    print("output:", out.shape)
    print("kv cache len:", len(cache))


if __name__ == "__main__":
    main()
