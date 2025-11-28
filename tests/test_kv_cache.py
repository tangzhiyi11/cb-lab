import torch

from cb_lab.core.kv_cache import DenseKVCache, PagedKVCache


def test_dense_kv_append():
    cache = DenseKVCache(torch.device("cpu"))
    k1 = torch.randn(2, 4)
    v1 = torch.randn(2, 4)
    cache.append(k1, v1)
    assert len(cache) == 2
    k2 = torch.randn(1, 4)
    v2 = torch.randn(1, 4)
    cache.append(k2, v2)
    assert len(cache) == 3
    k_all, v_all = cache.get_kv()
    assert torch.allclose(k_all[:2], k1)
    assert torch.allclose(v_all[-1], v2[0])


def test_paged_kv_append_and_flatten():
    cache = PagedKVCache(block_size=2, device=torch.device("cpu"))
    k = torch.randn(3, 4)
    v = torch.randn(3, 4)
    cache.append(k, v)
    assert len(cache) == 3
    k_all, v_all = cache.get_kv()
    assert k_all.shape[0] == 3
    assert v_all.shape[0] == 3
