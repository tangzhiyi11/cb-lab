# Phase 5: KV Cache Layouts

Focus: dense vs. paged KV storage and append semantics.

- **Read:** `core/kv_cache.py`
- **Run:**  
  ```bash
  python -m demos.demo_paged_attention
  ```
- **Observe:** `len(cache)` after prefill/decode; `block_size` effects in `PagedKVCache`.
- **Exercise:** Switch between `DenseKVCache` and `PagedKVCache` in the demo; inspect `PagedAllocator.flatten` output shapes.
