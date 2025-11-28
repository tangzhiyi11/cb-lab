# 阶段 5：KV 缓存布局

聚焦：稠密与分页 KV 存储及追加语义。

- **阅读：** `core/kv_cache.py`
- **运行：**  
  ```bash
  python -m demos.demo_paged_attention
  ```
- **观察：** prefill/decode 后的 `len(cache)`；`PagedKVCache` 中 `block_size` 的影响。
- **练习：** 在 demo 中切换 `DenseKVCache` / `PagedKVCache`；查看 `PagedAllocator.flatten` 的输出形状。*** End Patch
