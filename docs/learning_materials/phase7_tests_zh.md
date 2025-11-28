# 阶段 7：用测试固化理解

聚焦：通过测试确认期望行为。

- **运行：**  
  ```bash
  pytest tests
  ```
- **阅读：**  
  - `tests/test_mask.py` / `test_ragged_batching.py`：mask 因果与 token 表。  
  - `tests/test_scheduler.py`：调度完成与 decode 预算上限。  
  - `tests/test_kv_cache.py`：稠密/分页追加语义。  
  - `tests/test_paged_attention.py`：分页 KV 上的解码。
- **练习：** 故意改错（如修改 mask 条件）观察失败，理解不变量。*** End Patch
