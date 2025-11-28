# 原理 → 实验手册

针对关键原理，提供在 cb-lab 中观测或验证的方法。

## Prefill / Decode 混合
- **观察点**：decode 优先，prefill 用剩余预算。
- **做法**：运行 `python -m demos.demo_prefill_decode_mix`，看日志中的 `decode=...`、`prefill=...`。调整 `core/scheduler.py` 的 `max_tokens_per_step` / `prefill_chunk_size`。

## 无 padding 的 ragged batching
- **观察点**：token 表与 ragged 因果 mask，序列隔离。
- **做法**：运行 `python -m demos.demo_ragged_mask`，可添加更多分片观察 mask 变化；实现详见 `core/batch_builder.py`。

## KV 缓存增长（稠密）
- **观察点**：prefill、decode 后 KV 追加长度。
- **做法**：`demo_prefill_only.py`（prefill），`demo_prefill_decode_mix.py`（混合）；打印 `len(cache)`。实现：`core/kv_cache.py`。

## Paged KV 布局
- **观察点**：块分配与 flatten 长度。
- **做法**：`python -m demos.demo_paged_attention`，调整 `PagedKVCache` 的 `block_size`，比较 prefill/decode 后长度。

## 调度完成性
- **观察点**：达到 `max_new_tokens` 后请求退出 active 集。
- **做法**：运行 `demo_scheduler_timeline.py` 查看总结；`tests/test_scheduler.py` 断言调度器清空请求。

## 注意力正确性（基础）
- **观察点**：输出形状正确，mask 保持因果。
- **做法**：`pytest tests/test_mask.py`、`test_ragged_batching.py` 检查 mask；`test_paged_attention.py` 检查 paged 解码；`test_kv_cache.py` 检查追加语义。
