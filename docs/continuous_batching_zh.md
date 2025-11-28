# cb-lab 中的连续批处理

本文扩展连续批处理（可参考 Hugging Face continuous batching 文章）的动机与机制，并标注在 cb-lab 中的对应位置。

## 为什么需要连续批处理
- **延迟 vs 吞吐**：单请求推理 GPU 利用率低，大静态 batch 虽提升吞吐但增加排队延迟。连续批处理让请求在 token 边界自由加入/退出，兼顾利用率与响应。
- **Prefill / Decode 非对称**：prefill 长且并行，decode 短且顺序（每步 1 token）。混合两者能填满算力。
- **KV 复用**：保存历史 K/V，decode 仅关注缓存，避免重复计算。

## 核心机制（cb-lab 实现）
- **每步 token 预算 (`max_tokens_per_step`)**：先放 decode token（每个 decode 请求 1 个，受预算上限），剩余容量再放 prefill 分片。
- **Prefill 分块 (`prefill_chunk_size`)**：长 prompt 切小块，减少首 token 延迟，也便于塞进预算。
- **Ragged batching**：prefill 分片直接拼接，无 padding；布尔 ragged 因果 mask 保证序列隔离且保持因果。
- **KV 增长**：prefill 和 decode 后都追加新 K/V，供后续 decode 使用（dense 或 paged）。
- **分页布局（简化）**：KV 按固定块存储，模拟 paged attention，方便观察解码数据布局。

## 可调旋钮
- `max_tokens_per_step`：大值偏吞吐，小值偏延迟；观察调度日志体会 decode/prefill 比例。
- `prefill_chunk_size`：小块加快首 token，大块提升 prefill 效率；与预算共同决定混合比例。
- 缓存后端：切换 `DenseKVCache` / `PagedKVCache`，比较 flatten 长度与布局。

## 代码定位
- Prefill 分块与 decode seed：`core/request.py`
- 调度预算与混合策略：`core/scheduler.py`
- Ragged mask 构建：`core/batch_builder.py`
- KV 缓存实现：`core/kv_cache.py`
- 注意力数学：`attention/*.py`
- 模型接口：`model/tiny_llm.py`

## 上手实验
- **预算压力测试**：调 `max_tokens_per_step`、`prefill_chunk_size`，运行 `python -m demos.demo_prefill_decode_mix`，观察每步 decode/prefill 计数。
- **Ragged 规模增长**：在 `demo_ragged_mask.py` 扩展 prompt 长度，看 mask 如何在无 padding 下增长并保持隔离。
- **Paged vs Dense**：在 `demo_paged_attention.py` 切换缓存后端，比较 prefill/decode 后的 `len(cache)`。
- **更长活跃集**：在 `demo_scheduler_timeline.py` 提高 `max_new_tokens` 或增加请求，看 active 集停留时间与预算消耗。
