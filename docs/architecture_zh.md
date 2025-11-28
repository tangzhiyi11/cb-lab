# 架构概览

cb-lab 围绕最小化的连续批处理管线组织，每个目录对应推理循环的一个阶段。

## 模块映射
- `core/request.py`：请求生命周期，管理 prompt 位置、decode 状态与 KV 缓存。
- `core/kv_cache.py`：稠密与分页 KV 存储，均为追加语义。
- `core/batch_builder.py`：ragged token 表与布尔因果 mask 构建。
- `core/scheduler.py`：按 token 预算混合 decode 与 prefill 的主循环。
- `attention/*.py`：稠密、ragged、paged 解码注意力。
- `model/tiny_llm.py`：单层单头注意力（Q/K/V/O）。
- `demos/*.py`：练习脚本。
- `tests/*.py`：mask、KV、调度、paged 解码的基础校验。

## 单步数据流
1. **Decode 选择**：对每个 decode 中的请求取 1 个 token（受预算限制）。
2. **Prefill 选择**：用剩余预算选取 prompt 分片（上限 `prefill_chunk_size`）。
3. **Ragged 构建**：拼接 prefill 分片 → token 表 → ragged 因果 mask。
4. **Prefill 前向**：`forward_prefill_ragged` 产出输出 + 新 K/V，并追加到各自请求。
5. **Decode 前向**：对 decode token 调用 `forward_decode`，读取各自 KV，产出新 token 并追加 K/V。
6. **完成判定**：达到 `max_new_tokens` 的请求被移出 active 列表。

## KV 布局
- **DenseKVCache**：时间维度直接拼接，最易观察。
- **PagedKVCache**：固定大小块，模仿 paged attention 数据布局，可 flatten 查看。

## Mask 策略
- Prefill 使用 ragged 布尔 mask 隔离序列且保持因果。
- Decode 路径天然因果，只访问自身 KV 历史。
