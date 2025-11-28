# 连续批处理学习路线

目标：按阶段利用 cb-lab 深入理解 continuous batching，从 prefill 到 paged attention。每步都提供阅读与运行指引。

## 阶段 0：准备
- 安装并跑测试：`python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pytest tests`
- 快速浏览术语：`docs/get_started.md`、`docs/continuous_batching.md`

## 阶段 1：Prefill 基础
- **概念**：prefill 仅构建 KV，不生成新 token。
- **阅读**：`core/request.py`（prefill_pos, get_prefill_chunk），`model/tiny_llm.py`（`forward_prefill_dense`）。
- **运行**：`python -m demos.demo_prefill_only`
- **观察**：KV 长度等于 prompt 长度；注意力为稠密因果。

## 阶段 2：Ragged batching
- **概念**：无 padding，拼接分片，用 ragged 因果 mask 隔离序列。
- **阅读**：`core/batch_builder.py`，`attention/ragged_attention.py`
- **运行**：`python -m demos.demo_ragged_mask`
- **观察**：token 表（req_id/pos），mask 仅允许同序列的因果可见。

## 阶段 3：Prefill + Decode 混合
- **概念**：每步在 token 预算内优先放 decode，再放 prefill 分片。
- **阅读**：`core/scheduler.py`（decode-first、预算），`core/request.py`（decode_seed、append_token）。
- **运行**：`python -m demos.demo_prefill_decode_mix`
- **观察**：日志 `decode=...`、`prefill=...` 及请求 ID；`max_new_tokens` 达成后请求完成。

## 阶段 4：调度时间线
- **概念**：连续批处理保持请求流动直到完成。
- **运行**：`python -m demos.demo_scheduler_timeline`
- **观察**：请求活跃时长、每步预算消耗。
- **尝试**：调 `max_tokens_per_step` / `prefill_chunk_size` 体验吞吐/延迟取舍。

## 阶段 5：KV 布局
- **概念**：稠密与分页存储，均为追加。
- **阅读**：`core/kv_cache.py`
- **运行**：`python -m demos.demo_paged_attention`
- **观察**：prefill/decode 后的 `len(cache)`；调整 `block_size` 看块分配。

## 阶段 6：注意力路径
- **概念**：prefill 走 ragged/稠密因果；decode 仅访问缓存。
- **阅读**：`attention/dense_attention.py`、`attention/ragged_attention.py`、`attention/paged_attention.py`
- **练习**：对单序列比较 ragged 与 dense 输出（应一致）。

## 阶段 7：测试即示例
- **运行**：`pytest tests`
- **阅读**：`tests/test_*` 理解期望行为（mask 因果、调度完成、KV 追加）。

## 阶段 8：扩展探索
- 增加更多请求与更长 prompt，观察 mask/KV 增长。
- 降低 `max_tokens_per_step` 模拟高负载，评估吞吐。
- 在 decode 输出上加简单“采样”（如 tanh）观察 token 轨迹变化。

## 分阶段学习资料
- 0：`docs/learning_materials/phase0_setup_zh.md`
- 1：`docs/learning_materials/phase1_prefill_zh.md`
- 2：`docs/learning_materials/phase2_ragged_zh.md`
- 3：`docs/learning_materials/phase3_mix_zh.md`
- 4：`docs/learning_materials/phase4_timeline_zh.md`
- 5：`docs/learning_materials/phase5_kv_zh.md`
- 6：`docs/learning_materials/phase6_attention_zh.md`
- 7：`docs/learning_materials/phase7_tests_zh.md`
- 8：`docs/learning_materials/phase8_exploration_zh.md`
