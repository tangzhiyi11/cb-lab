# 架构概览

cb-lab 围绕最小化的连续批处理管线组织，每个目录对应推理循环的一个阶段。

## 模块映射
- `core/request.py`：请求生命周期，管理 prompt 位置、decode 状态与 KV 缓存。
- `core/kv_cache.py`：稠密与分页 KV 存储，均为追加语义。
- `core/batch_builder.py`：ragged token 表与布尔因果 mask 构建。
- `core/scheduler.py`：按 token 预算混合 decode 与 prefill 的主循环。
- `attention/*.py`：稠密、ragged、paged 解码注意力。
- `model/tiny_llm.py`：单层单头注意力（Q/K/V/O）。
- `monitoring/metrics.py`：性能监控与指标收集工具。
- `monitoring/memory_profiler.py`：高级内存分析与泄漏检测。
- `plugins/base.py`：可扩展性的插件系统基础架构。
- `plugins/builtin.py`：内置插件：日志、指标、压缩、可视化。
- `demos/*.py`：练习脚本和交互式演示。
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

## 插件系统
- **PluginManager**：管理调度器、注意力、缓存插件的中央注册器。
- **SchedulerPlugin**：钩子函数，用于 before_step、after_step 和 on_request_completion 事件。
- **AttentionPlugin**：自定义注意力计算实现。
- **CachePlugin**：KV 缓存压缩与优化策略。
- 内置插件：LoggingPlugin、MetricsPlugin、CacheCompressionPlugin、AttentionVisualizationPlugin。

## 监控与性能分析
- **MetricsCollector**：收集步骤级统计信息（处理 token 数、步骤时长、内存使用）。
- **MemoryProfiler**：跟踪系统和 GPU 内存使用，支持基线比较。
- **DetailedMemoryProfiler**：高级性能分析，支持上下文感知快照和泄漏检测。
- **PerformanceBenchmark**：测量和比较组件性能的工具。
