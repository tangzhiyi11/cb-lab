# 连续批处理面试 Q&A

从基础到深入，结合 cb-lab 理解与演示。

## 基础
- **问：什么是连续批处理？**  
  **答：** 在 token 边界动态组 batch，新请求可加入，完成的请求可退出，不必等待固定批次。
- **问：为什么要区分 prefill 和 decode？**  
  **答：** Prefill 长且并行，decode 短且顺序；拆开后可混合执行，提高利用率并兼顾延迟。
- **问：什么是 KV cache？**  
  **答：** 存储历史 K/V，decode 直接对缓存做注意力，避免重算全序列。

## 机制
- **问：每步 token 预算如何使用？**  
  **答：** 每步有预算，先放 decode（每个 decode 请求 1 个，受预算限制），剩余再放 prefill 分片（见 `core/scheduler.py`）。
- **问：跨请求如何避免 padding 开销？**  
  **答：** 用 ragged batching：直接拼接 tokens，布尔 ragged 因果 mask 隔离序列（`core/batch_builder.py`）。
- **问：ragged batching 如何保证因果？**  
  **答：** mask 仅在 `seq(i)==seq(j)` 且 `pos(j)<=pos(i)` 时为 True，其他为 False。
- **问：模型如何从 prefill 切换到 decode？**  
  **答：** 用 `prefill_pos` 追踪；prompt 用完进入 decode，以最后一个 prompt token（或 seed）开始，直到 `max_new_tokens`。

## 性能与权衡
- **问：吞吐与延迟怎么权衡？**  
  **答：** `max_tokens_per_step` 大偏吞吐、小偏延迟；decode 优先，预算剩余给 prefill。
- **问：prefill 分块大小的影响？**  
  **答：** 小块降低首 token 延迟但开销高；大块效率高但可能占满预算推迟 decode。
- **问：若 decode 需求超预算会怎样？**  
  **答：** 仅处理不超过预算的 decode，其余等待下一步（`core/scheduler.py` 的 decode 上限）。

## KV 布局
- **问：稠密 vs. 分页 KV？**  
  **答：** 稠密直接拼接，简单；分页按块存储，模拟 paged attention，长上下文解码时减少数据移动（`core/kv_cache.py`）。
- **问：分页 KV 如何用于注意力？**  
  **答：** 将块拼接并裁剪未用尾部，解码对 flatten 视图做注意力（`PagedAllocator.flatten`）。

## 正确性与失效模式
- **问：混合批次如何保持请求隔离？**  
  **答：** 正确构造 ragged mask + 每请求独立 KV；跨请求注意力被屏蔽。
- **问：如何验证调度正确性？**  
  **答：** 单测确保请求完成且预算遵守（`tests/test_scheduler.py`）。
- **问：注意力的数值稳定性？**  
  **答：** mask 后对非可见位置加大负值再 softmax，并用小 EPS 限制概率（`attention/*.py`）。

## 设计与扩展
- **问：如何加入优先级/SLA 调度？**  
  **答：** 选择 decode/prefill 时按优先级排序，或预留预算给高优先级请求。
- **问：如何支持每步多 token decode？**  
  **答：** 允许每请求每步生成多个 token，调整预算与 KV 追加逻辑。
- **问：如何做多机 KV 分片？**
  **答：** 引入分布式 KV 存储并路由查询，需跨设备聚合或分层注意力；cb-lab 未涵盖，但概念上与分布式 KV 访问类似。

## 插件系统
- **问：连续批处理中插件系统的目的是什么？
  **答：** 允许在不修改核心组件的情况下扩展功能，适用于日志记录、指标收集、自定义注意力、缓存优化和业务逻辑集成。
- **问：调度器插件如何工作？
  **答：** 钩子到调度器生命周期事件：`before_step()`、`after_step()` 和 `on_request_completion()`，可以访问调度器状态并修改行为。
- **问：SchedulerPlugin 和 AttentionPlugin 有什么区别？
  **答：** SchedulerPlugin 钩子到调度逻辑；AttentionPlugin 替换或增强注意力计算本身。
- **问：如何实现自定义缓存优化插件？
  **答：** 实现 CachePlugin 接口，包含 `before_append()` 和 `after_append()` 钩子；可以实现压缩、去重或自定义淘汰策略。

## 监控与性能分析
- **问：为什么监控在连续批处理系统中很重要？
  **答：** 为了跟踪利用率、识别瓶颈、检测内存泄漏、确保 SLA，并优化批处理大小等性能参数。
- **问：MetricsCollector 如何工作？
  **答：** 跟踪步骤级统计信息（decode/prefill token 数、时长、内存使用），可以计算请求级指标如吞吐量和延迟。
- **问：MemoryProfiler 和 DetailedMemoryProfiler 有什么区别？
  **答：** MemoryProfiler 提供基本内存跟踪；DetailedMemoryProfiler 提供基于快照的性能分析，支持上下文感知跟踪和泄漏检测。
- **问：如何在连续批处理系统中检测内存泄漏？
  **答：** 使用 MemoryLeakDetector 跟踪内存增长模式，或使用 DetailedMemoryProfiler 分析不同上下文中的内存增量，识别异常增长。

## 性能与可扩展性
- **问：如何衡量连续批处理的效率？
  **答：** 跟踪指标如 token 吞吐量、请求延迟、GPU 利用率、内存效率，并与基准静态批处理进行比较。
- **问：哪些因素影响连续批处理系统的可扩展性？
  **答：** Token 预算大小、请求组合（prompt/生成比例）、KV 缓存效率、内存带宽和 GPU 计算能力。
- **问：如何针对不同工作负载模式进行优化？
  **答：** 根据请求特征调整 `max_tokens_per_step` 和 `prefill_chunk_size`；为不同序列长度使用不同的缓存策略。

## 高级实现
- **问：如何使用插件实现请求优先级？
  **答：** 创建调度器插件，在预算分配前对请求重新排序，可能为高优先级请求预留 token 预算。
- **问：**压缩插件在内存受限环境中的作用是什么？
  **答：** 通过量化、剪枝或自适应块分配等技术减少 KV 缓存内存占用。
- **问：如何实现自适应批处理大小？
  **答：** 监控系统利用率和请求模式，根据当前负载和性能目标动态调整 `max_tokens_per_step`。
