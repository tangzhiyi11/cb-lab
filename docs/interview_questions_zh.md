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
  **答：** 引入分布式 KV 存储并路由查询，需跨设备聚合或分层注意力；cb-lab 未涵盖，但概念上与分布式 KV 访问类似。*** End Patch
