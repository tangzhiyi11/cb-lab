# 阶段 8：自由探索

聚焦：扩展场景，观察连续批处理的表现。

- **更多请求：** 在 demo 中增加长 prompt 或更多并发请求，观察 ragged mask 与 KV 增长。
- **预算压力：** 减小 `max_tokens_per_step` 模拟高负载，观察 decode 排队与利用率。
- **Prefill 分块调优：** 调整 `prefill_chunk_size`，权衡首 token 延迟与效率。
- **采样变化：** 在 decode 输出上加简单非线性（如 `torch.tanh`）模拟 token 变化。
- **指标：** 打印每步 decode/prefill 数、缓存长度，感受利用率。
