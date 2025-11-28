# 阶段 4：调度时间线

聚焦：请求在多步中的活跃与退出。

- **运行：**  
  ```bash
  python -m demos.demo_scheduler_timeline
  ```
- **观察：** 每步 active 集、decode/prefill 数量、请求何时移除。
- **练习：** 提高 `max_new_tokens` 或增加请求；调整 `max_tokens_per_step` 体验延迟/吞吐变化。
