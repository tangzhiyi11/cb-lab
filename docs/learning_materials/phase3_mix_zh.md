# 阶段 3：Prefill + Decode 混合

聚焦：token 预算下优先 decode，剩余容量填 prefill 分片。

- **阅读：**  
  - `core/scheduler.py`（decode-first、预算、分片选择）  
  - `core/request.py`（decode_seed、append_token）
- **运行：**  
  ```bash
  python -m demos.demo_prefill_decode_mix
  ```
- **观察：** 日志中的 `decode=...`、`prefill=...` 及请求 ID；到达 `max_new_tokens` 后请求完成。
- **练习：** 调整 `max_tokens_per_step` 或 `prefill_chunk_size` 观察 decode/prefill 比例；减小预算观察 decode 排队。*** End Patch
