# 阶段 2：Ragged Batching

聚焦：拼接 + ragged 因果 mask，无需 padding。

- **阅读：**  
  - `core/batch_builder.py`（token 表、mask 规则）  
  - `attention/ragged_attention.py`
- **运行：**  
  ```bash
  python -m demos.demo_ragged_mask
  ```
- **观察：** Token 表字段（`req_id`、`pos_in_seq`、`global_idx`）；mask 仅允许同序列因果可见。
- **练习：** 在 demo 中加入更多分片/请求，验证跨请求 mask 仍为 False。*** End Patch
