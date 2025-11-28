# 阶段 1：Prefill 基础

聚焦：prompt 如何构建 KV，而不生成新 token。

- **阅读：**  
  - `core/request.py`（`prefill_pos`，`get_prefill_chunk`）  
  - `model/tiny_llm.py`（`forward_prefill_dense`）
- **运行：**  
  ```bash
  python -m demos.demo_prefill_only
  ```
- **观察：** KV 长度等于 prompt 长度；注意力为稠密因果。
- **练习：** 改变 prompt 长度查看 `len(cache)`；在 `forward_prefill_dense` 打印 Q/K/V 形状。*** End Patch
