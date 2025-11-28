# 阶段 6：注意力路径

聚焦：prefill 与 decode 使用的不同注意力分支。

- **阅读：**  
  - `attention/dense_attention.py`（单序列因果）  
  - `attention/ragged_attention.py`（依赖 mask）  
  - `attention/paged_attention.py`（缓存上的解码）
- **练习：** 对单序列分别用 dense 与 ragged（构造单序列 mask），比较输出确认一致；打印 Q/K/V 与分数观察 mask 作用。
- **关联：** `model/tiny_llm.py` 通过 `forward_prefill_ragged`、`forward_decode` 连接这些分支。*** End Patch
