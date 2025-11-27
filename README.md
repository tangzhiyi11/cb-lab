# cb-lab
A minimal learning framework to understand Continuous Batching, Ragged Batching, Dynamic Scheduling, KV Cache, and Paged Attention from scratch.

## 1. Overview

cb-lab 是一个面向学习者的、完全“从零开始”的 continuous batching 框架。
目标不是追求性能，而是让你 彻底吃透现代大模型推理系统背后的核心技术，包括：

-	KV Cache（缓存历史 K/V，降低解码成本）
-	Chunked Prefill（长 Prompt 分块，适应显存限制）
-	Ragged Batching（无 batch 维度、拼接 token、mask 控制交互）
-	Dynamic Scheduling（混合 prefill + decode 的 token-level 调度）
-	Paged Attention（稀疏化 KV 布局，decode 快路径）

它是一个 nano-scale 的 vLLM / LMDeploy 内核……用于学习，而非工程生产。

在 cb-lab 中：

- 每个组件都极简透明，可单独运行
- 所有结构、mask、KV-cache 都可打印、可观察、可调试
- 每一步调度都能看到发生了什么
- 可以逐行 debug，从核心原理彻底掌握 continuous batching

## 2. Key Features

✔ 从零构建完整 continuous batching 推理内核

你可以看到 continuous batching 的整个执行流程：

- 多请求同时 prefill
- decode 生成新 token
-	ragged 方式合并 prefill chunk
-	dynamic scheduling 填满 token budget
-	KV cache 按序追加
-	request 完成立刻替换（continuous）

✔ Ragged batching + attention mask

无需 padding；通过 ragged causal mask 保证：

- 相同序列能互相看到
- 不同序列完全隔离
- 局部 causal 约束正确

并提供 ragged 可视化工具。

✔ 简化版 paged attention（可选）

包含最小 block allocator：

- KV 按 block 存储
- decode 查询 block → 计算 q·kᵀ
- 无 mask，天然 causal

✔ 全过程可视化与 debug

提供：

-	ragged-token-table 可视化
-	ragged causal mask 可视化
-	调度 timeline 输出
-	每步 GPU token 利用率打印
