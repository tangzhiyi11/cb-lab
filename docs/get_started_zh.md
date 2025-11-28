# 开始使用 cb-lab

本指南介绍项目结构、核心概念、运行示例和测试的方法，方便快速上手 continuous batching 实验。

## 1) 目录结构
```
cb_lab/           # 库代码
  core/           # 请求生命周期、调度循环、KV 缓存、ragged batch 构建
  attention/      # 稠密、ragged、paged 注意力工具
  model/          # TinyLLM 单层注意力模块
demos/            # 可运行示例：prefill、decode、mask、调度
tests/            # 基础校验：mask、KV、调度、paged 解码
cb-lab-logo.svg
README.md
docs/
```

## 2) 安装与运行
使用 Python 3.9+，CPU 版 PyTorch 即可。
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

- Ragged mask 可视化：
  ```bash
  python -m demos.demo_ragged_mask
  ```
- 调度器混合 prefill + decode：
  ```bash
  python -m demos.demo_prefill_decode_mix
  ```
- Paged 注意力解码路径：
  ```bash
  python -m demos.demo_paged_attention
  ```
- 测试：
  ```bash
  pytest tests
  ```

## 3) 更多文档
- [architecture.md](./architecture.md)：模块映射、数据流、KV/Mask 设计。
- [continuous_batching.md](./continuous_batching.md)：连续批处理概念与代码位置。
- [principle_recipes.md](./principle_recipes.md)：每个原理的验证方法与对应 demo/test。

## 4) 核心概念
- **请求生命周期 (`core/request.py`)**：跟踪 prompt、prefill 进度、生成 token、KV cache 与 decode seed；prompt 用完后进入 decode。
- **KV 缓存 (`core/kv_cache.py`)**：`DenseKVCache` 直接拼接；`PagedKVCache` 用固定块模拟 paged 布局。
- **Ragged batch (`core/batch_builder.py`)**：拼接多请求的 prompt 分片，构造布尔 ragged 因果 mask，隔离不同序列。
- **注意力 (`attention/*.py`)**：稠密因果注意力、ragged 注意力（显式 mask）、paged 解码注意力。
- **模型 (`model/tiny_llm.py`)**：单头注意力，暴露 `forward_prefill_ragged` 与 `forward_decode`。
- **调度器 (`core/scheduler.py`)**：按 token 预算优先放 decode，再填 prefill，构建 ragged batch，追加 KV，移除完成请求。

## 5) 设计取舍
- 优先透明度：无 CUDA kernel，易打印/调试。
- Ragged 优先：去掉 batch 维度，用 mask 保证隔离与因果。
- 连续批处理：decode 优先，用剩余预算填 prefill。
- KV 布局对比：dense 与 paged 并存，便于观察差异。
