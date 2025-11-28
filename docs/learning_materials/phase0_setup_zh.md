# 阶段 0：准备

目标：环境就绪并快速浏览术语。

- **安装与测试：**  
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
  pip install -r requirements.txt
  pytest tests
  ```
- **术语速览：** `docs/get_started.md`、`docs/continuous_batching.md`
- **可查看文件：** `requirements.txt`，`tests/conftest.py`（确保导入路径）。
