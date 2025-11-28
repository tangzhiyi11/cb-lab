# 开发指南

本文档介绍 cb-lab 项目的开发规范和工具使用。

## 代码格式化

项目使用以下工具确保代码质量：

- **black**: 代码格式化工具，统一代码风格
- **flake8**: 代码质量检查，检查语法错误和风格问题
- **mypy**: 类型注解检查，确保类型安全

### 快速开始

```bash
# 安装开发依赖
pip install black flake8 mypy

# 检查代码质量（不修改文件）
python format_code.py --check-only

# 自动修复格式问题
python format_code.py --fix

# 查看项目统计信息
python format_code.py --stats
```

### 手动运行工具

```bash
# 格式化所有Python文件
python -m black --line-length 88 cb_lab/ demos/ tests/ benchmarks/

# 检查代码质量
python -m flake8 --max-line-length 88 --extend-ignore E203,W503 cb_lab/ demos/ tests/ benchmarks/

# 检查类型注解
python -m mypy cb_lab/
```

## 代码规范

### 格式规范

- 行长度限制：88 字符
- 使用 black 自动格式化
- 遵循 PEP 8 风格指南

### 类型注解

- 所有公共函数必须包含类型注解
- 使用 `typing` 模块的类型
- 复杂类型使用 `TypeVar` 和泛型

### 导入规范

- 导入顺序：标准库 → 第三方库 → 本地模块
- 每组导入之间用空行分隔
- 避免未使用的导入

### 文档字符串

- 所有公共函数和类必须包含文档字符串
- 使用 Google 风格的文档字符串
- 包含参数说明、返回值和异常

## 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_integration.py

# 运行测试并生成覆盖率报告
python -m pytest tests/ --cov=cb_lab --cov-report=html
```

## 提交规范

提交前请确保：

1. 所有测试通过
2. 代码格式化检查通过
3. 没有类型注解错误
4. 更新了相关文档

```bash
# 完整检查流程
python format_code.py --fix
python -m pytest tests/
python -m mypy cb_lab/
```

## 开发环境设置

```bash
# 克隆项目
git clone <repository-url>
cd cb-lab

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
pip install -e ".[dev]"  # 安装开发依赖

# 运行快速验证
python demos/demo_prefill_decode_mix.py
```

## 项目结构

```
cb_lab/
├── cb_lab/           # 核心代码
│   ├── core/         # 核心模块
│   ├── model/        # 模型定义
│   ├── attention/    # 注意力机制
│   ├── monitoring/   # 监控工具
│   ├── plugins/      # 插件系统
│   └── exceptions.py # 异常定义
├── demos/            # 演示脚本
├── tests/            # 测试代码
├── benchmarks/       # 性能基准测试
└── format_code.py    # 代码格式化脚本
```

## 性能优化

- 使用 `torch.jit.script` 编译关键函数
- 合理使用批处理减少GPU调用
- 监控内存使用，避免内存泄漏
- 使用 `cb_lab.monitoring` 模块分析性能

## 常见问题

### 类型检查错误

如果遇到 mypy 错误：

1. 检查是否正确导入类型
2. 添加适当的类型注解
3. 使用 `# type: ignore` 忽略无法避免的错误

### 格式化问题

如果 black 格式化后的代码不符合预期：

1. 检查是否有语法错误
2. 确保所有括号、引号正确配对
3. 运行 `python -m black --check` 预览更改

### 测试失败

如果测试失败：

1. 检查环境设置是否正确
2. 确保所有依赖已安装
3. 查看详细错误信息进行调试