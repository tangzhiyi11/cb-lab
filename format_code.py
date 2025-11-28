#!/usr/bin/env python3
"""代码格式化和质量检查脚本。

此脚本会自动运行以下工具：
1. black - 代码格式化
2. flake8 - 代码质量检查
3. mypy - 类型注解检查

使用方法:
    python format_code.py [--fix] [--check-only]

选项:
    --fix: 自动修复代码格式问题
    --check-only: 仅检查代码质量，不进行格式化
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class CodeFormatter:
    """代码格式化和质量检查工具。"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_files = list(self.project_root.glob("cb_lab/**/*.py"))
        self.demo_files = list(self.project_root.glob("demos/**/*.py"))
        self.test_files = list(self.project_root.glob("tests/**/*.py"))
        self.benchmark_files = list(self.project_root.glob("benchmarks/**/*.py"))

        # 所有Python文件
        self.all_files = (
            self.python_files
            + self.demo_files
            + self.test_files
            + self.benchmark_files
            + [self.project_root / "format_code.py"]
        )

    def run_command(self, cmd: List[str], description: str) -> Tuple[int, str, str]:
        """运行命令并返回结果。"""
        print(f"🔧 {description}...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                print(f"✅ {description} 完成")
            else:
                print(f"❌ {description} 失败")
                if result.stdout:
                    print(f"输出: {result.stdout}")
                if result.stderr:
                    print(f"错误: {result.stderr}")
            return result.returncode, result.stdout, result.stderr
        except FileNotFoundError:
            print(f"❌ 找不到命令: {cmd[0]}")
            print("请确保已安装所需的工具: pip install black flake8 mypy")
            return 1, "", "Command not found"

    def format_with_black(self, fix: bool = True) -> bool:
        """使用black格式化代码。"""
        if fix:
            cmd = ["python", "-m", "black", "--line-length", "88"] + [
                str(f) for f in self.all_files
            ]
        else:
            cmd = ["python", "-m", "black", "--check", "--line-length", "88"] + [
                str(f) for f in self.all_files
            ]

        action = "格式化" if fix else "检查格式"
        returncode, _, _ = self.run_command(cmd, f"使用black {action}")
        return returncode == 0

    def check_with_flake8(self) -> bool:
        """使用flake8检查代码质量。"""
        cmd = [
            "python",
            "-m",
            "flake8",
            "--max-line-length",
            "88",
            "--extend-ignore",
            "E203,W503",
        ] + [str(f) for f in self.all_files]
        returncode, _, _ = self.run_command(cmd, "使用flake8检查代码质量")
        return returncode == 0

    def check_with_mypy(self) -> bool:
        """使用mypy检查类型注解。"""
        cmd = ["python", "-m", "mypy", "cb_lab/"]
        returncode, _, _ = self.run_command(cmd, "使用mypy检查类型注解")
        return returncode == 0

    def run_all_checks(self, fix: bool = False, check_only: bool = False) -> bool:
        """运行所有代码质量检查。"""
        print("🎯 cb-lab 代码格式化和质量检查")
        print("=" * 50)

        all_passed = True

        if not check_only:
            # Black 格式化
            if not self.format_with_black(fix=fix):
                all_passed = False

        # Flake8 检查
        if not self.check_with_flake8():
            all_passed = False

        # MyPy 检查
        if not self.check_with_mypy():
            all_passed = False

        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 所有检查都通过了！代码质量良好。")
        else:
            print("⚠️  发现了一些问题，请查看上述输出进行修复。")
            if fix and not check_only:
                print("💡 提示：运行 'python format_code.py --fix' 来自动修复格式问题")

        return all_passed

    def print_stats(self):
        """打印项目统计信息。"""
        print(f"📊 项目统计:")
        print(f"  核心文件: {len(self.python_files)}")
        print(f"  演示文件: {len(self.demo_files)}")
        print(f"  测试文件: {len(self.test_files)}")
        print(f"  基准测试文件: {len(self.benchmark_files)}")
        print(f"  总计Python文件: {len(self.all_files)}")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="cb-lab 代码格式化和质量检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python format_code.py              # 检查所有问题
    python format_code.py --fix        # 自动修复格式问题
    python format_code.py --check-only # 仅检查，不修改文件
        """,
    )

    parser.add_argument("--fix", action="store_true", help="自动修复代码格式问题")

    parser.add_argument(
        "--check-only", action="store_true", help="仅检查代码质量，不进行格式化"
    )

    parser.add_argument("--stats", action="store_true", help="显示项目统计信息")

    args = parser.parse_args()

    formatter = CodeFormatter()

    if args.stats:
        formatter.print_stats()
        return

    success = formatter.run_all_checks(fix=args.fix, check_only=args.check_only)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
