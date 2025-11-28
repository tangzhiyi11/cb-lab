#!/usr/bin/env python3
"""Setup script for cb-lab educational framework."""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    version_file = Path(__file__).parent / "cb_lab" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read README for long description
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Get requirements
def get_requirements():
    requirements = []
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, "r", encoding="utf-8") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

# Get development requirements
def get_dev_requirements():
    dev_requirements = [
        "pytest>=6.0.0",
        "pytest-cov>=2.0.0",
        "black>=21.0.0",
        "flake8>=3.8.0",
        "mypy>=0.900",
        "pre-commit>=2.0.0",
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "matplotlib>=3.3.0",
        "psutil>=5.8.0",
    ]
    return dev_requirements

# Get optional requirements for specific features
def get_optional_requirements():
    optional = {
        "monitoring": [
            "matplotlib>=3.3.0",
            "psutil>=5.8.0",
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "benchmarking": [
            "matplotlib>=3.3.0",
            "numpy>=1.21.0",
        ],
        "dev": get_dev_requirements(),
    }
    return optional

setup(
    name="cb-lab",
    version=get_version(),
    author="cb-lab Educational Framework Team",
    author_email="contact@cb-lab.org",
    description="A tiny learning framework for continuous batching inference",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/cb-lab",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/cb-lab/issues",
        "Documentation": "https://cb-lab.readthedocs.io/",
        "Source Code": "https://github.com/your-org/cb-lab",
    },
    packages=find_packages(exclude=["tests", "tests.*", "demos", "benchmarks"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require=get_optional_requirements(),
    entry_points={
        "console_scripts": [
            "cb-lab-demo=demos.interactive_demo:main",
            "cb-lab-viz=demos.visualization_demo:run_visualization_demo",
            "cb-lab-bench=benchmarks.test_scalability:run_comprehensive_benchmarks",
        ],
    },
    include_package_data=True,
    package_data={
        "cb_lab": [
            "py.typed",
        ],
    },
    keywords=[
        "machine learning",
        "deep learning",
        "transformers",
        "continuous batching",
        "inference",
        "educational",
        "llm",
        "attention",
    ],
    zip_safe=False,
    platforms=["any"],
)