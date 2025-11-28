"""cb-lab: minimal continuous batching lab."""

__version__ = "0.2.0"
__author__ = "cb-lab Educational Framework Team"
__email__ = "contact@cb-lab.org"

# Core exports
from cb_lab.core.request import Request
from cb_lab.core.scheduler import Scheduler
from cb_lab.core.kv_cache import BaseKVCache, DenseKVCache, PagedKVCache
from cb_lab.core.batch_builder import build_ragged_batch
from cb_lab.model.tiny_llm import TinyLLM

# Monitoring exports
from cb_lab.monitoring.metrics import MetricsCollector
from cb_lab.monitoring.memory_profiler import DetailedMemoryProfiler

# Plugin exports
from cb_lab.plugins.base import (
    SchedulerPlugin,
    AttentionPlugin,
    CachePlugin,
    PluginManager,
)
from cb_lab.plugins.builtin import (
    LoggingPlugin,
    MetricsPlugin,
    AdaptiveSchedulingPlugin,
)

# Exception exports
from cb_lab.exceptions import (
    CBLabException,
    ValidationError,
    RequestError,
    SchedulerError,
    KVCacheError,
    ModelError,
    ConfigurationError,
    MemoryError,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Request",
    "Scheduler",
    "BaseKVCache",
    "DenseKVCache",
    "PagedKVCache",
    "build_ragged_batch",
    "TinyLLM",
    # Monitoring
    "MetricsCollector",
    "DetailedMemoryProfiler",
    # Plugins
    "SchedulerPlugin",
    "AttentionPlugin",
    "CachePlugin",
    "PluginManager",
    "LoggingPlugin",
    "MetricsPlugin",
    "AdaptiveSchedulingPlugin",
    # Exceptions
    "CBLabException",
    "ValidationError",
    "RequestError",
    "SchedulerError",
    "KVCacheError",
    "ModelError",
    "ConfigurationError",
    "MemoryError",
]
