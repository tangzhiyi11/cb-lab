# cb-lab API Reference

This document provides comprehensive API documentation for the cb-lab educational framework.

## Table of Contents

- [Core Components](#core-components)
  - [Request](#request)
  - [Scheduler](#scheduler)
  - [KV Cache](#kv-cache)
  - [Batch Builder](#batch-builder)
- [Model Components](#model-components)
  - [TinyLLM](#tinylm)
- [Attention Mechanisms](#attention-mechanisms)
- [Plugin System](#plugin-system)
  - [PluginManager](#pluginmanager)
  - [Built-in Plugins](#built-in-plugins)
- [Monitoring and Metrics](#monitoring-and-metrics)
  - [Performance Metrics](#performance-metrics)
  - [Memory Profiling](#memory-profiling)
- [Exceptions](#exceptions)
- [Examples](#examples)

## Core Components

### Request

The `Request` class represents a single inference request flowing through the continuous batching system.

```python
from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class Request:
    req_id: str
    prompt: torch.Tensor
    max_new_tokens: int
    kv_cache: BaseKVCache
    prefill_pos: int = 0
    generated_tokens: List[torch.Tensor] = field(default_factory=list)
    finished: bool = False
    decode_seed: Optional[torch.Tensor] = None
```

#### Attributes

- **req_id** (`str`): Unique identifier for the request
- **prompt** (`torch.Tensor`): Input prompt tensor of shape [seq_len, dim]
- **max_new_tokens** (`int`): Maximum number of tokens to generate
- **kv_cache** (`BaseKVCache`): KV cache for this request
- **prefill_pos** (`int`): Current position in prompt processing (default: 0)
- **generated_tokens** (`List[torch.Tensor]`): List of generated tokens (default: empty)
- **finished** (`bool`): Whether request has completed (default: False)
- **decode_seed** (`Optional[torch.Tensor]`): Initial token for decode phase (default: None)

#### Properties

- **in_prefill** (`bool`): True if request is still in prefill phase
- **in_decode** (`bool`): True if request is in decode phase

#### Methods

##### get_prefill_chunk(chunk_size: int) -> torch.Tensor

Returns a slice of the prompt to prefill.

**Parameters:**
- `chunk_size` (`int`): Maximum number of tokens to return

**Returns:**
- `torch.Tensor`: Tensor of shape [chunk_len, dim] where chunk_len ≤ chunk_size

**Example:**
```python
req = Request("test", torch.randn(10, 16), 5, cache)
chunk = req.get_prefill_chunk(3)  # Get up to 3 tokens for prefill
```

##### append_kv(k_new: torch.Tensor, v_new: torch.Tensor) -> None

Appends newly computed keys/values to the KV cache.

**Parameters:**
- `k_new` (`torch.Tensor`): New keys tensor of shape [seq_len, dim]
- `v_new` (`torch.Tensor`): New values tensor of shape [seq_len, dim]

**Raises:**
- `KVCacheError`: If KV cache append operation fails
- `ValidationError`: If input tensors are invalid

##### append_token(tok: torch.Tensor) -> None

Appends a generated token to the request.

**Parameters:**
- `tok` (`torch.Tensor`): Generated token tensor of shape [dim] or [1, dim]

**Raises:**
- `ValidationError`: If token tensor is invalid

### Scheduler

The `Scheduler` class manages continuous batching by mixing prefill and decode operations.

```python
class Scheduler:
    def __init__(
        self,
        model: TinyLLM,
        max_tokens_per_step: int = 8,
        prefill_chunk_size: int = 4,
    ) -> None:
```

#### Parameters

- **model** (`TinyLLM`): The language model to use for inference
- **max_tokens_per_step** (`int`): Maximum tokens to process in each step (default: 8)
- **prefill_chunk_size** (`int`): Maximum chunk size for prefill operations (default: 4)

#### Methods

##### add_request(req: Request) -> None

Adds a new request to the scheduler.

**Parameters:**
- `req` (`Request`): Request to add to the active pool

**Raises:**
- `SchedulerError`: If request cannot be added

**Example:**
```python
scheduler = Scheduler(model, max_tokens_per_step=6)
req = Request("user1", prompt, 5, cache)
scheduler.add_request(req)
```

##### step() -> Dict[str, Any]

Executes one step of continuous batching, mixing prefill and decode operations.

**Returns:**
- `Dict[str, Any]`: Dictionary with step statistics and metrics

**Example:**
```python
while scheduler.active:
    stats = scheduler.step()
    print(f"Step {stats['step_id']}: processed {stats['total_tokens']} tokens")
```

### KV Cache

#### BaseKVCache (Abstract)

Abstract base class for KV cache implementations.

```python
class BaseKVCache(ABC):
    @abstractmethod
    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        ...

    @abstractmethod
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
```

#### DenseKVCache

Append-only dense KV cache used for prefill and decode.

```python
class DenseKVCache(BaseKVCache):
    def __init__(self, device: Optional[torch.device] = None) -> None:
```

**Example:**
```python
cache = DenseKVCache(torch.device("cpu"))
cache.append(k_new, v_new)
k, v = cache.get_kv()
```

#### PagedKVCache

KV cache backed by paged blocks for better memory efficiency.

```python
class PagedKVCache(BaseKVCache):
    def __init__(self, block_size: int = 64, device: Optional[torch.device] = None) -> None:
```

**Parameters:**
- `block_size` (`int`): Number of token positions per block (default: 64)
- `device` (`Optional[torch.device]`): Device to store tensors on (default: CPU)

**Example:**
```python
cache = PagedKVCache(block_size=32, device=torch.device("cuda"))
cache.append(k_new, v_new)
k, v = cache.get_kv()
```

### Batch Builder

#### build_ragged_batch()

Builds ragged batches for variable-length sequences.

```python
def build_ragged_batch(
    chunks: Sequence[Tuple[str, int, torch.Tensor]],
) -> Tuple[torch.Tensor, List[TokenMeta], torch.Tensor]:
```

**Parameters:**
- `chunks` (`Sequence[Tuple[str, int, torch.Tensor]]`): Sequence of tuples (req_id, start_pos, tokens_chunk)

**Returns:**
- `Tuple[torch.Tensor, List[TokenMeta], torch.Tensor]`:
  - Concatenated tokens tensor of shape [N_total, d]
  - List of TokenMeta objects with metadata per token
  - Boolean causal mask of shape [N_total, N_total]

**Example:**
```python
chunks = [
    ("req1", 0, torch.randn(3, 16)),
    ("req2", 0, torch.randn(2, 16)),
]
tokens, meta, mask = build_ragged_batch(chunks)
```

## Model Components

### TinyLLM

A minimal single-layer attention model used for demonstrations.

```python
class TinyLLM(nn.Module):
    def __init__(self, dim: int = 16) -> None:
```

#### Parameters

- **dim** (`int`): Model dimension (default: 16)

#### Methods

##### forward_prefill_ragged(tokens, ragged_mask)

Prefill path with ragged attention over concatenated prompt chunks.

**Parameters:**
- `tokens` (`torch.Tensor`): Input tokens of shape [N_total, dim]
- `ragged_mask` (`torch.Tensor`): Boolean causal mask of shape [N_total, N_total]

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: (output, keys, values)

##### forward_decode(new_token, kv_cache)

Decode path for single token generation.

**Parameters:**
- `new_token` (`torch.Tensor`): New token to process
- `kv_cache` (`BaseKVCache`): KV cache for attention

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: (output, new_keys, new_values)

## Plugin System

### PluginManager

Central manager for organizing and executing plugins across different phases of the continuous batching pipeline.

```python
class PluginManager:
    def __init__(self) -> None:
```

#### Methods

##### register_scheduler_plugin(plugin, config=None)

Register a scheduler plugin with optional configuration.

**Parameters:**
- `plugin` (`SchedulerPlugin`): Plugin instance to register
- `config` (`Optional[Dict[str, Any]]`): Plugin configuration (default: None)

##### register_attention_plugin(plugin, config=None)

Register an attention plugin with optional configuration.

**Parameters:**
- `plugin` (`AttentionPlugin`): Plugin instance to register
- `config` (`Optional[Dict[str, Any]]`): Plugin configuration (default: None)

##### register_cache_plugin(plugin, config=None)

Register a cache plugin with optional configuration.

**Parameters:**
- `plugin` (`CachePlugin`): Plugin instance to register
- `config` (`Optional[Dict[str, Any]]`): Plugin configuration (default: None)

##### get_plugin_config(plugin_name) -> Dict[str, Any]

Get configuration for a specific plugin.

**Parameters:**
- `plugin_name` (`str`): Name of the plugin

**Returns:**
- `Dict[str, Any]`: Plugin configuration dictionary

##### set_plugin_config(plugin_name, config)

Set configuration for a specific plugin.

**Parameters:**
- `plugin_name` (`str`): Name of the plugin
- `config` (`Dict[str, Any]`): Configuration dictionary

##### list_plugins() -> Dict[str, List[str]]

List all registered plugins by type.

**Returns:**
- `Dict[str, List[str]]`: Dictionary with plugin names by category

**Example:**
```python
from cb_lab.plugins.base import PluginManager
from cb_lab.plugins.builtin import LoggingPlugin, MetricsPlugin

manager = PluginManager()
manager.register_scheduler_plugin(LoggingPlugin())
manager.register_scheduler_plugin(MetricsPlugin())

plugins = manager.list_plugins()
print(f"Registered scheduler plugins: {plugins['scheduler']}")
```

### Built-in Plugins

#### LoggingPlugin

Simple logging plugin for scheduler events.

```python
class LoggingPlugin(SchedulerPlugin):
    def __init__(self, log_level: str = "INFO"):
```

**Parameters:**
- `log_level` (`str`): Logging level (default: "INFO")

#### MetricsPlugin

Detailed metrics collection plugin.

```python
class MetricsPlugin(SchedulerPlugin):
    def __init__(self) -> None:
```

#### CacheCompressionPlugin

KV cache compression plugin for memory optimization.

```python
class CacheCompressionPlugin(CachePlugin):
    def __init__(self, compression_ratio: float = 0.5):
```

**Parameters:**
- `compression_ratio` (`float`): Target compression ratio (default: 0.5)

#### AttentionVisualizationPlugin

Plugin for visualizing attention patterns.

```python
class AttentionVisualizationPlugin(AttentionPlugin):
    def __init__(self, save_dir: str = "attention_viz"):
```

**Parameters:**
- `save_dir` (`str`): Directory to save visualizations (default: "attention_viz")

## Monitoring and Metrics

### MetricsCollector

Centralized metrics collection for cb-lab operations.

```python
class MetricsCollector:
    def __init__(self, enable_memory_profiling: bool = True):
```

#### Methods

##### record_step()

Record step completion metrics.

```python
def record_step(
    self,
    step_id: int,
    decode_tokens: int,
    prefill_tokens: int,
    active_requests: int,
    finished_requests: int,
    step_duration: float,
    status: str = "completed"
) -> StepMetrics:
```

##### get_step_summary()

Get summary statistics for all recorded steps.

**Returns:**
- `Dict[str, Any]`: Summary statistics including throughput, duration, etc.

##### start_request_tracking(req_id, total_tokens)

Start tracking a new request.

**Parameters:**
- `req_id` (`str`): Request identifier
- `total_tokens` (`int`): Expected total tokens for the request

## Memory Profiling

### MemoryProfiler

Basic memory usage profiler for monitoring system and GPU memory.

```python
class MemoryProfiler:
    def __init__(self) -> None:
```

#### Methods

##### get_current_memory() -> float

Get current system memory usage in MB.

**Returns:**
- `float`: Current memory usage in MB

##### get_gpu_memory() -> float

Get current GPU memory usage in MB.

**Returns:**
- `float`: GPU memory usage in MB (0.0 if no GPU available)

##### get_memory_delta() -> float

Get memory usage change from baseline in MB.

**Returns:**
- `float`: Memory delta from baseline

##### set_baseline() -> None

Set baseline memory measurement.

##### log_memory_usage(context="") -> Dict[str, float]

Log current memory usage with context.

**Parameters:**
- `context` (`str`): Optional context label

**Returns:**
- `Dict[str, float]`: Memory usage statistics

### DetailedMemoryProfiler

Advanced memory profiler with snapshot capabilities and leak detection.

```python
class DetailedMemoryProfiler:
    def __init__(self, sampling_interval: float = 0.1):
```

**Parameters:**
- `sampling_interval` (`float`): Sampling interval in seconds (default: 0.1)

#### Methods

##### take_snapshot(context="") -> MemorySnapshot

Take a snapshot of current memory usage.

**Parameters:**
- `context` (`str`): Context label for the snapshot

**Returns:**
- `MemorySnapshot`: Memory usage snapshot

##### set_baseline(context="baseline") -> None

Set baseline memory measurement.

**Parameters:**
- `context` (`str`): Context label for baseline

##### get_memory_delta(snapshot=None) -> Dict[str, float]

Get memory usage delta from baseline.

**Parameters:**
- `snapshot` (`Optional[MemorySnapshot]`): Specific snapshot to compare (default: latest)

**Returns:**
- `Dict[str, float]`: Memory deltas

##### get_peak_memory_usage() -> Dict[str, float]

Get peak memory usage statistics.

**Returns:**
- `Dict[str, float]`: Peak memory statistics

##### profile_context(context_name) -> ContextManager

Context manager for profiling specific code blocks.

**Parameters:**
- `context_name` (`str`): Name for the profiling context

**Example:**
```python
from cb_lab.monitoring.memory_profiler import DetailedMemoryProfiler

profiler = DetailedMemoryProfiler()

with profiler.profile_context("heavy_computation"):
    # Your code here
    result = some_heavy_function()

print(f"Memory delta: {profiler.get_memory_delta()}")
```

### MemoryLeakDetector

Tool for detecting potential memory leaks.

```python
class MemoryLeakDetector:
    def __init__(self, threshold_mb: float = 10.0):
```

**Parameters:**
- `threshold_mb` (`float`): Memory growth threshold in MB (default: 10.0)

#### Methods

##### set_baseline() -> None

Set baseline memory measurement.

##### record_measurement(label="") -> None

Record a memory measurement.

**Parameters:**
- `label` (`str`): Optional label for the measurement

##### check_for_leaks() -> Dict[str, Any]

Check if memory usage indicates a potential leak.

**Returns:**
- `Dict[str, Any]`: Leak detection results

## Exceptions

cb-lab provides a hierarchical exception system:

### Base Exception

- **CBLabException**: Base exception for all cb-lab errors

### Specific Exceptions

- **ValidationError**: Raised when input validation fails
- **RequestError**: Raised when request processing fails
- **SchedulerError**: Raised when scheduler operations fail
- **KVCacheError**: Raised when KV cache operations fail
- **ModelError**: Raised when model operations fail
- **ConfigurationError**: Raised when configuration is invalid
- **MemoryError**: Raised when memory operations fail

## Examples

### Basic Usage

```python
import torch
from cb_lab.core.scheduler import Scheduler
from cb_lab.core.request import Request
from cb_lab.core.kv_cache import DenseKVCache
from cb_lab.model.tiny_llm import TinyLLM

# Create model and scheduler
model = TinyLLM(dim=16)
scheduler = Scheduler(model, max_tokens_per_step=8, prefill_chunk_size=4)

# Create request
device = torch.device("cpu")
prompt = torch.randn(10, 16, device=device)
cache = DenseKVCache(device)
request = Request("user1", prompt, 5, cache)

# Add request and run
scheduler.add_request(request)

while scheduler.active:
    stats = scheduler.step()
    print(f"Step {stats['step_id']}: {stats['decode_tokens']} decode, {stats['prefill_tokens']} prefill")
```

### Using Metrics

```python
from cb_lab.monitoring.metrics import MetricsCollector

metrics = MetricsCollector()

# Start tracking request
metrics.start_request_tracking("user1", total_tokens=15)

# During execution
for req in requests:
    if req.in_decode:
        metrics.update_request_progress(req.req_id, decode_tokens=len(req.generated_tokens))

# Get summary
summary = metrics.get_step_summary()
print(f"Processed {summary['total_tokens_processed']} tokens at {summary['tokens_per_second']:.2f} tok/s")
```

### Working with Different Cache Types

```python
from cb_lab.core.kv_cache import DenseKVCache, PagedKVCache

# Dense cache (simple, contiguous storage)
dense_cache = DenseKVCache(torch.device("cpu"))

# Paged cache (efficient for variable sequences)
paged_cache = PagedKVCache(block_size=32, device=torch.device("cuda"))

# Use either cache type with requests
req1 = Request("user1", prompt1, 5, dense_cache)
req2 = Request("user2", prompt2, 3, paged_cache)
```

### Error Handling

```python
from cb_lab.exceptions import RequestError, ValidationError

try:
    req = Request("test", invalid_prompt, 5, cache)
except ValidationError as e:
    print(f"Validation failed: {e}")
except RequestError as e:
    print(f"Request error: {e}")
```

### Using Plugins

```python
from cb_lab.plugins.base import PluginManager
from cb_lab.plugins.builtin import LoggingPlugin, MetricsPlugin, CacheCompressionPlugin

# Create plugin manager
manager = PluginManager()

# Register plugins
manager.register_scheduler_plugin(LoggingPlugin("DEBUG"))
manager.register_scheduler_plugin(MetricsPlugin())
manager.register_cache_plugin(CacheCompressionPlugin(0.7))

# Set plugin configurations
manager.set_plugin_config("LoggingPlugin", {"log_level": "INFO"})

# List registered plugins
plugins = manager.list_plugins()
print(f"Plugins: {plugins}")
```

### Memory Profiling

```python
from cb_lab.monitoring.memory_profiler import DetailedMemoryProfiler, MemoryLeakDetector

# Basic memory profiling
profiler = DetailedMemoryProfiler()
profiler.set_baseline()

with profiler.profile_context("inference"):
    # Run your inference code
    scheduler.step()

delta = profiler.get_memory_delta()
print(f"Memory delta: {delta['rss_delta_mb']:.2f} MB")

# Leak detection
detector = MemoryLeakDetector(threshold_mb=5.0)
detector.set_baseline()

for i in range(100):
    run_inference_step()
    detector.record_measurement(f"step_{i}")

results = detector.check_for_leaks()
if results['status'] == 'leak_detected':
    print("Potential memory leak detected!")
```

## Performance Tips

1. **Choose appropriate cache types**: Use `DenseKVCache` for simplicity, `PagedKVCache` for memory efficiency
2. **Tune batch sizes**: Adjust `max_tokens_per_step` and `prefill_chunk_size` for your workload
3. **Monitor performance**: Use `MetricsCollector` to track throughput and identify bottlenecks
4. **Handle errors gracefully**: Use try-catch blocks around request operations
5. **Resource management**: Monitor memory usage, especially with large models or many requests
6. **Use plugins for profiling**: Leverage built-in plugins like `MetricsPlugin` and `LoggingPlugin` for better insights
7. **Memory leak detection**: Use `MemoryLeakDetector` for long-running applications
8. **Cache compression**: Apply `CacheCompressionPlugin` when memory is constrained
9. **Batch optimization**: Experiment with different chunk sizes to find optimal throughput
10. **Context-aware profiling**: Use `DetailedMemoryProfiler.profile_context()` for targeted optimization