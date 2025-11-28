# Architecture Overview

cb-lab is organized around a minimal continuous batching pipeline. Each directory maps to a stage in the inference loop.

## Module map
- `core/request.py`: Request lifecycle; tracks prompt position, decode state, and KV cache.
- `core/kv_cache.py`: Dense and paged KV storage; append-only.
- `core/batch_builder.py`: Ragged token table + boolean causal mask builder.
- `core/scheduler.py`: Token-budgeted loop that mixes decode tokens and prefill chunks.
- `attention/*.py`: Dense, ragged, and paged decode attention helpers.
- `model/tiny_llm.py`: Single-layer, single-head attention block with Q/K/V/O projections.
- `monitoring/metrics.py`: Performance monitoring and metrics collection tools.
- `monitoring/memory_profiler.py`: Advanced memory profiling and leak detection.
- `plugins/base.py`: Plugin system infrastructure for extensibility.
- `plugins/builtin.py`: Built-in plugins for logging, metrics, compression, and visualization.
- `demos/*.py`: Small scripts that exercise the above components.
- `tests/*.py`: Sanity checks for masks, KV cache behavior, scheduler completion, and paged decode.

## Data flow per scheduler step
1. **Decode selection:** One decode token per in-decode request (max_new_tokens not reached).
2. **Prefill selection:** Remaining token budget is filled with prompt chunks (`prefill_chunk_size` capped by `max_tokens_per_step` minus decode count).
3. **Ragged batch build:** Concatenate prefill chunks → token table → ragged causal mask.
4. **Prefill forward:** `TinyLLM.forward_prefill_ragged` → outputs + new K/V for each chunk; append to each request's KV cache.
5. **Decode forward:** For each decode token, run `forward_decode` against the request's KV cache; append new K/V and produced token.
6. **Completion:** Mark requests finished once `max_new_tokens` reached; remove them from the active list.

## KV layouts
- **DenseKVCache:** Plain concatenation of K and V along time; simplest to inspect.
- **PagedKVCache:** Fixed-size blocks with a tiny allocator to mimic paged attention layouts; can be flattened for inspection.

## Masking strategy
- Prefill uses a ragged boolean mask so sequences are isolated without padding; causality is enforced per sequence.
- Decode path is causal by construction because each token attends to its own cached history.

## Plugin system
- **PluginManager**: Central registry for managing scheduler, attention, and cache plugins.
- **SchedulerPlugin**: Hooks for before_step, after_step, and on_request_completion events.
- **AttentionPlugin**: Custom attention computation implementations.
- **CachePlugin**: KV cache compression and optimization strategies.
- Built-in plugins include LoggingPlugin, MetricsPlugin, CacheCompressionPlugin, and AttentionVisualizationPlugin.

## Monitoring and profiling
- **MetricsCollector**: Collects step-level statistics (tokens processed, step duration, memory usage).
- **MemoryProfiler**: Tracks system and GPU memory usage with baseline comparisons.
- **DetailedMemoryProfiler**: Advanced profiling with context-aware snapshots and leak detection.
- **PerformanceBenchmark**: Utilities for measuring and comparing component performance.
