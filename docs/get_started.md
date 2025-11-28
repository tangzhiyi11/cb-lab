# Get Started with cb-lab

This guide walks through the project structure, core ideas, and how to run the demos and tests. cb-lab is intentionally small so you can step through every component that makes continuous batching tick.

## 1) Project layout
```
cb_lab/           # library code
  core/           # request lifecycle, scheduler loop, KV cache, ragged batch builder
  attention/      # dense, ragged, paged attention helpers and masks
  model/          # TinyLLM single-layer attention block
  plugins/        # plugin system for extensibility
  monitoring/     # performance monitoring and memory profiling tools
demos/            # runnable examples: prefill, decode, masking, scheduling, monitoring
tests/            # sanity checks: mask, KV cache, scheduler, paged decode
benchmarks/       # performance benchmarks and scalability tests
format_code.py    # code formatting and quality checking script
cb-lab-logo.svg
README.md
docs/
```

## 2) Install & run
Use Python 3.9+; CPU-only PyTorch is fine. Recommended setup:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

- Ragged mask visualization:
  ```bash
  python -m demos.demo_ragged_mask
  ```
- Mixed prefill + decode with the scheduler:
  ```bash
  python -m demos.demo_prefill_decode_mix
  ```
- Paged attention decode path:
  ```bash
  python -m demos.demo_paged_attention
  ```
- Interactive demo with real-time parameter adjustment:
  ```bash
  python demos/interactive_demo.py
  ```
- Visualization demo:
  ```bash
  python demos/visualization_demo.py
  ```
- Performance benchmarks:
  ```bash
  python benchmarks/test_scalability.py
  ```
- Test suite:
  ```bash
  pytest tests
  ```
- Code formatting and quality check:
  ```bash
  python format_code.py --check-only
  ```

## 3) More docs
- [Architecture](./architecture.md)
- [API Reference](./api_reference.md)
- [Continuous batching](./continuous_batching.md)
- [Principles → recipes](./principle_recipes.md)
- [Learning plan](./learning_plan.md)
- [Interview Q&A](./interview_questions.md)
- Learning materials per phase: `docs/learning_materials/`
- Chinese versions: `_zh` files in `docs/`

## 4) New Features

### Plugin System
cb-lab now includes a comprehensive plugin system for extending functionality:
- **SchedulerPlugin**: Hook into scheduler lifecycle events
- **AttentionPlugin**: Implement custom attention mechanisms
- **CachePlugin**: Optimize KV cache storage and retrieval
- Built-in plugins: LoggingPlugin, MetricsPlugin, CacheCompressionPlugin

### Monitoring & Profiling
Advanced monitoring tools for performance analysis:
- **MetricsCollector**: Track throughput, latency, and resource usage
- **MemoryProfiler**: Monitor system and GPU memory consumption
- **DetailedMemoryProfiler**: Advanced profiling with leak detection
- **PerformanceBenchmark**: Compare different configurations

### Interactive Demos
New interactive demos for hands-on learning:
- **interactive_demo.py**: Real-time parameter adjustment and visualization
- **visualization_demo.py**: Generate comprehensive visualizations
- **benchmarks/**: Scalability and performance testing tools

### Development Tools
Enhanced development experience:
- **format_code.py**: Automated code formatting and quality checking
- **Comprehensive testing**: Integration tests and edge case validation
- **Type safety**: Full mypy type annotation support

## 3) Core concepts
- **Request lifecycle (`core/request.py`):** tracks prompt tokens, prefill position, generated tokens, KV cache, and decode seed. Switches from prefill to decode once the prompt is consumed.
- **KV cache (`core/kv_cache.py`):** two backends:
  - `DenseKVCache`: append-only tensors for K/V.
  - `PagedKVCache`: tiny block allocator that stores K/V in fixed-size blocks, mimicking paged attention layouts.
- **Ragged batch builder (`core/batch_builder.py`):** concatenates prompt chunks across requests and builds a boolean ragged causal mask so sequences stay isolated without padding.
- **Attention helpers (`attention/*.py`):**
  - Dense causal attention for single-sequence prefill.
  - Ragged attention that consumes the ragged mask.
  - Paged decode attention that reads from the KV cache (dense or paged).
- **Model (`model/tiny_llm.py`):** single-head attention block exposing:
  - `forward_prefill_ragged(tokens, mask) -> out, K, V`
  - `forward_decode(new_token, kv_cache) -> out, K_new, V_new`
- **Scheduler (`core/scheduler.py`):** continuous batching loop that:
  1) Takes one decode token per in-decode request.
  2) Fills remaining token budget with prefill chunks.
  3) Builds ragged batch + mask for prefill, runs model, appends K/V.
  4) Runs decode attention for decode tokens and appends K/V.
  5) Marks finished requests and keeps going until empty.

## 4) How the demos map to concepts
- `demo_prefill_only.py`: dense prefill of a single prompt, KV cache growth.
- `demo_ragged_mask.py`: prints token table and ragged causal mask for multiple sequences.
- `demo_prefill_decode_mix.py`: mixes prefill and decode in one loop, showing token budgets and request completion.
- `demo_scheduler_timeline.py`: step-by-step scheduling log over multiple requests.
- `demo_paged_attention.py`: shows paged KV layout and decode attention reading from paged blocks.

## 5) Design notes
- **Transparency over performance:** everything is eager, printed, and easy to inspect; no CUDA kernels.
- **Ragged first-class:** batch dimension is removed; masks enforce isolation and causality.
- **Continuous batching scheduling:** decode tokens prioritized, prefill chunks fill the leftover budget; this mirrors modern inference engines in a tiny form factor.
- **KV layout focus:** both dense and paged caches exist so you can contrast data layouts and decode behavior.

## 6) Next steps
- Tweak `max_tokens_per_step` and `prefill_chunk_size` in `core/scheduler.py` to see different scheduling behaviors.
- Add more requests or longer prompts in demos to watch the ragged mask and KV cache scale.
- Instrument prints or step through with a debugger to deepen intuition about continuous batching internals.
