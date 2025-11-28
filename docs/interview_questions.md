# Continuous Batching Interview Q&A

From warm-up to deeper topics. Use cb-lab to reason about answers or demonstrate with code.

## Basics
- **Q:** What is continuous batching?  
  **A:** Dynamically batching requests at token boundaries so new requests can join and finished ones can leave without waiting for a static batch cycle.
- **Q:** Why separate prefill and decode?  
  **A:** Prefill is long and parallel; decode is short and sequential. Separating allows mixing to keep utilization high while controlling latency.
- **Q:** What is a KV cache?  
  **A:** Stored keys/values from past tokens so decode only computes attention against history instead of recomputing the full sequence.

## Mechanics
- **Q:** How does token budgeting work per step?  
  **A:** A step has a token budget; decode tokens (1 per active decode request) are scheduled first, then prefill chunks fill remaining capacity (`core/scheduler.py`).
- **Q:** How do you avoid padding overhead across requests?  
  **A:** Use ragged batching: concatenate tokens and apply a boolean ragged causal mask so sequences are isolated without padding (`core/batch_builder.py`).
- **Q:** How is causality enforced in ragged batching?  
  **A:** Mask only allows positions where `seq(i)==seq(j)` and `pos(j)<=pos(i)`; others are False.
- **Q:** How does the model transition from prefill to decode?  
  **A:** Track `prefill_pos`; once prompt is consumed, decode starts using the last prompt token (or provided seed) and appends outputs until `max_new_tokens`.

## Performance and trade-offs
- **Q:** Throughput vs. latency trade-offs?  
  **A:** Larger `max_tokens_per_step` improves throughput but increases per-step latency; smaller values reduce latency but can underutilize the GPU.
- **Q:** How does prefill chunk size affect latency?  
  **A:** Smaller chunks reduce time-to-first-token; larger chunks reduce overhead but may delay decode if they fill the budget.
- **Q:** What happens if decode demand exceeds the token budget?  
  **A:** Only up to the budgeted decode tokens run; remaining decode requests wait for the next step (see decode cap in `core/scheduler.py`).

## KV layout
- **Q:** Dense vs. paged KV cache?  
  **A:** Dense concatenates K/V; simple and contiguous. Paged stores fixed-size blocks to mirror paged attention layouts, reducing movement on decode for long contexts (`core/kv_cache.py`).
- **Q:** How do you flatten paged KV for attention?  
  **A:** Concatenate blocks and trim unused tail; decode attends over the flattened view (`PagedAllocator.flatten`).

## Failure modes and correctness
- **Q:** How to ensure requests are isolated in mixed batches?  
  **A:** Correct ragged mask construction and per-request KV caches; cross-request attention is blocked.
- **Q:** How to validate scheduler correctness?  
  **A:** Unit tests that assert all requests finish and budgets are respected (`tests/test_scheduler.py`).
- **Q:** What about numerical stability in attention?  
  **A:** Use masking with large negative values before softmax and small EPS clamps (`attention/*.py`).

## Advanced/Design
- **Q:** How would you add prioritization (e.g., SLA-aware scheduling)?  
  **A:** Modify selection to order requests by priority when picking decode/prefill, possibly reserving budget slices for high-priority items.
- **Q:** How to support variable decode batch sizes?  
  **A:** Allow more than one decode token per request per step, adjusting budget accounting and KV appends.
- **Q:** How would you shard KV across devices?
  **A:** Introduce partitioned KV caches and route attention queries to shards; requires cross-device gather or hierarchical attention—outside cb-lab scope but conceptually similar with distributed KV access.

## Plugin System
- **Q:** What's the purpose of a plugin system in continuous batching?
  **A:** Allows extending functionality without modifying core components - useful for logging, metrics, custom attention, cache optimization, and business logic integration.
- **Q:** How do scheduler plugins work?
  **A:** They hook into scheduler lifecycle events: `before_step()`, `after_step()`, and `on_request_completion()`. Can access scheduler state and modify behavior.
- **Q:** What's the difference between SchedulerPlugin and AttentionPlugin?
  **A:** SchedulerPlugin hooks into scheduling logic; AttentionPlugin replaces or augments attention computation itself.
- **Q:** How would you implement a custom cache optimization plugin?
  **A:** Implement CachePlugin interface with `before_append()` and `after_append()` hooks; could implement compression, deduplication, or custom eviction strategies.

## Monitoring and Profiling
- **Q:** Why is monitoring important in continuous batching systems?
  **A:** To track utilization, identify bottlenecks, detect memory leaks, ensure SLAs, and optimize performance parameters like batch sizes.
- **Q:** How does MetricsCollector work?
  **A:** Tracks step-level statistics (decode/prefill tokens, duration, memory usage) and can compute request-level metrics like throughput and latency.
- **Q:** What's the difference between MemoryProfiler and DetailedMemoryProfiler?
  **A:** MemoryProfiler provides basic memory tracking; DetailedMemoryProfiler offers snapshot-based profiling with context-aware tracking and leak detection.
- **Q:** How would you detect memory leaks in a continuous batching system?
  **A:** Use MemoryLeakDetector to track memory growth patterns, or DetailedMemoryProfiler to analyze memory deltas across different contexts and identify unexpected growth.

## Performance and Scalability
- **Q:** How do you measure the efficiency of continuous batching?
  **A:** Track metrics like token throughput, request latency, GPU utilization, memory efficiency, and compare against baseline static batching.
- **Q:** What factors affect scalability of a continuous batching system?
  **A:** Token budget size, request mix (prompt/generate ratios), KV cache efficiency, memory bandwidth, and GPU compute capability.
- **Q:** How would you optimize for different workload patterns?
  **A:** Adjust `max_tokens_per_step` and `prefill_chunk_size` based on request characteristics; use different cache strategies for different sequence lengths.

## Advanced Implementation
- **Q:** How would you implement request prioritization with plugins?
  **A:** Create a scheduler plugin that reorders requests before budget allocation, potentially reserving token budgets for high-priority requests.
- **Q:** What's the role of compression plugins in memory-constrained environments?
  **A:** They can reduce KV cache memory footprint through techniques like quantization, pruning, or adaptive block allocation.
- **Q:** How would you implement adaptive batch sizing?
  **A:** Monitor system utilization and request patterns, dynamically adjust `max_tokens_per_step` based on current load and performance targets.
