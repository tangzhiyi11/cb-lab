# Continuous Batching in cb-lab

This note expands the intuition behind continuous batching (see also the Hugging Face post on continuous batching) and points to where each idea lives in cb-lab.

## Why continuous batching matters
- **Latency vs. throughput:** Serving many prompts individually leaves GPU underutilized; large static batches improve throughput but add queuing latency. Continuous batching keeps the GPU busy while letting requests enter/exit at token boundaries.
- **Prefill vs. decode asymmetry:** Prefill is long and parallelizable; decode is short and sequential (one token per step). Mixing them is key to utilization.
- **KV cache reuse:** Avoid recomputing history—store K/V and only attend to cached tokens during decode.

## Core mechanics in cb-lab
- **Token budget per step (`max_tokens_per_step`):** Each scheduler step allocates capacity. Decode tokens are placed first (one per in-decode request, capped by the budget), then prefill chunks fill the leftover.
- **Prefill chunking (`prefill_chunk_size`):** Long prompts are split into smaller chunks to fit the budget and reduce latency before first decode.
- **Ragged batching (no padding):** Prefill chunks from different requests are concatenated into one tensor. A ragged causal mask isolates sequences while preserving per-sequence causality—no wasted computation on padding.
- **KV cache growth:** After every prefill and decode, new K/V pairs are appended. Decode attends only to cached history (dense or paged layout).
- **Paged layout (simplified):** KV is stored in fixed-size blocks. This models paged attention layouts that reduce memory movement on decode.

## Control knobs to explore
- `max_tokens_per_step`: Higher values favor throughput; lower values reduce per-step latency. Observe utilization by changing this and watching scheduler logs.
- `prefill_chunk_size`: Smaller chunks reduce time-to-first-token but increase overhead; larger chunks improve prefill efficiency. Combine with the token budget to see different mixes.
- Cache backend: Swap `DenseKVCache` vs. `PagedKVCache` to compare data layouts and flattened lengths.

## Where to look in the code
- Prefill chunking and decode seed: `core/request.py`
- Scheduler token budgeting and mixing: `core/scheduler.py`
- Ragged mask building: `core/batch_builder.py`
- KV cache backends: `core/kv_cache.py`
- Attention math: `attention/*.py`
- Model interface: `model/tiny_llm.py`

## Hands-on recipes
- **Budget stress test:** Change `max_tokens_per_step` and `prefill_chunk_size` in `core/scheduler.py`; run `python -m demos.demo_prefill_decode_mix` to see how decode/prefill counts shift per step.
- **Ragged growth:** Extend prompt lengths in `demo_ragged_mask.py` to see the mask grow without padding and verify isolation between requests.
- **Paged vs. dense KV:** In `demo_paged_attention.py`, toggle between `DenseKVCache` and `PagedKVCache` and compare `len(cache)` after prefill/decode.
- **Longer active set:** Increase `max_new_tokens` or add more requests in `demo_scheduler_timeline.py` to watch how long items stay active and how budgets are consumed.
