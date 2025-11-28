# Principles → cb-lab Recipes

For each key principle, here is how to observe or validate it using cb-lab.

## Prefill vs. Decode mixing
- **What to see:** Decode tokens prioritized, prefill fills remaining budget.
- **How:** Run `python -m demos.demo_prefill_decode_mix` and watch the per-step log (`decode=..., prefill=...`). Tweak `max_tokens_per_step` / `prefill_chunk_size` in `core/scheduler.py`.

## Ragged batching without padding
- **What to see:** Token table and ragged causal mask isolating sequences.
- **How:** Run `python -m demos.demo_ragged_mask`. Extend chunks or add new requests to watch the mask grow. Inspect `core/batch_builder.py` for mask construction.

## KV cache growth (dense)
- **What to see:** Append-only K/V accumulation after prefill and decode.
- **How:** Inspect `demo_prefill_only.py` (prefill) and `demo_prefill_decode_mix.py` (mixed). Check `len(cache)` before/after steps. Code lives in `core/kv_cache.py`.

## Paged KV layout
- **What to see:** Blocks allocated as tokens are appended; flattened length tracks used slots.
- **How:** Run `python -m demos.demo_paged_attention`. Change `block_size` in `PagedKVCache` and observe lengths after prefill and decode.

## Scheduler completion
- **What to see:** Requests leave the active set once `max_new_tokens` is reached.
- **How:** Run `demo_scheduler_timeline.py` and watch the final summary. Tests `tests/test_scheduler.py` assert the scheduler drains all requests.

## Attention correctness (sanity)
- **What to see:** Outputs have expected shapes; masks enforce causality.
- **How:** Run `pytest tests/test_mask.py` and `test_ragged_batching.py` for mask logic; `test_paged_attention.py` for decode over paged cache; `test_kv_cache.py` for append semantics.
