# Continuous Batching Learning Plan

Goal: guide you through cb-lab to deeply understand continuous batching, from basics to paged attention. Follow the steps; each module is small enough to read and run.

## Phase 0: Setup
- Install and run tests: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pytest tests`
- Skim `docs/get_started.md` and `docs/continuous_batching.md` for terminology.

## Phase 1: Prefill basics
- **Concept:** Prefill builds KV for the prompt; no generation yet.
- **Read:** `core/request.py` (prefill_pos, get_prefill_chunk), `model/tiny_llm.py` (`forward_prefill_dense`).
- **Run:** `python -m demos.demo_prefill_only`
- **Observe:** KV cache length equals prompt length; attention is dense causal.

## Phase 2: Ragged batching
- **Concept:** No padding; concatenate chunks; mask isolates sequences.
- **Read:** `core/batch_builder.py`, `attention/ragged_attention.py`
- **Run:** `python -m demos.demo_ragged_mask`
- **Observe:** Token table (`req_id`, `pos`), ragged causal mask only allows same-sequence causal visibility.

## Phase 3: Mixing prefill + decode
- **Concept:** Each step mixes decode tokens (1 per active decode request) with prefill chunks under a token budget.
- **Read:** `core/scheduler.py` (token budget, decode-first), `core/request.py` (decode_seed, append_token).
- **Run:** `python -m demos.demo_prefill_decode_mix`
- **Observe:** Logs of `decode=...`, `prefill=...`, req IDs; watch requests finish when `max_new_tokens` reached.

## Phase 4: Scheduler timeline
- **Concept:** Continuous batching keeps active requests flowing until completion.
- **Run:** `python -m demos.demo_scheduler_timeline`
- **Observe:** How long requests stay active; how budgets are consumed per step.
- **Try:** Change `max_tokens_per_step` / `prefill_chunk_size` to see utilization vs. latency trade-offs.

## Phase 5: KV cache layouts
- **Concept:** Dense vs. paged storage; both append-only.
- **Read:** `core/kv_cache.py`
- **Run:** `python -m demos.demo_paged_attention`
- **Observe:** `len(cache)` after prefill/decode; tweak `block_size` to see block allocation.

## Phase 6: Attention paths
- **Concept:** Prefill uses ragged/dense causal attention; decode attends over cached KV.
- **Read:** `attention/dense_attention.py`, `attention/ragged_attention.py`, `attention/paged_attention.py`
- **Exercise:** Compare outputs with/without ragged masking for a single sequence (they should match dense).

## Phase 7: Tests as examples
- **Run:** `pytest tests`
- **Read:** `tests/test_*` to see expected behaviors (mask causality, scheduler completion, KV append).

## Phase 8: Open-ended explorations
- Add more requests with varying prompt lengths; watch mask and KV growth.
- Simulate higher load by reducing `max_tokens_per_step` and inspecting throughput.
- Add simple “sampling” (e.g., tanh) on decode outputs to see different token trajectories.

## Learning materials (per phase)
- 0: `docs/learning_materials/phase0_setup.md`
- 1: `docs/learning_materials/phase1_prefill.md`
- 2: `docs/learning_materials/phase2_ragged.md`
- 3: `docs/learning_materials/phase3_mix.md`
- 4: `docs/learning_materials/phase4_timeline.md`
- 5: `docs/learning_materials/phase5_kv.md`
- 6: `docs/learning_materials/phase6_attention.md`
- 7: `docs/learning_materials/phase7_tests.md`
- 8: `docs/learning_materials/phase8_exploration.md`
