# Phase 3: Mixing Prefill + Decode

Focus: token budgeting that prioritizes decode and fills remaining capacity with prefill chunks.

- **Read:**  
  - `core/scheduler.py` (decode-first, budget, chunk picking)  
  - `core/request.py` (decode_seed, append_token)
- **Run:**  
  ```bash
  python -m demos.demo_prefill_decode_mix
  ```
- **Observe:** Logs showing `decode=...`, `prefill=...` with req IDs; requests finish when `max_new_tokens` reached.
- **Exercise:** Change `max_tokens_per_step` or `prefill_chunk_size` to see decode/prefill ratio change; cap budget to force decode queuing.*** End Patch
