# Phase 7: Tests as Guides

Focus: use tests to cement expected behaviors.

- **Run:**  
  ```bash
  pytest tests
  ```
- **Read:**  
  - `tests/test_mask.py` / `test_ragged_batching.py`: mask causality and token table.
  - `tests/test_scheduler.py`: scheduler completion and decode budget cap.
  - `tests/test_kv_cache.py`: append semantics for dense/paged.
  - `tests/test_paged_attention.py`: decode over paged KV.
- **Exercise:** Break a test intentionally (e.g., change mask condition) to see failures and understand invariants.
