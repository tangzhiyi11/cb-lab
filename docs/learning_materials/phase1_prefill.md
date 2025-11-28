# Phase 1: Prefill Basics

Focus: how prompt tokens become KV without generating new tokens.

- **Read:**  
  - `core/request.py` (`prefill_pos`, `get_prefill_chunk`)  
  - `model/tiny_llm.py` (`forward_prefill_dense`)
- **Run:**  
  ```bash
  python -m demos.demo_prefill_only
  ```
- **Observe:** KV length equals prompt length; attention uses dense causal mask.
- **Exercise:** Change prompt length and see `len(cache)`; add prints inside `forward_prefill_dense` to trace Q/K/V shapes.
