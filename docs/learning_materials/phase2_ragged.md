# Phase 2: Ragged Batching

Focus: batching without padding via concatenation + ragged causal mask.

- **Read:**  
  - `core/batch_builder.py` (token table, mask rules)  
  - `attention/ragged_attention.py`
- **Run:**  
  ```bash
  python -m demos.demo_ragged_mask
  ```
- **Observe:** Token table fields (`req_id`, `pos_in_seq`, `global_idx`); mask allows only same-sequence causal visibility.
- **Exercise:** Add more chunks/requests to the demo; verify cross-request mask entries stay False.
