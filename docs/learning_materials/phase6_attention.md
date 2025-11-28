# Phase 6: Attention Paths

Focus: how prefill and decode use different attention helpers.

- **Read:**  
  - `attention/dense_attention.py` (single-sequence causal)  
  - `attention/ragged_attention.py` (mask-driven)  
  - `attention/paged_attention.py` (decode over cache)
- **Exercise:** For a single sequence, compare outputs from dense vs. ragged (mask constructed for one seq) to confirm equivalence. Add prints for Q/K/V and attention scores to see masking effects.
- **Connect:** `model/tiny_llm.py` wires these paths via `forward_prefill_ragged` and `forward_decode`.
