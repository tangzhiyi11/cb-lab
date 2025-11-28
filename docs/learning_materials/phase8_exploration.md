# Phase 8: Open Explorations

Focus: extend scenarios to stress continuous batching behaviors.

- **More requests:** Add longer prompts or more concurrent requests in demos; watch ragged mask and KV growth.
- **Budget stress:** Lower `max_tokens_per_step` to simulate load; observe decode queuing and utilization.
- **Prefill chunk tuning:** Vary `prefill_chunk_size` to balance time-to-first-token vs. efficiency.
- **Sampling tweaks:** Add simple nonlinearities to decode outputs (e.g., `torch.tanh`) to simulate token variation.
- **Metrics:** Instrument prints for per-step utilization (decode vs. prefill counts) and cache lengths.
