# Phase 4: Scheduler Timeline

Focus: how requests stay active over multiple steps.

- **Run:**  
  ```bash
  python -m demos.demo_scheduler_timeline
  ```
- **Observe:** Active set per step, decode/prefill counts, and when requests drop.
- **Exercise:** Increase `max_new_tokens` or add more requests; tweak `max_tokens_per_step` to see latency/throughput shifts.
