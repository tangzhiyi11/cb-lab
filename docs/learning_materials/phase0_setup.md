# Phase 0: Setup

Goal: get the environment ready and skim key docs.

- **Install & test:**  
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
  pip install -r requirements.txt
  pytest tests
  ```
- **Skim terminology:** `docs/get_started.md`, `docs/continuous_batching.md`
- **Artifacts to inspect:** `requirements.txt`, `tests/conftest.py` (ensures imports work).
