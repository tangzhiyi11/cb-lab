"""A tiny continuous batching scheduler."""

from typing import List, Sequence, Tuple, Optional, Dict, Any
import time

import torch

from cb_lab.core.batch_builder import build_ragged_batch
from cb_lab.core.request import Request
from cb_lab.model.tiny_llm import TinyLLM
from cb_lab.exceptions import SchedulerError, ValidationError, ConfigurationError


class Scheduler:
    """Continuous batching scheduler that mixes prefill and decode operations.

    This scheduler manages a token budget and intelligently mixes decode tokens
    with prefill chunks to maximize GPU utilization while maintaining request
    prioritization and fairness.
    """

    def __init__(
        self,
        model: TinyLLM,
        max_tokens_per_step: int = 8,
        prefill_chunk_size: int = 4,
    ) -> None:
        """Initialize continuous batching scheduler.

        Args:
            model: The language model to use for inference
            max_tokens_per_step: Maximum tokens to process in each step
            prefill_chunk_size: Maximum chunk size for prefill operations

        Raises:
            ConfigurationError: If scheduler configuration is invalid
        """
        self._validate_config(max_tokens_per_step, prefill_chunk_size)

        self.model: TinyLLM = model
        self.max_tokens_per_step: int = max_tokens_per_step
        self.prefill_chunk_size: int = prefill_chunk_size
        self.active: List[Request] = []
        self.step_id: int = 0

    def _validate_config(
        self, max_tokens_per_step: int, prefill_chunk_size: int
    ) -> None:
        """Validate scheduler configuration parameters."""
        if not isinstance(max_tokens_per_step, int) or max_tokens_per_step <= 0:
            raise ConfigurationError(
                f"max_tokens_per_step must be positive int, got {max_tokens_per_step}"
            )

        if not isinstance(prefill_chunk_size, int) or prefill_chunk_size <= 0:
            raise ConfigurationError(
                f"prefill_chunk_size must be positive int, got {prefill_chunk_size}"
            )

        if prefill_chunk_size > max_tokens_per_step:
            raise ConfigurationError(
                f"prefill_chunk_size ({prefill_chunk_size}) cannot exceed "
                f"max_tokens_per_step ({max_tokens_per_step})"
            )

    def add_request(self, req: Request) -> None:
        """Add a new request to the scheduler.

        Args:
            req: Request to add to the active pool

        Raises:
            SchedulerError: If request cannot be added
        """
        try:
            if not isinstance(req, Request):
                raise ValidationError(f"req must be Request instance, got {type(req)}")

            # Check for duplicate request IDs
            existing_ids = {r.req_id for r in self.active}
            if req.req_id in existing_ids:
                raise SchedulerError(
                    f"Request ID '{req.req_id}' already exists in active requests"
                )

            self.active.append(req)

        except Exception as e:
            if isinstance(e, (ValidationError, SchedulerError)):
                raise
            raise SchedulerError(f"Unexpected error adding request: {e}") from e

    def _pick_prefill_chunks(
        self, budget: int
    ) -> Tuple[int, List[Tuple[str, int, torch.Tensor, Request]]]:
        """Select prefill chunks from active requests within budget.

        Args:
            budget: Maximum number of tokens to allocate for prefill

        Returns:
            Tuple of (tokens_spent, List of (req_id, start_pos, chunk, request))

        Raises:
            ValidationError: If budget is invalid
        """
        if not isinstance(budget, int) or budget < 0:
            raise ValidationError(f"budget must be non-negative int, got {budget}")

        picked: List[Tuple[str, int, torch.Tensor, Request]] = []
        spent = 0

        for req in self.active:
            if not req.in_prefill or spent >= budget:
                continue

            allowed = min(self.prefill_chunk_size, budget - spent)
            if allowed <= 0:
                break

            try:
                chunk = req.get_prefill_chunk(allowed)
                if chunk.numel() == 0:
                    continue

                # Store request metadata
                start_pos = req.prefill_pos - chunk.size(0)
                picked.append((req.req_id, start_pos, chunk, req))
                spent += chunk.size(0)

                if spent >= budget:
                    break

            except Exception as e:
                # Log error but continue with other requests
                print(
                    f"[scheduler] Warning: Failed to get prefill chunk for {req.req_id}: {e}"
                )
                continue

        return spent, picked

    def step(self) -> Dict[str, Any]:
        """Execute one step of continuous batching.

        This method orchestrates the mixing of prefill and decode operations
        within the token budget constraints.

        Returns:
            Dictionary with step statistics and metrics

        Raises:
            SchedulerError: If step execution fails
        """
        step_start_time = time.time()

        try:
            if not self.active:
                print("[scheduler] no active requests")
                return {
                    "step_id": self.step_id,
                    "decode_tokens": 0,
                    "prefill_tokens": 0,
                    "active_requests": 0,
                    "finished_requests": 0,
                    "step_duration": 0.0,
                    "status": "no_active_requests",
                }

            # Select decode requests
            decode_reqs = [r for r in self.active if r.in_decode]
            decode_cap = min(len(decode_reqs), self.max_tokens_per_step)
            decode_selected = decode_reqs[:decode_cap]
            decode_tokens = len(decode_selected)

            # Allocate remaining budget for prefill
            budget = max(self.max_tokens_per_step - decode_tokens, 0)
            prefill_spent, picked_chunks = self._pick_prefill_chunks(budget)

            # Log step information
            self._log_step_info(
                decode_selected, picked_chunks, prefill_spent, decode_tokens
            )

            # Process prefill chunks
            if picked_chunks:
                self._process_prefill_chunks(picked_chunks)

            # Process decode requests
            processed_decode = self._process_decode_requests(decode_selected)

            # Remove finished requests
            finished_count = len([r for r in self.active if r.finished])
            self.active = [r for r in self.active if not r.finished]

            step_duration = time.time() - step_start_time

            step_stats = {
                "step_id": self.step_id,
                "decode_tokens": decode_tokens,
                "prefill_tokens": prefill_spent,
                "active_requests": len(self.active),
                "finished_requests": finished_count,
                "step_duration": step_duration,
                "status": "completed",
            }

            self.step_id += 1
            return step_stats

        except Exception as e:
            error_msg = f"Step {self.step_id} failed: {e}"
            print(f"[scheduler] Error: {error_msg}")
            self.step_id += 1
            raise SchedulerError(error_msg) from e

    def _log_step_info(
        self,
        decode_selected: List[Request],
        picked_chunks: List[Tuple[str, int, torch.Tensor, Request]],
        prefill_spent: int,
        decode_tokens: int,
    ) -> None:
        """Log detailed step information for debugging."""
        print(
            f"\n[step {self.step_id}] decode={decode_tokens}, prefill={prefill_spent}, active={len(self.active)}"
        )
        if decode_selected:
            print("  decode reqs:", [r.req_id for r in decode_selected])
        if picked_chunks:
            print(
                "  prefill reqs:",
                [(rid, chunk.shape[0]) for rid, _, chunk, _ in picked_chunks],
            )

    def _process_prefill_chunks(
        self, picked_chunks: List[Tuple[str, int, torch.Tensor, Request]]
    ) -> None:
        """Process prefill chunks using ragged batching."""
        try:
            tokens_cat, table, ragged_mask = build_ragged_batch(
                [(rid, start, chunk) for rid, start, chunk, _ in picked_chunks]
            )
            print(
                f"  ragged tokens={tokens_cat.shape}, mask={ragged_mask.shape}, entries={len(table)}"
            )

            # Forward pass through model
            out, k_new, v_new = self.model.forward_prefill_ragged(
                tokens_cat, ragged_mask
            )

            # Distribute KV pairs back to requests
            cursor = 0
            for rid, start, chunk, req in picked_chunks:
                length = chunk.size(0)
                req.append_kv(
                    k_new[cursor : cursor + length],
                    v_new[cursor : cursor + length],
                )
                cursor += length

        except Exception as e:
            raise SchedulerError(f"Failed to process prefill chunks: {e}") from e

    def _process_decode_requests(self, decode_selected: List[Request]) -> int:
        """Process individual decode requests."""
        processed = 0
        for req in decode_selected:
            try:
                if req.decode_seed is None:
                    print(
                        f"[scheduler] Warning: No decode seed for request {req.req_id}"
                    )
                    continue

                new_token = req.decode_seed
                out, k_new, v_new = self.model.forward_decode(new_token, req.kv_cache)

                req.append_kv(k_new, v_new)
                req.append_token(out.squeeze(0))
                req.decode_seed = out.squeeze(0).detach()
                req.maybe_finish()
                processed += 1

            except Exception as e:
                print(f"[scheduler] Error processing decode for {req.req_id}: {e}")
                # Continue with other requests even if one fails

        return processed
