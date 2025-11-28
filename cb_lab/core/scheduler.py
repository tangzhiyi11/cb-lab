"""A tiny continuous batching scheduler."""

from typing import List, Sequence, Tuple

import torch

from cb_lab.core.batch_builder import build_ragged_batch
from cb_lab.core.request import Request
from cb_lab.model.tiny_llm import TinyLLM


class Scheduler:
    def __init__(
        self,
        model: TinyLLM,
        max_tokens_per_step: int = 8,
        prefill_chunk_size: int = 4,
    ) -> None:
        if max_tokens_per_step <= 0:
            raise ValueError("max_tokens_per_step must be positive")
        self.model = model
        self.max_tokens_per_step = max_tokens_per_step
        self.prefill_chunk_size = prefill_chunk_size
        self.active: List[Request] = []
        self.step_id = 0

    def add_request(self, req: Request) -> None:
        self.active.append(req)

    def _pick_prefill_chunks(self, budget: int) -> Tuple[int, Sequence]:
        picked = []
        spent = 0
        for req in self.active:
            if not req.in_prefill or spent >= budget:
                continue
            allowed = min(self.prefill_chunk_size, budget - spent)
            if allowed <= 0:
                break
            chunk = req.get_prefill_chunk(allowed)
            if chunk.numel() == 0:
                continue
            picked.append((req.req_id, req.prefill_pos - chunk.size(0), chunk, req))
            spent += chunk.size(0)
            if spent >= budget:
                break
        return spent, picked

    def step(self) -> None:
        if not self.active:
            print("[scheduler] no active requests")
            return

        decode_reqs = [r for r in self.active if r.in_decode]
        decode_cap = min(len(decode_reqs), self.max_tokens_per_step)
        decode_selected = decode_reqs[:decode_cap]
        decode_tokens = len(decode_selected)

        budget = max(self.max_tokens_per_step - decode_tokens, 0)
        prefill_spent, picked_chunks = self._pick_prefill_chunks(budget)

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

        # Prefill path
        if picked_chunks:
            tokens_cat, table, ragged_mask = build_ragged_batch(
                [(rid, start, chunk) for rid, start, chunk, _ in picked_chunks]
            )
            print(
                f"  ragged tokens={tokens_cat.shape}, mask={ragged_mask.shape}, entries={len(table)}"
            )
            out, k_new, v_new = self.model.forward_prefill_ragged(
                tokens_cat, ragged_mask
            )
            cursor = 0
            for rid, start, chunk, req in picked_chunks:
                length = chunk.size(0)
                req.append_kv(
                    k_new[cursor : cursor + length],
                    v_new[cursor : cursor + length],
                )
                cursor += length

        # Decode path
        for req in decode_selected:
            new_token = req.decode_seed
            out, k_new, v_new = self.model.forward_decode(new_token, req.kv_cache)
            req.append_kv(k_new, v_new)
            req.append_token(out.squeeze(0))
            req.decode_seed = out.squeeze(0).detach()
            req.maybe_finish()

        # Drop finished
        self.active = [r for r in self.active if not r.finished]
        self.step_id += 1
