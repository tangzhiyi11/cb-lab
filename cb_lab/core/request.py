from dataclasses import dataclass, field
from typing import List, Optional

import torch

from cb_lab.core.kv_cache import BaseKVCache


@dataclass
class Request:
    """A single inference request that flows through prefill -> decode."""

    req_id: str
    prompt: torch.Tensor
    max_new_tokens: int
    kv_cache: BaseKVCache
    prefill_pos: int = 0
    generated_tokens: List[torch.Tensor] = field(default_factory=list)
    finished: bool = False
    decode_seed: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.prompt.dim() != 2:
            raise ValueError("prompt must be a 2D tensor [T, d]")
        if self.max_new_tokens == 0:
            self.finished = True
        if self.decode_seed is None:
            # Use the last prompt token as the first decode seed.
            self.decode_seed = self.prompt[-1].detach().clone()

    @property
    def in_prefill(self) -> bool:
        return self.prefill_pos < self.prompt.size(0)

    @property
    def in_decode(self) -> bool:
        return not self.in_prefill and not self.finished

    def get_prefill_chunk(self, chunk_size: int) -> torch.Tensor:
        """Return a slice of the prompt to prefill."""
        if not self.in_prefill:
            return torch.empty(0, self.prompt.size(1), device=self.prompt.device)
        end = min(self.prefill_pos + chunk_size, self.prompt.size(0))
        chunk = self.prompt[self.prefill_pos : end]
        self.prefill_pos = end
        if self.prefill_pos >= self.prompt.size(0):
            # Switch to decode once prefill is consumed.
            self.prefill_pos = self.prompt.size(0)
        return chunk

    def append_kv(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append freshly computed keys/values into the KV cache."""
        self.kv_cache.append(k_new, v_new)

    def append_token(self, tok: torch.Tensor) -> None:
        self.generated_tokens.append(tok)
        if len(self.generated_tokens) >= self.max_new_tokens:
            self.finished = True

    def maybe_finish(self) -> None:
        if len(self.generated_tokens) >= self.max_new_tokens:
            self.finished = True
