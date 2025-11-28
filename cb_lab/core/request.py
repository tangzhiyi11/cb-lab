"""Request lifecycle management for continuous batching."""

from dataclasses import dataclass, field
from typing import List, Optional
import warnings

import torch

from cb_lab.core.kv_cache import BaseKVCache
from cb_lab.exceptions import (
    ValidationError,
    RequestError,
    ConfigurationError,
    KVCacheError,
)


@dataclass
class Request:
    """A single inference request that flows through prefill -> decode.

    Attributes:
        req_id: Unique identifier for the request
        prompt: Input prompt tensor of shape [seq_len, dim]
        max_new_tokens: Maximum number of tokens to generate
        kv_cache: KV cache for this request
        prefill_pos: Current position in prompt processing
        generated_tokens: List of generated tokens
        finished: Whether request has completed
        decode_seed: Initial token for decode phase
    """

    req_id: str
    prompt: torch.Tensor
    max_new_tokens: int
    kv_cache: BaseKVCache
    prefill_pos: int = 0
    generated_tokens: List[torch.Tensor] = field(default_factory=list)
    finished: bool = False
    decode_seed: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        """Validate request state after initialization."""
        try:
            self._validate_prompt()
            self._validate_max_new_tokens()
            self._validate_kv_cache()
            self._initialize_decode_seed()
        except (ValidationError, ConfigurationError):
            # Re-raise validation errors directly
            raise
        except Exception as e:
            raise RequestError(f"Invalid request configuration: {e}") from e

    def _validate_prompt(self) -> None:
        """Validate prompt tensor properties."""
        if not isinstance(self.prompt, torch.Tensor):
            raise ValidationError(
                f"prompt must be torch.Tensor, got {type(self.prompt)}"
            )

        if self.prompt.dim() != 2:
            raise ValidationError(
                f"prompt must be 2D tensor [seq_len, dim], got {self.prompt.dim()}D tensor "
                f"with shape {self.prompt.shape}"
            )

        if self.prompt.size(0) == 0:
            raise ValidationError("prompt sequence length cannot be zero")

        if self.prompt.size(1) == 0:
            raise ValidationError("prompt feature dimension cannot be zero")

    def _validate_max_new_tokens(self) -> None:
        """Validate max_new_tokens parameter."""
        if not isinstance(self.max_new_tokens, int):
            raise ValidationError(
                f"max_new_tokens must be int, got {type(self.max_new_tokens)}"
            )

        if self.max_new_tokens < 0:
            raise ValidationError("max_new_tokens must be non-negative")

        if self.max_new_tokens > 10000:  # Reasonable upper bound
            warnings.warn(
                f"Large max_new_tokens ({self.max_new_tokens}) may cause memory issues",
                UserWarning,
            )

    def _validate_kv_cache(self) -> None:
        """Validate KV cache compatibility."""
        if not isinstance(self.kv_cache, BaseKVCache):
            raise ValidationError(
                f"kv_cache must inherit from BaseKVCache, got {type(self.kv_cache)}"
            )

    def _initialize_decode_seed(self) -> None:
        """Initialize decode seed and handle max_new_tokens=0 case."""
        if self.max_new_tokens == 0:
            self.finished = True

        if self.decode_seed is None:
            if self.prompt.size(0) == 0:
                raise ValidationError("Cannot create decode_seed from empty prompt")
            # Use the last prompt token as the first decode seed.
            self.decode_seed = self.prompt[-1].detach().clone()

    @property
    def in_prefill(self) -> bool:
        """Check if request is still in prefill phase."""
        return self.prefill_pos < self.prompt.size(0)

    @property
    def in_decode(self) -> bool:
        """Check if request is in decode phase."""
        return not self.in_prefill and not self.finished

    def get_prefill_chunk(self, chunk_size: int) -> torch.Tensor:
        """Return a slice of the prompt to prefill.

        Args:
            chunk_size: Maximum number of tokens to return in this chunk

        Returns:
            Tensor of shape [chunk_len, dim] where chunk_len <= chunk_size

        Raises:
            ValidationError: If chunk_size is invalid
        """
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValidationError(f"chunk_size must be positive int, got {chunk_size}")

        if not self.in_prefill:
            return torch.empty(0, self.prompt.size(1), device=self.prompt.device)

        end = min(self.prefill_pos + chunk_size, self.prompt.size(0))
        chunk = self.prompt[self.prefill_pos : end]
        self.prefill_pos = end

        # Ensure we don't go beyond prompt bounds
        if self.prefill_pos >= self.prompt.size(0):
            self.prefill_pos = self.prompt.size(0)

        return chunk

    def append_kv(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append freshly computed keys/values into the KV cache.

        Args:
            k_new: New keys tensor of shape [seq_len, dim]
            v_new: New values tensor of shape [seq_len, dim]

        Raises:
            KVCacheError: If KV cache append operation fails
            ValidationError: If input tensors are invalid
        """
        try:
            self._validate_kv_tensors(k_new, v_new)
            self.kv_cache.append(k_new, v_new)
        except Exception as e:
            raise KVCacheError(f"Failed to append KV cache: {e}") from e

    def append_token(self, tok: torch.Tensor) -> None:
        """Append a generated token to the request.

        Args:
            tok: Generated token tensor of shape [dim] or [1, dim]

        Raises:
            ValidationError: If token tensor is invalid
        """
        try:
            self._validate_token(tok)
            self.generated_tokens.append(tok)
            if len(self.generated_tokens) >= self.max_new_tokens:
                self.finished = True
        except Exception as e:
            raise RequestError(f"Failed to append token: {e}") from e

    def maybe_finish(self) -> None:
        """Check if request should be finished and update status accordingly."""
        if len(self.generated_tokens) >= self.max_new_tokens:
            self.finished = True

    def _validate_kv_tensors(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Validate KV tensor shapes and compatibility."""
        if not isinstance(k_new, torch.Tensor) or not isinstance(v_new, torch.Tensor):
            raise ValidationError("k_new and v_new must be torch.Tensor")

        if k_new.shape != v_new.shape:
            raise ValidationError(
                f"k_new and v_new must have same shape, got {k_new.shape} vs {v_new.shape}"
            )

        if k_new.dim() != 2:
            raise ValidationError(
                f"KV tensors must be 2D, got {k_new.dim()}D for k_new"
            )

        if k_new.device != v_new.device:
            raise ValidationError(
                f"k_new and v_new must be on same device, got {k_new.device} vs {v_new.device}"
            )

        # Check device compatibility with cache
        cache_device = getattr(self.kv_cache, "device", torch.device("cpu"))
        if k_new.device != cache_device:
            raise ValidationError(
                f"KV tensors device {k_new.device} must match cache device {cache_device}"
            )

    def _validate_token(self, tok: torch.Tensor) -> None:
        """Validate generated token tensor."""
        if not isinstance(tok, torch.Tensor):
            raise ValidationError(f"token must be torch.Tensor, got {type(tok)}")

        if tok.dim() == 0:
            # scalar tensor - reshape to [1]
            tok = tok.unsqueeze(0)
        elif tok.dim() == 1:
            # vector tensor of shape [dim] - keep as is
            pass
        elif tok.dim() == 2 and tok.size(0) == 1:
            # batch tensor of shape [1, dim] - squeeze to [dim]
            tok = tok.squeeze(0)
        else:
            raise ValidationError(
                f"token must be scalar, 1D vector, or 1-batch 2D, got shape {tok.shape}"
            )
