"""Simple KV cache implementations for dense and paged layouts."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import torch

from cb_lab.exceptions import KVCacheError, ValidationError


class BaseKVCache(ABC):
    """Abstract base class for KV cache implementations."""

    @abstractmethod
    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append new keys and values to the cache.

        Args:
            k_new: New keys tensor of shape [seq_len, dim]
            v_new: New values tensor of shape [seq_len, dim]

        Raises:
            KVCacheError: If append operation fails
        """
        ...

    @abstractmethod
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve all cached keys and values.

        Returns:
            Tuple of (keys, values) tensors, each of shape [cached_len, dim]
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of cached sequences."""
        ...


class DenseKVCache(BaseKVCache):
    """Append-only dense KV cache used for prefill and decode.

    This implementation stores keys and values in contiguous tensors that grow
    by concatenation. Simple and easy to understand, making it ideal for educational
    purposes.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Initialize dense KV cache.

        Args:
            device: Device to store tensors on. Defaults to CPU if not specified.
        """
        self.device = device or torch.device("cpu")
        if not isinstance(self.device, torch.device):
            raise ValidationError(
                f"device must be torch.device, got {type(self.device)}"
            )

        # Initialize empty tensors with proper dimension
        self.k: torch.Tensor = torch.empty(
            0, 0, device=self.device, dtype=torch.float32
        )
        self.v: torch.Tensor = torch.empty(
            0, 0, device=self.device, dtype=torch.float32
        )

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append new keys and values to the dense cache.

        Args:
            k_new: New keys tensor of shape [seq_len, dim]
            v_new: New values tensor of shape [seq_len, dim]

        Raises:
            KVCacheError: If append operation fails due to shape/device mismatches
        """
        try:
            self._validate_append_inputs(k_new, v_new)

            if self.k.numel() == 0:
                # First append - initialize with input tensor shapes
                self.k = k_new.detach().clone()
                self.v = v_new.detach().clone()
            else:
                # Validate dimension compatibility
                if k_new.size(1) != self.k.size(1):
                    raise KVCacheError(
                        f"Feature dimension mismatch: cache has dim {self.k.size(1)}, "
                        f"new k has dim {k_new.size(1)}"
                    )

                if v_new.size(1) != self.v.size(1):
                    raise KVCacheError(
                        f"Feature dimension mismatch: cache has dim {self.v.size(1)}, "
                        f"new v has dim {v_new.size(1)}"
                    )

                # Concatenate new tensors
                self.k = torch.cat([self.k, k_new.detach()], dim=0)
                self.v = torch.cat([self.v, v_new.detach()], dim=0)

        except Exception as e:
            if isinstance(e, (KVCacheError, ValidationError)):
                raise
            raise KVCacheError(f"Unexpected error during append: {e}") from e

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve all cached keys and values.

        Returns:
            Tuple of (keys, values) tensors, each of shape [cached_len, dim]
        """
        return self.k.clone(), self.v.clone()

    def __len__(self) -> int:
        """Return the number of cached token positions."""
        return self.k.size(0)

    def _validate_append_inputs(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Validate inputs for append operation."""
        if not isinstance(k_new, torch.Tensor) or not isinstance(v_new, torch.Tensor):
            raise ValidationError("k_new and v_new must be torch.Tensor")

        if k_new.shape != v_new.shape:
            raise ValidationError(
                f"k_new and v_new must have same shape, got {k_new.shape} vs {v_new.shape}"
            )

        if k_new.dim() != 2:
            raise ValidationError(
                f"k_new must be 2D tensor [seq_len, dim], got {k_new.dim()}D"
            )

        if k_new.device != self.device or v_new.device != self.device:
            raise ValidationError(
                f"Tensors must be on cache device {self.device}, "
                f"got k_new: {k_new.device}, v_new: {v_new.device}"
            )


class PagedAllocator:
    """A tiny block allocator for paged attention.

    This allocator manages fixed-size blocks for KV storage, similar to how
    production systems handle paged attention. It provides more efficient memory
    usage compared to dense concatenation.
    """

    def __init__(self, block_size: int = 64) -> None:
        """Initialize paged allocator.

        Args:
            block_size: Number of token positions per block

        Raises:
            ValidationError: If block_size is invalid
        """
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValidationError(f"block_size must be positive int, got {block_size}")

        self.block_size: int = block_size
        self.blocks: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.next_free_offset: int = 0
        self.device: torch.device = torch.device("cpu")
        self.dim: int = 0

    def allocate_block(self, dim: int, device: torch.device) -> None:
        """Allocate a new block for KV storage.

        Args:
            dim: Feature dimension for tensors
            device: Device to store tensors on

        Raises:
            ValidationError: If parameters are invalid
        """
        if not isinstance(dim, int) or dim <= 0:
            raise ValidationError(f"dim must be positive int, got {dim}")

        if not isinstance(device, torch.device):
            raise ValidationError(f"device must be torch.device, got {type(device)}")

        self.device = device
        self.dim = dim
        k_block = torch.zeros(self.block_size, dim, device=device, dtype=torch.float32)
        v_block = torch.zeros(self.block_size, dim, device=device, dtype=torch.float32)
        self.blocks.append((k_block, v_block))
        self.next_free_offset = 0

    def store(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Store new KV pairs in the allocated blocks.

        Args:
            k_new: New keys tensor of shape [seq_len, dim]
            v_new: New values tensor of shape [seq_len, dim]

        Raises:
            KVCacheError: If storage operation fails
        """
        try:
            self._validate_store_inputs(k_new, v_new)

            # Initialize first block if needed
            if not self.blocks:
                self.allocate_block(k_new.size(1), k_new.device)

            # Store tokens one by one across blocks
            for idx in range(k_new.size(0)):
                if self.next_free_offset >= self.block_size:
                    # Allocate new block when current is full
                    self.allocate_block(k_new.size(1), k_new.device)

                k_block, v_block = self.blocks[-1]
                k_block[self.next_free_offset] = k_new[idx]
                v_block[self.next_free_offset] = v_new[idx]
                self.next_free_offset += 1

        except Exception as e:
            if isinstance(e, (KVCacheError, ValidationError)):
                raise
            raise KVCacheError(f"Unexpected error during store: {e}") from e

    def flatten(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten all blocks into contiguous tensors.

        Returns:
            Tuple of (keys, values) tensors, each of shape [total_stored, dim]
        """
        if not self.blocks:
            return (
                torch.empty(0, self.dim or 0, device=self.device, dtype=torch.float32),
                torch.empty(0, self.dim or 0, device=self.device, dtype=torch.float32),
            )

        # Concatenate all blocks
        k_all = torch.cat([blk[0] for blk in self.blocks], dim=0)
        v_all = torch.cat([blk[1] for blk in self.blocks], dim=0)

        # Trim unused tail in the last block
        used = (len(self.blocks) - 1) * self.block_size + self.next_free_offset
        return k_all[:used], v_all[:used]

    def __len__(self) -> int:
        """Return total number of stored token positions."""
        if not self.blocks:
            return 0
        return (len(self.blocks) - 1) * self.block_size + self.next_free_offset

    def _validate_store_inputs(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Validate inputs for store operation."""
        if not isinstance(k_new, torch.Tensor) or not isinstance(v_new, torch.Tensor):
            raise ValidationError("k_new and v_new must be torch.Tensor")

        if k_new.shape != v_new.shape:
            raise ValidationError(
                f"k_new and v_new must have same shape, got {k_new.shape} vs {v_new.shape}"
            )

        if k_new.dim() != 2:
            raise ValidationError(
                f"k_new must be 2D tensor [seq_len, dim], got {k_new.dim()}D"
            )

        # Check device compatibility
        if self.blocks and k_new.device != self.device:
            raise ValidationError(
                f"New tensors device {k_new.device} must match allocator device {self.device}"
            )


class PagedKVCache(BaseKVCache):
    """KV cache backed by paged blocks.

    This implementation uses a paged allocator to store KV pairs in fixed-size blocks,
    which provides better memory efficiency and allocation patterns compared to dense
    concatenation, especially for variable-length sequences.
    """

    def __init__(
        self, block_size: int = 64, device: Optional[torch.device] = None
    ) -> None:
        """Initialize paged KV cache.

        Args:
            block_size: Number of token positions per block
            device: Device to store tensors on. Defaults to CPU if not specified.

        Raises:
            ValidationError: If block_size is invalid
        """
        self.device = device or torch.device("cpu")
        self.allocator: PagedAllocator = PagedAllocator(block_size)
        self.dim: int = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append new keys and values to the paged cache.

        Args:
            k_new: New keys tensor of shape [seq_len, dim]
            v_new: New values tensor of shape [seq_len, dim]

        Raises:
            KVCacheError: If append operation fails
        """
        try:
            # Set dimension on first append
            if self.dim == 0:
                self.dim = k_new.size(1)
            elif k_new.size(1) != self.dim:
                raise KVCacheError(
                    f"Feature dimension mismatch: cache has dim {self.dim}, "
                    f"new k has dim {k_new.size(1)}"
                )

            self.allocator.store(k_new.detach(), v_new.detach())

        except Exception as e:
            if isinstance(e, (KVCacheError, ValidationError)):
                raise
            raise KVCacheError(f"Unexpected error during append: {e}") from e

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve all cached keys and values as flattened tensors.

        Returns:
            Tuple of (keys, values) tensors, each of shape [cached_len, dim]
        """
        return self.allocator.flatten()

    def __len__(self) -> int:
        """Return the number of cached token positions."""
        return len(self.allocator)
