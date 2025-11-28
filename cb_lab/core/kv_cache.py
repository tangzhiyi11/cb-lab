"""Simple KV cache implementations for dense and paged layouts."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class BaseKVCache(ABC):
    @abstractmethod
    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        ...

    @abstractmethod
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class DenseKVCache(BaseKVCache):
    """Append-only dense KV cache used for prefill and decode."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.k = torch.empty(0, 0, device=device)
        self.v = torch.empty(0, 0, device=device)

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self.k.numel() == 0:
            self.k = k_new.detach()
            self.v = v_new.detach()
        else:
            self.k = torch.cat([self.k, k_new.detach()], dim=0)
            self.v = torch.cat([self.v, v_new.detach()], dim=0)

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k, self.v

    def __len__(self) -> int:
        return self.k.size(0)


class PagedAllocator:
    """A tiny block allocator for paged attention."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.blocks: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.next_free_offset = 0
        self.device: torch.device = torch.device("cpu")
        self.dim: int = 0

    def allocate_block(self, dim: int, device: torch.device) -> None:
        self.device = device
        self.dim = dim
        k_block = torch.zeros(self.block_size, dim, device=device)
        v_block = torch.zeros(self.block_size, dim, device=device)
        self.blocks.append((k_block, v_block))
        self.next_free_offset = 0

    def store(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if not self.blocks:
            self.allocate_block(k_new.size(1), k_new.device)
        for idx in range(k_new.size(0)):
            if self.next_free_offset >= self.block_size:
                self.allocate_block(k_new.size(1), k_new.device)
            k_block, v_block = self.blocks[-1]
            k_block[self.next_free_offset] = k_new[idx]
            v_block[self.next_free_offset] = v_new[idx]
            self.next_free_offset += 1

    def flatten(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.blocks:
            return (
                torch.empty(0, self.dim or 0, device=self.device),
                torch.empty(0, self.dim or 0, device=self.device),
            )
        k_all = torch.cat([blk[0] for blk in self.blocks], dim=0)
        v_all = torch.cat([blk[1] for blk in self.blocks], dim=0)
        # Trim unused tail in the last block.
        used = (len(self.blocks) - 1) * self.block_size + self.next_free_offset
        return k_all[:used], v_all[:used]

    def __len__(self) -> int:
        return (len(self.blocks) - 1) * self.block_size + self.next_free_offset


class PagedKVCache(BaseKVCache):
    """KV cache backed by paged blocks."""

    def __init__(self, block_size: int, device: torch.device) -> None:
        self.allocator = PagedAllocator(block_size)
        self.device = device
        self.dim: int = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self.dim == 0:
            self.dim = k_new.size(1)
        self.allocator.store(k_new.detach(), v_new.detach())

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.allocator.flatten()

    def __len__(self) -> int:
        return len(self.allocator)
