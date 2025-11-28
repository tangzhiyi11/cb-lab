#!/usr/bin/env python3
"""Visualization demo for cb-lab continuous batching concepts."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch

from cb_lab.core.scheduler import Scheduler
from cb_lab.core.request import Request
from cb_lab.core.batch_builder import build_ragged_batch, TokenMeta
from cb_lab.core.kv_cache import DenseKVCache, PagedKVCache
from cb_lab.model.tiny_llm import TinyLLM


class ContinuousBatchingVisualizer:
    """Visualization tools for continuous batching concepts."""

    def __init__(self, dim: int = 16):
        self.model = TinyLLM(dim)
        self.fig_size = (12, 8)
        self.colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#48C9B0"]

    def visualize_ragged_mask(
        self,
        chunks_data: List[Tuple[str, int, torch.Tensor]],
        save_path: Optional[str] = None,
    ):
        """Visualize ragged causal mask construction."""
        # Build ragged batch
        tokens_cat, token_table, ragged_mask = build_ragged_batch(chunks_data)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "Ragged Batching and Causal Mask Visualization",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: Token sequences
        self._plot_token_sequences(ax1, token_table, chunks_data)

        # Plot 2: Ragged mask
        self._plot_ragged_mask(ax2, ragged_mask, token_table)

        # Plot 3: Attention pattern
        self._plot_attention_pattern(ax3, ragged_mask, token_table)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_token_sequences(
        self,
        ax,
        token_table: List[TokenMeta],
        chunks_data: List[Tuple[str, int, torch.Tensor]],
    ):
        """Plot individual token sequences."""
        ax.set_title("Token Sequences by Request")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Request ID")

        # Group tokens by request
        req_tokens = {}
        for token_meta in token_table:
            if token_meta.req_id not in req_tokens:
                req_tokens[token_meta.req_id] = []
            req_tokens[token_meta.req_id].append(token_meta)

        # Plot each request's tokens
        y_pos = 0
        for req_id, tokens in req_tokens.items():
            x_positions = [t.pos_in_seq for t in tokens]
            y_positions = [y_pos] * len(tokens)

            color = self.colors[len(req_tokens) % len(self.colors)]
            ax.scatter(
                x_positions, y_positions, s=100, c=color, alpha=0.7, label=req_id
            )

            # Connect tokens with lines
            if len(x_positions) > 1:
                ax.plot(x_positions, y_positions, color=color, alpha=0.5, linewidth=2)

            # Add request label
            ax.text(-0.5, y_pos, req_id, ha="right", va="center", fontweight="bold")

            y_pos += 1

        ax.set_xlim(
            -1,
            max([max([t.pos_in_seq for t in tokens]) for tokens in req_tokens.values()])
            + 1,
        )
        ax.set_ylim(-0.5, len(req_tokens) - 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_ragged_mask(
        self, ax, ragged_mask: torch.Tensor, token_table: List[TokenMeta]
    ):
        """Plot the ragged causal mask."""
        ax.set_title("Ragged Causal Mask")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

        # Convert to numpy for plotting
        mask_np = ragged_mask.cpu().numpy()

        # Create colored mask
        colored_mask = np.zeros((*mask_np.shape, 3))
        for i in range(mask_np.shape[0]):
            for j in range(mask_np.shape[1]):
                if mask_np[i, j]:
                    # Use different colors for different requests
                    req_id = token_table[i].req_id
                    if token_table[j].req_id == req_id:
                        color_idx = hash(req_id) % len(self.colors)
                        color = plt.cm.colors.to_rgb(self.colors[color_idx])
                        colored_mask[i, j] = (*color, 0.8)
                    else:
                        colored_mask[i, j] = (
                            0.8,
                            0.8,
                            0.8,
                            0.3,
                        )  # Gray for cross-request

        ax.imshow(colored_mask, origin="upper", interpolation="nearest")

        # Add request separators
        current_req = None
        for i, meta in enumerate(token_table):
            if meta.req_id != current_req:
                current_req = meta.req_id
                ax.axhline(y=i - 0.5, color="white", linewidth=2)
                ax.axvline(x=i - 0.5, color="white", linewidth=2)

        ax.set_xlim(-0.5, len(token_table) - 0.5)
        ax.set_ylim(-0.5, len(token_table) - 0.5)

    def _plot_attention_pattern(
        self, ax, ragged_mask: torch.Tensor, token_table: List[TokenMeta]
    ):
        """Plot the attention pattern visualization."""
        ax.set_title("Attention Pattern (Query attends to Keys)")
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")

        mask_np = ragged_mask.cpu().numpy()

        # Create attention visualization
        attention_vis = np.zeros((*mask_np.shape, 4))
        for i in range(mask_np.shape[0]):
            for j in range(mask_np.shape[1]):
                if mask_np[i, j]:
                    if token_table[i].req_id == token_table[j].req_id:
                        # Same request - show attention
                        attention_vis[i, j] = [0.2, 0.6, 1.0, 0.8]  # Blue
                    else:
                        # Different request - no attention
                        attention_vis[i, j] = [1.0, 1.0, 1.0, 0.1]  # Light gray
                else:
                    attention_vis[i, j] = [1.0, 1.0, 1.0, 0.0]  # Transparent

        ax.imshow(attention_vis, origin="upper", interpolation="nearest")

        # Add token labels
        for i, meta in enumerate(token_table):
            if i % 2 == 0:  # Show every other label to avoid crowding
                ax.text(
                    len(token_table) + 1,
                    i,
                    f"{meta.req_id}\\n{meta.pos_in_seq}",
                    va="center",
                    fontsize=8,
                )

    def visualize_scheduler_timeline(
        self, scheduler: Scheduler, max_steps: int = 10, save_path: Optional[str] = None
    ):
        """Visualize scheduler execution timeline."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(
            "Continuous Batching Scheduler Timeline", fontsize=14, fontweight="bold"
        )

        step_data = []
        request_data = {req.req_id: [] for req in scheduler.active}

        # Run scheduler and collect data
        for step in range(max_steps):
            if not scheduler.active:
                break

            # Record state before step
            prefill_requests = [r for r in scheduler.active if r.in_prefill]
            decode_requests = [r for r in scheduler.active if r.in_decode]

            step_info = {
                "step": step,
                "prefill_count": len(prefill_requests),
                "decode_count": len(decode_requests),
                "total_active": len(scheduler.active),
            }
            step_data.append(step_info)

            # Record per-request state
            for req in scheduler.active:
                req_info = {
                    "step": step,
                    "phase": (
                        "prefill"
                        if req.in_prefill
                        else "decode" if req.in_decode else "finished"
                    ),
                    "tokens_generated": len(req.generated_tokens),
                    "prefill_progress": (
                        req.prefill_pos / req.prompt.size(0) if req.in_prefill else 1.0
                    ),
                }
                request_data[req.req_id].append(req_info)

            # Execute step
            scheduler.step()

        # Plot step statistics
        self._plot_step_statistics(ax1, step_data)

        # Plot request timeline
        self._plot_request_timeline(ax2, request_data)

        # Plot token utilization
        self._plot_token_utilization(ax3, step_data)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_step_statistics(self, ax, step_data: List[Dict]):
        """Plot step-level statistics."""
        ax.set_title("Step-Level Statistics")
        ax.set_xlabel("Step")
        ax.set_ylabel("Count")

        steps = [d["step"] for d in step_data]
        prefill_counts = [d["prefill_count"] for d in step_data]
        decode_counts = [d["decode_count"] for d in step_data]
        total_active = [d["total_active"] for d in step_data]

        width = 0.25
        x = np.arange(len(steps))

        ax.bar(
            x - width,
            prefill_counts,
            width,
            label="Prefill Requests",
            color="#4ECDC4",
            alpha=0.8,
        )
        ax.bar(
            x, decode_counts, width, label="Decode Requests", color="#FF6B6B", alpha=0.8
        )
        ax.bar(
            x + width,
            total_active,
            width,
            label="Total Active",
            color="#45B7D1",
            alpha=0.8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(steps)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_request_timeline(self, ax, request_data: Dict[str, List[Dict]]):
        """Plot per-request timeline."""
        ax.set_title("Request Execution Timeline")
        ax.set_xlabel("Step")
        ax.set_ylabel("Request ID")

        y_pos = 0
        for req_id, req_steps in request_data.items():
            x_positions = []
            phases = []
            colors = []

            for step_info in req_steps:
                x_positions.append(step_info["step"])
                phases.append(step_info["phase"])

                if step_info["phase"] == "prefill":
                    colors.append("#4ECDC4")
                elif step_info["phase"] == "decode":
                    colors.append("#FF6B6B")
                else:
                    colors.append("#96CEB4")

            # Plot timeline for this request
            for i in range(len(x_positions)):
                ax.scatter(x_positions[i], y_pos, s=100, c=colors[i], alpha=0.8)

            # Connect points
            if len(x_positions) > 1:
                ax.plot(
                    x_positions,
                    [y_pos] * len(x_positions),
                    color="gray",
                    alpha=0.5,
                    linewidth=1,
                )

            # Add request label
            ax.text(-0.5, y_pos, req_id, ha="right", va="center", fontweight="bold")
            y_pos += 1

        ax.set_xlim(
            -1,
            max(
                max(
                    [
                        max([s["step"] for s in steps])
                        for steps in request_data.values()
                    ],
                    default=0,
                )
            )
            + 1,
        )
        ax.set_ylim(-0.5, len(request_data) - 0.5)
        ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#4ECDC4", alpha=0.8, label="Prefill"),
            Patch(facecolor="#FF6B6B", alpha=0.8, label="Decode"),
            Patch(facecolor="#96CEB4", alpha=0.8, label="Finished"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    def _plot_token_utilization(self, ax, step_data: List[Dict]):
        """Plot token utilization over time."""
        ax.set_title("Token Budget Utilization")
        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens")

        steps = [d["step"] for d in step_data]
        prefill_counts = [d["prefill_count"] for d in step_data]
        decode_counts = [d["decode_count"] for d in step_data]

        # Stack plot for token utilization
        ax.fill_between(
            steps, 0, decode_counts, alpha=0.6, color="#FF6B6B", label="Decode Tokens"
        )
        ax.fill_between(
            steps,
            decode_counts,
            [d + p for d, p in zip(decode_counts, prefill_counts)],
            alpha=0.6,
            color="#4ECDC4",
            label="Prefill Tokens",
        )

        # Add total tokens line
        total_tokens = [d + p for d, p in zip(decode_counts, prefill_counts)]
        ax.plot(
            steps,
            total_tokens,
            "o-",
            color="#45B7D1",
            linewidth=2,
            label="Total Tokens",
        )

        ax.legend()
        ax.grid(True, alpha=0.3)

    def visualize_kv_cache_layout(
        self,
        cache_type: str = "dense",
        sequence_length: int = 100,
        block_size: int = 32,
        save_path: Optional[str] = None,
    ):
        """Visualize KV cache memory layout."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            f"KV Cache Memory Layout ({cache_type.capitalize()})",
            fontsize=14,
            fontweight="bold",
        )

        if cache_type == "dense":
            self._visualize_dense_cache(ax1, ax2, sequence_length)
        elif cache_type == "paged":
            self._visualize_paged_cache(ax1, ax2, sequence_length, block_size)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def _visualize_dense_cache(self, ax1, ax2, sequence_length: int):
        """Visualize dense KV cache layout."""
        ax1.set_title("Dense Cache - Key Layout")
        ax2.set_title("Dense Cache - Value Layout")

        # Simulate dense cache as continuous blocks
        for ax in [ax1, ax2]:
            # Draw contiguous memory blocks
            block_height = 2
            for i in range(0, sequence_length, 20):
                rect = patches.Rectangle(
                    (0, i),
                    10,
                    min(block_height, sequence_length - i),
                    linewidth=1,
                    edgecolor="black",
                    facecolor=self.colors[i // 20 % len(self.colors)],
                    alpha=0.7,
                )
                ax.add_patch(rect)
                ax.text(
                    5,
                    i + block_height / 2,
                    f"Tokens {i}-{min(i+19, sequence_length)}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

            ax.set_xlim(0, 10)
            ax.set_ylim(0, sequence_length)
            ax.set_xlabel("Feature Dimension")
            ax.set_ylabel("Sequence Position")
            ax.grid(True, alpha=0.3)

    def _visualize_paged_cache(self, ax1, ax2, sequence_length: int, block_size: int):
        """Visualize paged KV cache layout."""
        ax1.set_title("Paged Cache - Key Blocks")
        ax2.set_title("Paged Cache - Value Blocks")

        num_blocks = (sequence_length + block_size - 1) // block_size

        for ax in [ax1, ax2]:
            # Draw memory pages
            for i in range(num_blocks):
                start_pos = i * block_size
                end_pos = min(start_pos + block_size, sequence_length)
                actual_size = end_pos - start_pos

                # Position blocks in a grid
                x = (i % 4) * 3
                y = (i // 4) * (block_size + 5)

                rect = patches.Rectangle(
                    (x, y),
                    2,
                    actual_size,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=self.colors[i % len(self.colors)],
                    alpha=0.7,
                )
                ax.add_patch(rect)

                ax.text(
                    x + 1,
                    y + actual_size / 2,
                    f"Block {i}\\n({actual_size} tokens)",
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

            ax.set_xlim(-1, 13)
            max_y = ((num_blocks - 1) // 4) * (block_size + 5) + block_size
            ax.set_ylim(-1, max_y)
            ax.set_xlabel("Memory Address")
            ax.set_ylabel("Memory Layout")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")


def run_visualization_demo():
    """Run a comprehensive visualization demo."""
    print("🎨 cb-lab Visualization Demo")
    print("=" * 40)

    visualizer = ContinuousBatchingVisualizer()

    # Create test data for ragged mask visualization
    chunks_data = [
        ("req1", 0, torch.randn(3, 16)),
        ("req2", 0, torch.randn(2, 16)),
        ("req1", 3, torch.randn(2, 16)),
        ("req3", 0, torch.randn(4, 16)),
    ]

    print("1. Visualizing ragged batching and causal mask...")
    visualizer.visualize_ragged_mask(chunks_data, "ragged_mask_demo.png")

    # Create scheduler for timeline visualization
    print("2. Visualizing scheduler timeline...")
    model = TinyLLM(16)
    scheduler = Scheduler(model, max_tokens_per_step=6, prefill_chunk_size=3)

    # Add some test requests
    requests = [
        Request("req_A", torch.randn(5, 16), 3, DenseKVCache(torch.device("cpu"))),
        Request("req_B", torch.randn(8, 16), 4, DenseKVCache(torch.device("cpu"))),
        Request("req_C", torch.randn(3, 16), 2, DenseKVCache(torch.device("cpu"))),
    ]

    for req in requests:
        scheduler.add_request(req)

    visualizer.visualize_scheduler_timeline(
        scheduler, max_steps=15, save_path="timeline_demo.png"
    )

    # Visualize KV cache layouts
    print("3. Visualizing dense KV cache layout...")
    visualizer.visualize_kv_cache_layout(
        "dense", sequence_length=100, save_path="dense_cache_demo.png"
    )

    print("4. Visualizing paged KV cache layout...")
    visualizer.visualize_kv_cache_layout(
        "paged", sequence_length=100, block_size=32, save_path="paged_cache_demo.png"
    )

    print("✅ Visualization demo completed! Check the generated PNG files.")


if __name__ == "__main__":
    run_visualization_demo()
