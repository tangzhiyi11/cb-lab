"""Built-in plugins for cb-lab framework."""

import time
import json
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from collections import defaultdict

import torch

from cb_lab.plugins.base import SchedulerPlugin, AttentionPlugin, CachePlugin

if TYPE_CHECKING:
    from cb_lab.core.scheduler import Scheduler
    from cb_lab.core.request import Request
    from cb_lab.core.kv_cache import BaseKVCache


class LoggingPlugin(SchedulerPlugin):
    """Plugin for detailed logging of scheduler operations."""

    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.log_file = log_file
        self.verbose = verbose
        self.step_count = 0
        self.request_lifecycles: Dict[str, Dict[str, Any]] = {}

    def before_step(self, scheduler: Any) -> None:
        """Called before each scheduler step."""
        self.step_count += 1
        active_requests = [r.req_id for r in scheduler.active]
        message = f"[Step {self.step_count}] Starting with {len(active_requests)} active requests: {active_requests}"

        if self.verbose:
            print(message)

        if self.log_file:
            self._write_log(message)

    def after_step(self, scheduler: Any, step_stats: Dict[str, Any]) -> None:
        """Called after each scheduler step."""
        message = (
            f"[Step {self.step_count}] Completed: "
            f"decode={step_stats['decode_tokens']}, "
            f"prefill={step_stats['prefill_tokens']}, "
            f"duration={step_stats['step_duration']:.4f}s, "
            f"active={step_stats['active_requests']}"
        )

        if self.verbose:
            print(message)

        if self.log_file:
            self._write_log(message)

    def on_request_added(self, scheduler: Any, request: Any) -> None:
        """Called when a request is added to the scheduler."""
        self.request_lifecycles[request.req_id] = {
            "added_time": time.time(),
            "prompt_length": request.prompt.size(0),
            "max_tokens": request.max_new_tokens,
            "status": "added",
        }

        message = f"Request {request.req_id} added: prompt_len={request.prompt.size(0)}, max_tokens={request.max_new_tokens}"

        if self.verbose:
            print(message)

        if self.log_file:
            self._write_log(message)

    def on_request_finished(self, scheduler: Any, request: Any) -> None:
        """Called when a request finishes."""
        if request.req_id in self.request_lifecycles:
            lifecycle = self.request_lifecycles[request.req_id]
            lifecycle["finished_time"] = time.time()
            lifecycle["total_duration"] = (
                lifecycle["finished_time"] - lifecycle["added_time"]
            )
            lifecycle["tokens_generated"] = len(request.generated_tokens)
            lifecycle["status"] = "finished"

        message = f"Request {request.req_id} finished: generated {len(request.generated_tokens)} tokens"

        if self.verbose:
            print(message)

        if self.log_file:
            self._write_log(message)

    def _write_log(self, message: str) -> None:
        """Write message to log file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        except Exception as e:
            print(f"Failed to write to log file {self.log_file}: {e}")

    def get_request_statistics(self) -> Dict[str, Any]:
        """Get statistics for all tracked requests."""
        if not self.request_lifecycles:
            return {}

        finished_requests = [
            r for r in self.request_lifecycles.values() if r["status"] == "finished"
        ]

        if not finished_requests:
            return {"total_requests": len(self.request_lifecycles), "finished": 0}

        avg_duration = sum(r["total_duration"] for r in finished_requests) / len(
            finished_requests
        )
        total_tokens = sum(r["tokens_generated"] for r in finished_requests)

        return {
            "total_requests": len(self.request_lifecycles),
            "finished_requests": len(finished_requests),
            "average_duration": avg_duration,
            "total_tokens_generated": total_tokens,
            "average_tokens_per_request": total_tokens / len(finished_requests),
        }


class MetricsPlugin(SchedulerPlugin):
    """Plugin for collecting detailed metrics."""

    def __init__(self) -> None:
        self.step_metrics: List[Dict[str, Any]] = []
        self.request_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.start_time = time.time()

    def before_step(self, scheduler: Any) -> None:
        """Record pre-step metrics."""
        return  # No pre-step metrics needed

    def after_step(self, scheduler: Any, step_stats: Dict[str, Any]) -> None:
        """Collect post-step metrics."""
        enhanced_stats = step_stats.copy()
        enhanced_stats["timestamp"] = time.time() - self.start_time

        # Calculate additional metrics
        total_tokens = step_stats["decode_tokens"] + step_stats["prefill_tokens"]
        if step_stats["step_duration"] > 0:
            enhanced_stats["tokens_per_second"] = (
                total_tokens / step_stats["step_duration"]
            )
        else:
            enhanced_stats["tokens_per_second"] = 0

        # Request-specific metrics
        active_reqs = [r for r in scheduler.active if not r.finished]
        enhanced_stats["request_details"] = [
            {
                "req_id": r.req_id,
                "phase": "prefill" if r.in_prefill else "decode",
                "tokens_generated": len(r.generated_tokens),
                "prefill_progress": (
                    r.prefill_pos / r.prompt.size(0) if r.in_prefill else 1.0
                ),
            }
            for r in active_reqs
        ]

        self.step_metrics.append(enhanced_stats)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from collected metrics."""
        if not self.step_metrics:
            return {}

        total_steps = len(self.step_metrics)
        total_decode_tokens = sum(s["decode_tokens"] for s in self.step_metrics)
        total_prefill_tokens = sum(s["prefill_tokens"] for s in self.step_metrics)
        total_tokens = total_decode_tokens + total_prefill_tokens

        avg_step_duration = (
            sum(s["step_duration"] for s in self.step_metrics) / total_steps
        )
        max_step_duration = max(s["step_duration"] for s in self.step_metrics)
        min_step_duration = min(s["step_duration"] for s in self.step_metrics)

        avg_throughput = (
            sum(s.get("tokens_per_second", 0) for s in self.step_metrics) / total_steps
        )
        max_throughput = max(s.get("tokens_per_second", 0) for s in self.step_metrics)

        return {
            "total_steps": total_steps,
            "total_tokens_processed": total_tokens,
            "decode_tokens": total_decode_tokens,
            "prefill_tokens": total_prefill_tokens,
            "average_step_duration": avg_step_duration,
            "max_step_duration": max_step_duration,
            "min_step_duration": min_step_duration,
            "average_throughput": avg_throughput,
            "peak_throughput": max_throughput,
            "total_execution_time": (
                self.step_metrics[-1]["timestamp"] if self.step_metrics else 0
            ),
        }


class AdaptiveSchedulingPlugin(SchedulerPlugin):
    """Plugin that adapts scheduling parameters based on workload."""

    def __init__(
        self, initial_prefill_chunk_size: int = 4, adaptive_threshold: int = 5
    ):
        self.initial_prefill_chunk_size = initial_prefill_chunk_size
        self.adaptive_threshold = adaptive_threshold
        self.recent_decode_counts: List[int] = []

    def before_step(self, scheduler: Any) -> None:
        """Adjust prefill chunk size based on recent decode activity."""
        if len(self.recent_decode_counts) >= self.adaptive_threshold:
            avg_decode = sum(self.recent_decode_counts) / len(self.recent_decode_counts)

            # If lots of decode activity, reduce prefill chunk size to prioritize decode
            if avg_decode > scheduler.max_tokens_per_step * 0.7:
                new_chunk_size = max(1, scheduler.prefill_chunk_size - 1)
                if new_chunk_size != scheduler.prefill_chunk_size:
                    print(
                        f"[AdaptiveScheduling] Reducing prefill chunk size: {scheduler.prefill_chunk_size} -> {new_chunk_size}"
                    )
                    scheduler.prefill_chunk_size = new_chunk_size

            # If little decode activity, increase prefill chunk size
            elif avg_decode < scheduler.max_tokens_per_step * 0.3:
                new_chunk_size = min(
                    scheduler.max_tokens_per_step, scheduler.prefill_chunk_size + 1
                )
                if new_chunk_size != scheduler.prefill_chunk_size:
                    print(
                        f"[AdaptiveScheduling] Increasing prefill chunk size: {scheduler.prefill_chunk_size} -> {new_chunk_size}"
                    )
                    scheduler.prefill_chunk_size = new_chunk_size

    def after_step(self, scheduler: Any, step_stats: Dict[str, Any]) -> None:
        """Track decode counts for adaptive decisions."""
        self.recent_decode_counts.append(step_stats["decode_tokens"])

        # Keep only recent history
        if len(self.recent_decode_counts) > self.adaptive_threshold:
            self.recent_decode_counts = self.recent_decode_counts[
                -self.adaptive_threshold :
            ]


class AttentionVisualizationPlugin(AttentionPlugin):
    """Plugin that visualizes attention patterns."""

    def __init__(self, save_dir: str = "attention_viz"):
        self.save_dir = save_dir
        self.attention_patterns: List[Dict[str, Any]] = []

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute attention and store pattern information."""
        # Standard attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        # Store pattern information
        self.attention_patterns.append(
            {
                "timestamp": time.time(),
                "q_shape": q.shape,
                "k_shape": k.shape,
                "v_shape": v.shape,
                "attention_shape": attention_weights.shape,
                "sparsity": (attention_weights == 0).float().mean().item(),
                "max_attention": attention_weights.max().item(),
                "mean_attention": attention_weights.mean().item(),
            }
        )

        return output

    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get statistics about attention patterns."""
        if not self.attention_patterns:
            return {}

        avg_sparsity = sum(p["sparsity"] for p in self.attention_patterns) / len(
            self.attention_patterns
        )
        max_sparsity = max(p["sparsity"] for p in self.attention_patterns)
        min_sparsity = min(p["sparsity"] for p in self.attention_patterns)

        avg_max_attention = sum(
            p["max_attention"] for p in self.attention_patterns
        ) / len(self.attention_patterns)
        avg_mean_attention = sum(
            p["mean_attention"] for p in self.attention_patterns
        ) / len(self.attention_patterns)

        return {
            "total_computations": len(self.attention_patterns),
            "average_sparsity": avg_sparsity,
            "max_sparsity": max_sparsity,
            "min_sparsity": min_sparsity,
            "average_max_attention": avg_max_attention,
            "average_mean_attention": avg_mean_attention,
        }


class CacheCompressionPlugin(CachePlugin):
    """Plugin that simulates cache compression for memory efficiency."""

    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio
        self.compressed_entries = 0
        self.original_size = 0
        self.compressed_size = 0

    def before_append(
        self, cache: Any, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> None:
        """Simulate compression before cache append."""
        self.original_size += k_new.numel() + v_new.numel()

        # Simulate compression by calculating theoretical size
        compressed_size = int((k_new.numel() + v_new.numel()) * self.compression_ratio)
        self.compressed_size += compressed_size
        self.compressed_entries += 1

    def after_append(
        self, cache: Any, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> None:
        """Log compression statistics."""
        if self.compressed_entries % 10 == 0:  # Log every 10 entries
            compression_efficiency = (
                (self.original_size - self.compressed_size) / self.original_size * 100
            )
            print(
                f"[CacheCompression] Processed {self.compressed_entries} entries, "
                f"compression efficiency: {compression_efficiency:.1f}%"
            )

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        if self.original_size == 0:
            return {}

        return {
            "entries_processed": self.compressed_entries,
            "original_size_mb": self.original_size
            * 4
            / 1024
            / 1024,  # Assuming float32
            "compressed_size_mb": self.compressed_size * 4 / 1024 / 1024,
            "compression_ratio": self.compression_ratio,
            "space_saved_percent": (self.original_size - self.compressed_size)
            / self.original_size
            * 100,
        }
