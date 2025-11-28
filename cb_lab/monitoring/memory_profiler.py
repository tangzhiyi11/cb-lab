"""Advanced memory profiling tools for cb-lab framework."""

import gc
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager

import torch
import psutil

from cb_lab.exceptions import MonitoringError


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage at a specific point in time."""

    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    torch_tensors_count: int = 0
    context: str = ""


class DetailedMemoryProfiler:
    """Detailed memory profiler for cb-lab operations."""

    def __init__(self, sampling_interval: float = 0.1):
        self.process = psutil.Process()
        self.sampling_interval = sampling_interval
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._context_stack: List[str] = []

    def take_snapshot(self, context: str = "") -> MemorySnapshot:
        """Take a snapshot of current memory usage."""
        try:
            memory_info = self.process.memory_info()
            gpu_allocated = 0.0
            gpu_reserved = 0.0
            tensor_count = 0

            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024

            # Count PyTorch tensors
            for obj in gc.get_objects():
                if isinstance(obj, torch.Tensor):
                    tensor_count += 1

            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                gpu_allocated_mb=gpu_allocated,
                gpu_reserved_mb=gpu_reserved,
                torch_tensors_count=tensor_count,
                context=context or self._get_current_context(),
            )

            self.snapshots.append(snapshot)
            return snapshot

        except Exception as e:
            raise MonitoringError(f"Failed to take memory snapshot: {e}") from e

    def _get_current_context(self) -> str:
        """Get current context from the stack."""
        return "->".join(self._context_stack) if self._context_stack else "global"

    def set_baseline(self, context: str = "baseline") -> None:
        """Set baseline memory measurement."""
        self.baseline_snapshot = self.take_snapshot(context)

    def get_memory_delta(
        self, snapshot: Optional[MemorySnapshot] = None
    ) -> Dict[str, float]:
        """Get memory usage delta from baseline."""
        if not self.baseline_snapshot:
            raise MonitoringError("Baseline not set. Call set_baseline() first.")

        current = (
            snapshot or self.snapshots[-1] if self.snapshots else self.baseline_snapshot
        )

        return {
            "rss_delta_mb": current.rss_mb - self.baseline_snapshot.rss_mb,
            "vms_delta_mb": current.vms_mb - self.baseline_snapshot.vms_mb,
            "gpu_allocated_delta_mb": current.gpu_allocated_mb
            - self.baseline_snapshot.gpu_allocated_mb,
            "gpu_reserved_delta_mb": current.gpu_reserved_mb
            - self.baseline_snapshot.gpu_reserved_mb,
            "tensor_count_delta": current.torch_tensors_count
            - self.baseline_snapshot.torch_tensors_count,
        }

    def start_monitoring(self) -> None:
        """Start continuous memory monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self.take_snapshot("monitoring")
                time.sleep(self.sampling_interval)
            except Exception:
                break

    def get_memory_timeline(self) -> List[MemorySnapshot]:
        """Get the complete memory usage timeline."""
        return self.snapshots.copy()

    def get_peak_memory_usage(self) -> Dict[str, float]:
        """Get peak memory usage statistics."""
        if not self.snapshots:
            return {}

        return {
            "peak_rss_mb": max(s.rss_mb for s in self.snapshots),
            "peak_vms_mb": max(s.vms_mb for s in self.snapshots),
            "peak_gpu_allocated_mb": max(s.gpu_allocated_mb for s in self.snapshots),
            "peak_gpu_reserved_mb": max(s.gpu_reserved_mb for s in self.snapshots),
            "peak_tensor_count": max(s.torch_tensors_count for s in self.snapshots),
        }

    def analyze_memory_growth(self) -> Dict[str, Any]:
        """Analyze memory growth patterns over time."""
        if len(self.snapshots) < 2:
            return {"message": "Insufficient data for analysis"}

        # Calculate growth rates
        time_diffs = []
        rss_diffs = []
        gpu_diffs = []

        for i in range(1, len(self.snapshots)):
            prev, curr = self.snapshots[i - 1], self.snapshots[i]
            time_diff = curr.timestamp - prev.timestamp
            if time_diff > 0:
                time_diffs.append(time_diff)
                rss_diffs.append(curr.rss_mb - prev.rss_mb)
                gpu_diffs.append(curr.gpu_allocated_mb - prev.gpu_allocated_mb)

        avg_rss_growth = sum(rss_diffs) / len(rss_diffs) if rss_diffs else 0
        avg_gpu_growth = sum(gpu_diffs) / len(gpu_diffs) if gpu_diffs else 0

        return {
            "total_duration": self.snapshots[-1].timestamp
            - self.snapshots[0].timestamp,
            "total_rss_growth": self.snapshots[-1].rss_mb - self.snapshots[0].rss_mb,
            "total_gpu_growth": self.snapshots[-1].gpu_allocated_mb
            - self.snapshots[0].gpu_allocated_mb,
            "avg_rss_growth_rate_mb_per_sec": avg_rss_growth,
            "avg_gpu_growth_rate_mb_per_sec": avg_gpu_growth,
            "snapshots_analyzed": len(self.snapshots),
        }

    def clear_snapshots(self) -> None:
        """Clear all memory snapshots."""
        self.snapshots.clear()
        self.baseline_snapshot = None

    @contextmanager
    def profile_context(self, context_name: str):
        """Context manager for profiling a specific code block."""
        self._context_stack.append(context_name)
        start_snapshot = self.take_snapshot(f"{context_name}_start")

        try:
            yield
        finally:
            end_snapshot = self.take_snapshot(f"{context_name}_end")
            self._context_stack.pop()

            # Calculate and report memory usage for this context
            memory_delta = {
                "rss_mb": end_snapshot.rss_mb - start_snapshot.rss_mb,
                "gpu_allocated_mb": end_snapshot.gpu_allocated_mb
                - start_snapshot.gpu_allocated_mb,
                "tensor_count": end_snapshot.torch_tensors_count
                - start_snapshot.torch_tensors_count,
            }

            print(
                f"[MemoryProfiler] {context_name}: "
                f"RSS {memory_delta['rss_mb']:+.2f}MB, "
                f"GPU {memory_delta['gpu_allocated_mb']:+.2f}MB, "
                f"Tensors {memory_delta['tensor_count']:+d}"
            )


class MemoryLeakDetector:
    """Tool for detecting potential memory leaks in cb-lab operations."""

    def __init__(self, threshold_mb: float = 10.0):
        self.threshold_mb = threshold_mb
        self.baseline_memory: Optional[float] = None
        self.measurements: List[float] = []

    def set_baseline(self) -> None:
        """Set baseline memory measurement."""
        self.baseline_memory = self._get_memory_usage()
        self.measurements = [self.baseline_memory]

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def record_measurement(self, label: str = "") -> None:
        """Record a memory measurement."""
        current = self._get_memory_usage()
        self.measurements.append(current)

        if self.baseline_memory:
            growth = current - self.baseline_memory
            if growth > self.threshold_mb:
                print(
                    f"[MemoryLeakDetector] WARNING: Memory grew by {growth:.2f}MB "
                    f"(threshold: {self.threshold_mb}MB) - {label}"
                )

    def check_for_leaks(self) -> Dict[str, Any]:
        """Check if memory usage indicates a potential leak."""
        if len(self.measurements) < 3:
            return {"status": "insufficient_data"}

        # Simple linear regression to detect trend
        n = len(self.measurements)
        x = list(range(n))
        y = self.measurements

        # Calculate slope (memory growth per measurement)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        total_growth = self.measurements[-1] - (
            self.baseline_memory or self.measurements[0]
        )

        return {
            "status": "leak_detected" if slope > self.threshold_mb / n else "normal",
            "slope_mb_per_measurement": slope,
            "total_growth_mb": total_growth,
            "measurements_count": n,
            "threshold_mb": self.threshold_mb,
        }

    def reset(self) -> None:
        """Reset detector state."""
        self.baseline_memory = None
        self.measurements.clear()


class TensorTracker:
    """Track PyTorch tensor allocation and deallocation."""

    def __init__(self) -> None:
        self.allocations: Dict[int, Dict[str, Any]] = {}
        self.total_allocated_mb = 0.0
        self.peak_allocated_mb = 0.0

    def start_tracking(self) -> None:
        """Start tracking tensor allocations."""
        # Clear any existing allocations count
        gc.collect()

        # Count initial tensors
        initial_tensors = []
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor):
                initial_tensors.append(id(obj))

        self.allocations = {tid: {"initial": True} for tid in initial_tensors}
        self.total_allocated_mb = 0.0
        self.peak_allocated_mb = 0.0

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current tensor allocation statistics."""
        current_tensors = []
        total_size = 0.0

        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor):
                tensor_id = id(obj)
                if tensor_id not in self.allocations:
                    self.allocations[tensor_id] = {
                        "size_bytes": obj.numel() * obj.element_size(),
                        "shape": tuple(obj.shape),
                        "dtype": str(obj.dtype),
                        "device": str(obj.device),
                    }
                    total_size += (
                        self.allocations[tensor_id]["size_bytes"] / 1024 / 1024
                    )

                current_tensors.append(tensor_id)

        self.total_allocated_mb = total_size
        self.peak_allocated_mb = max(self.peak_allocated_mb, total_size)

        # Find deallocated tensors
        deallocated = set(self.allocations.keys()) - set(current_tensors)

        return {
            "current_tensor_count": len(current_tensors),
            "total_allocated_mb": total_size,
            "peak_allocated_mb": self.peak_allocated_mb,
            "total_tracked_allocations": len(self.allocations),
            "deallocated_count": len(deallocated),
            "leaked_count": len(
                [
                    tid
                    for tid in deallocated
                    if not self.allocations[tid].get("initial", False)
                ]
            ),
        }
