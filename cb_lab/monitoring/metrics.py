"""Performance monitoring and metrics collection for cb-lab framework."""

import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager

import torch

from cb_lab.exceptions import MonitoringError


@dataclass
class StepMetrics:
    """Metrics collected for a single scheduler step."""

    step_id: int
    decode_tokens: int
    prefill_tokens: int
    active_requests: int
    finished_requests: int
    step_duration: float
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    status: str = "unknown"


@dataclass
class RequestMetrics:
    """Metrics for a single request's lifecycle."""

    req_id: str
    start_time: float
    end_time: Optional[float] = None
    total_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    peak_memory_mb: float = 0.0
    status: str = "pending"


class MemoryProfiler:
    """Memory usage profiler for monitoring system and GPU memory."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.baseline_memory = self._get_current_memory()
        self.cuda_available = torch.cuda.is_available()

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0

    def get_gpu_memory(self) -> float:
        """Get GPU memory usage in MB."""
        if not self.cuda_available:
            return 0.0

        try:
            return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0

    def get_memory_delta(self) -> float:
        """Get memory usage change from baseline in MB."""
        return self._get_current_memory() - self.baseline_memory

    def set_baseline(self) -> None:
        """Set baseline memory measurement."""
        self.baseline_memory = self._get_current_memory()

    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self._get_current_memory()

    def log_memory_usage(self, context: str = "") -> Dict[str, float]:
        """Log current memory usage with context."""
        current_memory = self._get_current_memory()
        gpu_memory = self.get_gpu_memory()
        memory_delta = current_memory - self.baseline_memory

        return {
            "context": context,  # type: ignore[dict-item]
            "current_memory_mb": current_memory,
            "gpu_memory_mb": gpu_memory,
            "memory_delta_mb": memory_delta,
        }


class MetricsCollector:
    """Centralized metrics collection for cb-lab operations."""

    def __init__(self, enable_memory_profiling: bool = True):
        self.step_metrics: List[StepMetrics] = []
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self.memory_profiler = MemoryProfiler() if enable_memory_profiling else None
        self.enable_memory_profiling = enable_memory_profiling
        self._lock = threading.Lock()

    def record_step_start(self, step_id: int) -> Dict[str, Any]:
        """Record the start of a scheduler step."""
        with self._lock:
            memory_info = {}
            if self.memory_profiler:
                memory_info = self.memory_profiler.log_memory_usage(
                    f"step_{step_id}_start"
                )

            return {
                "step_id": step_id,
                "start_time": time.time(),
                "memory_info": memory_info,
            }

    def record_step(
        self,
        step_id: int,
        decode_tokens: int,
        prefill_tokens: int,
        active_requests: int,
        finished_requests: int,
        step_duration: float,
        status: str = "completed",
    ) -> StepMetrics:
        """Record step completion metrics."""
        with self._lock:
            memory_usage = 0.0
            gpu_memory = 0.0

            if self.memory_profiler:
                memory_info = self.memory_profiler.log_memory_usage(
                    f"step_{step_id}_end"
                )
                memory_usage = memory_info.get("memory_delta_mb", 0.0)
                gpu_memory = memory_info.get("gpu_memory_mb", 0.0)

            metrics = StepMetrics(
                step_id=step_id,
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
                active_requests=active_requests,
                finished_requests=finished_requests,
                step_duration=step_duration,
                memory_usage_mb=memory_usage,
                gpu_memory_mb=gpu_memory,
                status=status,
            )

            self.step_metrics.append(metrics)
            return metrics

    def start_request_tracking(self, req_id: str, total_tokens: int = 0) -> None:
        """Start tracking a new request."""
        with self._lock:
            if req_id in self.request_metrics:
                raise MonitoringError(f"Request {req_id} is already being tracked")

            self.request_metrics[req_id] = RequestMetrics(
                req_id=req_id,
                start_time=time.time(),
                total_tokens=total_tokens,
                status="started",
            )

    def update_request_progress(
        self,
        req_id: str,
        prefill_tokens: Optional[int] = None,
        decode_tokens: Optional[int] = None,
        status: Optional[str] = None,
    ) -> None:
        """Update request progress metrics."""
        with self._lock:
            if req_id not in self.request_metrics:
                raise MonitoringError(f"Request {req_id} is not being tracked")

            metrics = self.request_metrics[req_id]
            if prefill_tokens is not None:
                metrics.prefill_tokens = prefill_tokens
            if decode_tokens is not None:
                metrics.decode_tokens = decode_tokens
            if status is not None:
                metrics.status = status

            # Update peak memory usage
            if self.memory_profiler:
                current_memory = self.memory_profiler.get_memory_delta()
                metrics.peak_memory_mb = max(metrics.peak_memory_mb, current_memory)

    def finish_request(self, req_id: str, status: str = "completed") -> None:
        """Mark request as finished."""
        with self._lock:
            if req_id not in self.request_metrics:
                raise MonitoringError(f"Request {req_id} is not being tracked")

            metrics = self.request_metrics[req_id]
            metrics.end_time = time.time()
            metrics.status = status

    def get_step_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all recorded steps."""
        with self._lock:
            if not self.step_metrics:
                return {"total_steps": 0, "message": "No steps recorded"}

            total_steps = len(self.step_metrics)
            total_decode_tokens = sum(s.decode_tokens for s in self.step_metrics)
            total_prefill_tokens = sum(s.prefill_tokens for s in self.step_metrics)
            avg_step_duration = (
                sum(s.step_duration for s in self.step_metrics) / total_steps
            )
            total_memory_delta = sum(s.memory_usage_mb for s in self.step_metrics)

            return {
                "total_steps": total_steps,
                "total_decode_tokens": total_decode_tokens,
                "total_prefill_tokens": total_prefill_tokens,
                "total_tokens_processed": total_decode_tokens + total_prefill_tokens,
                "avg_step_duration": avg_step_duration,
                "total_memory_delta_mb": total_memory_delta,
                "tokens_per_second": (total_decode_tokens + total_prefill_tokens)
                / max(avg_step_duration * total_steps, 0.001),
                "latest_gpu_memory_mb": max(
                    (s.gpu_memory_mb for s in self.step_metrics), default=0.0
                ),
            }

    def get_request_summary(self, req_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific request."""
        with self._lock:
            if req_id not in self.request_metrics:
                return None

            metrics = self.request_metrics[req_id]
            duration = (metrics.end_time or time.time()) - metrics.start_time
            total_tokens = metrics.prefill_tokens + metrics.decode_tokens

            return {
                "req_id": metrics.req_id,
                "duration": duration,
                "total_tokens": total_tokens,
                "prefill_tokens": metrics.prefill_tokens,
                "decode_tokens": metrics.decode_tokens,
                "tokens_per_second": total_tokens / max(duration, 0.001),
                "peak_memory_mb": metrics.peak_memory_mb,
                "status": metrics.status,
                "completed": metrics.end_time is not None,
            }

    def get_all_requests_summary(self) -> List[Dict[str, Any]]:
        """Get summary for all tracked requests."""
        with self._lock:
            return [self.get_request_summary(req_id) for req_id in self.request_metrics]

    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self.step_metrics.clear()
            self.request_metrics.clear()
            if self.memory_profiler:
                self.memory_profiler.baseline_memory = (
                    self.memory_profiler._get_current_memory()
                )


@contextmanager
def measure_step(collector: MetricsCollector, step_id: int):
    """Context manager for measuring scheduler step execution."""
    start_info = collector.record_step_start(step_id)
    start_time = start_info["start_time"]

    try:
        yield start_info
    except Exception as e:
        # Record failed step
        duration = time.time() - start_time
        collector.record_step(
            step_id=step_id,
            decode_tokens=0,
            prefill_tokens=0,
            active_requests=0,
            finished_requests=0,
            step_duration=duration,
            status="failed",
        )
        raise


class PerformanceBenchmark:
    """Performance benchmarking utilities for cb-lab."""

    def __init__(self) -> None:
        self.results: Dict[str, List[float]] = {}

    def benchmark_function(
        self, func: Callable, name: str, num_runs: int = 10, warmup_runs: int = 2
    ) -> Dict[str, float]:
        """Benchmark a function's execution time."""
        if name not in self.results:
            self.results[name] = []

        # Warmup runs
        for _ in range(warmup_runs):
            func()

        # Actual benchmark runs
        run_times = []
        for _ in range(num_runs):
            start_time = time.time()
            func()
            end_time = time.time()
            run_times.append(end_time - start_time)

        self.results[name].extend(run_times)

        return {
            "name": name,
            "num_runs": num_runs,
            "mean_time": sum(run_times) / len(run_times),
            "min_time": min(run_times),
            "max_time": max(run_times),
            "std_time": (
                sum((t - sum(run_times) / len(run_times)) ** 2 for t in run_times)
                / len(run_times)
            )
            ** 0.5,
        }

    def compare_functions(
        self, functions: Dict[str, Callable], num_runs: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple functions."""
        results = {}
        for name, func in functions.items():
            results[name] = self.benchmark_function(func, name, num_runs)
        return results

    def get_benchmark_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all benchmark results."""
        summary = {}
        for name, times in self.results.items():
            if not times:
                continue
            summary[name] = {
                "total_runs": len(times),
                "mean_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
            }
        return summary
