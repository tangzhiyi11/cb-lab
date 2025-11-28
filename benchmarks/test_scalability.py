"""Scalability benchmarks for cb-lab continuous batching system."""

import time
import statistics
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import torch

from cb_lab.core.scheduler import Scheduler
from cb_lab.core.request import Request
from cb_lab.core.kv_cache import DenseKVCache, PagedKVCache
from cb_lab.model.tiny_llm import TinyLLM
from cb_lab.monitoring.metrics import MetricsCollector


class ScalabilityBenchmark:
    """Comprehensive scalability testing for cb-lab components."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.model = TinyLLM(dim)
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def benchmark_scheduler_scaling(
        self,
        request_counts: List[int],
        max_tokens_per_step: int = 64,
        prompt_lengths: List[int] = None,
        num_runs: int = 3,
    ) -> Dict[str, Any]:
        """Benchmark scheduler performance with varying request counts."""
        if prompt_lengths is None:
            prompt_lengths = [10, 20, 30]

        results = {
            "request_counts": request_counts,
            "prompt_lengths": prompt_lengths,
            "completion_times": [],
            "throughput_tokens_per_sec": [],
            "memory_usage_mb": [],
            "step_statistics": [],
        }

        for req_count in request_counts:
            print(f"Testing {req_count} requests...")
            run_results = []

            for run in range(num_runs):
                # Create requests with varying lengths
                requests = self._create_test_requests(
                    req_count, prompt_lengths, max_new_tokens=5
                )

                # Track performance
                metrics = MetricsCollector()
                start_time = time.time()

                # Run scheduler
                scheduler = Scheduler(
                    self.model,
                    max_tokens_per_step=max_tokens_per_step,
                    prefill_chunk_size=16,
                )

                # Add requests and track
                for req in requests:
                    scheduler.add_request(req)
                    metrics.start_request_tracking(
                        req.req_id, total_tokens=req.prompt.size(0) + req.max_new_tokens
                    )

                # Run until completion
                steps = 0
                while scheduler.active and steps < 100:  # Safety limit
                    step_stats = scheduler.step()
                    metrics.record_step(**step_stats)
                    steps += 1

                completion_time = time.time() - start_time

                # Calculate statistics
                total_tokens = sum(
                    req.prompt.size(0) + req.max_new_tokens for req in requests
                )
                throughput = total_tokens / completion_time

                run_results.append(
                    {
                        "completion_time": completion_time,
                        "throughput": throughput,
                        "steps": steps,
                        "memory_usage": metrics.get_step_summary().get(
                            "total_memory_delta_mb", 0
                        ),
                    }
                )

            # Average across runs
            avg_result = {
                "request_count": req_count,
                "avg_completion_time": statistics.mean(
                    [r["completion_time"] for r in run_results]
                ),
                "avg_throughput": statistics.mean(
                    [r["throughput"] for r in run_results]
                ),
                "avg_steps": statistics.mean([r["steps"] for r in run_results]),
                "avg_memory_usage": statistics.mean(
                    [r["memory_usage"] for r in run_results]
                ),
            }

            results["completion_times"].append(avg_result["avg_completion_time"])
            results["throughput_tokens_per_sec"].append(avg_result["avg_throughput"])
            results["memory_usage_mb"].append(avg_result["avg_memory_usage"])
            results["step_statistics"].append(avg_result)

        self.results["scheduler_scaling"] = results
        return results

    def benchmark_kv_cache_performance(
        self,
        sequence_lengths: List[int],
        cache_types: List[str] = ["dense", "paged"],
        num_operations: int = 100,
    ) -> Dict[str, Any]:
        """Benchmark KV cache implementations."""
        results = {
            "sequence_lengths": sequence_lengths,
            "cache_types": cache_types,
            "append_times": {cache_type: [] for cache_type in cache_types},
            "retrieve_times": {cache_type: [] for cache_type in cache_types},
            "memory_usage": {cache_type: [] for cache_type in cache_types},
        }

        device = torch.device("cpu")

        for seq_len in sequence_lengths:
            print(f"Testing sequence length {seq_len}...")

            # Generate test data
            k_test = torch.randn(seq_len, self.dim, device=device)
            v_test = torch.randn(seq_len, self.dim, device=device)

            for cache_type in cache_types:
                # Initialize cache
                if cache_type == "dense":
                    cache = DenseKVCache(device)
                elif cache_type == "paged":
                    cache = PagedKVCache(block_size=32, device=device)
                else:
                    continue

                # Benchmark append operations
                append_times = []
                for _ in range(num_operations):
                    start = time.time()
                    cache.append(k_test, v_test)
                    append_times.append(time.time() - start)

                # Benchmark retrieve operations
                retrieve_times = []
                for _ in range(num_operations):
                    start = time.time()
                    k, v = cache.get_kv()
                    retrieve_times.append(time.time() - start)

                # Calculate memory usage
                memory_usage = (
                    k.numel() * k.element_size() + v.numel() * v.element_size()
                )
                memory_mb = memory_usage / 1024 / 1024

                results["append_times"][cache_type].append(
                    statistics.mean(append_times)
                )
                results["retrieve_times"][cache_type].append(
                    statistics.mean(retrieve_times)
                )
                results["memory_usage"][cache_type].append(memory_mb)

        self.results["kv_cache_performance"] = results
        return results

    def benchmark_attention_mechanisms(
        self, sequence_lengths: List[int], batch_sizes: List[int], num_runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark different attention mechanisms."""
        from cb_lab.attention.dense_attention import dense_causal_attention
        from cb_lab.attention.ragged_attention import ragged_attention

        results = {
            "sequence_lengths": sequence_lengths,
            "batch_sizes": batch_sizes,
            "dense_times": [],
            "ragged_times": [],
            "dense_memory": [],
            "ragged_memory": [],
        }

        device = torch.device("cpu")

        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                print(f"Testing seq_len={seq_len}, batch_size={batch_size}...")

                # Generate test data
                x = torch.randn(batch_size * seq_len, self.dim, device=device)

                # Dense attention test
                dense_times = []
                for _ in range(num_runs):
                    start = time.time()
                    q = x.view(batch_size, seq_len, self.dim)
                    k = x.view(batch_size, seq_len, self.dim)
                    v = x.view(batch_size, seq_len, self.dim)
                    _ = dense_causal_attention(q, k, v)
                    dense_times.append(time.time() - start)

                # Ragged attention test (simulate multiple requests)
                ragged_times = []
                for _ in range(num_runs):
                    start = time.time()
                    q = x
                    k = x
                    v = x
                    # Create ragged mask (simplified)
                    mask = torch.zeros(len(x), len(x), dtype=torch.bool, device=device)
                    for i in range(len(x)):
                        for j in range(i + 1):
                            mask[i, j] = True
                    _ = ragged_attention(q, k, v, mask)
                    ragged_times.append(time.time() - start)

                results["dense_times"].append(
                    {
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "time": statistics.mean(dense_times),
                    }
                )

                results["ragged_times"].append(
                    {
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "time": statistics.mean(ragged_times),
                    }
                )

        self.results["attention_performance"] = results
        return results

    def _create_test_requests(
        self, count: int, prompt_lengths: List[int], max_new_tokens: int = 5
    ) -> List[Request]:
        """Create test requests with varying prompt lengths."""
        requests = []
        device = torch.device("cpu")

        for i in range(count):
            # Vary prompt length
            prompt_len = prompt_lengths[i % len(prompt_lengths)]
            prompt = torch.randn(prompt_len, self.dim, device=device)

            # Alternate between cache types for variety
            if i % 2 == 0:
                kv_cache = DenseKVCache(device)
            else:
                kv_cache = PagedKVCache(block_size=16, device=device)

            request = Request(
                req_id=f"test_req_{i}",
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                kv_cache=kv_cache,
            )
            requests.append(request)

        return requests

    def plot_scalability_results(self, save_path: str = None) -> None:
        """Plot scalability benchmark results."""
        if "scheduler_scaling" not in self.results:
            print("No scheduler scaling results to plot")
            return

        results = self.results["scheduler_scaling"]
        request_counts = results["request_counts"]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("cb-lab Scalability Benchmarks", fontsize=16)

        # Plot 1: Completion Time vs Request Count
        axes[0, 0].plot(
            request_counts,
            results["completion_times"],
            "bo-",
            linewidth=2,
            markersize=8,
        )
        axes[0, 0].set_xlabel("Number of Requests")
        axes[0, 0].set_ylabel("Completion Time (seconds)")
        axes[0, 0].set_title("Completion Time Scalability")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Throughput vs Request Count
        axes[0, 1].plot(
            request_counts,
            results["throughput_tokens_per_sec"],
            "ro-",
            linewidth=2,
            markersize=8,
        )
        axes[0, 1].set_xlabel("Number of Requests")
        axes[0, 1].set_ylabel("Throughput (tokens/sec)")
        axes[0, 1].set_title("Throughput Scalability")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Memory Usage vs Request Count
        axes[1, 0].plot(
            request_counts, results["memory_usage_mb"], "go-", linewidth=2, markersize=8
        )
        axes[1, 0].set_xlabel("Number of Requests")
        axes[1, 0].set_ylabel("Memory Usage (MB)")
        axes[1, 0].set_title("Memory Usage Scalability")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Steps per Request vs Request Count
        steps_data = results["step_statistics"]
        steps_per_request = [
            stat["avg_steps"] / req_count
            for stat, req_count in zip(steps_data, request_counts)
        ]
        axes[1, 1].plot(
            request_counts, steps_per_request, "mo-", linewidth=2, markersize=8
        )
        axes[1, 1].set_xlabel("Number of Requests")
        axes[1, 1].set_ylabel("Average Steps per Request")
        axes[1, 1].set_title("Efficiency (Steps per Request)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        report = ["# cb-lab Scalability Benchmark Report\n"]

        if "scheduler_scaling" in self.results:
            results = self.results["scheduler_scaling"]
            report.append("## Scheduler Performance\n")

            max_throughput = max(results["throughput_tokens_per_sec"])
            min_time_idx = results["completion_times"].index(min(results["completion_times"]))
            min_time_per_request = (
                min(results["completion_times"]) / results["request_counts"][min_time_idx]
            )

            report.append(f"- **Peak Throughput**: {max_throughput:.2f} tokens/sec")
            report.append(
                f"- **Best Time per Request**: {min_time_per_request:.4f} sec"
            )
            report.append(
                f"- **Scalability Factor**: {results['throughput_tokens_per_sec'][-1] / results['throughput_tokens_per_sec'][0]:.2f}x"
            )

        if "kv_cache_performance" in self.results:
            results = self.results["kv_cache_performance"]
            report.append("\n## KV Cache Performance\n")

            # Compare dense vs paged performance
            dense_append = statistics.mean(results["append_times"]["dense"])
            paged_append = statistics.mean(results["append_times"]["paged"])

            report.append(
                f"- **Dense Cache Append Time**: {dense_append*1000:.2f} ms (avg)"
            )
            report.append(
                f"- **Paged Cache Append Time**: {paged_append*1000:.2f} ms (avg)"
            )
            report.append(f"- **Performance Ratio**: {dense_append/paged_append:.2f}x")

        if "attention_performance" in self.results:
            results = self.results["attention_performance"]
            report.append("\n## Attention Mechanism Performance\n")

            if results["dense_times"] and results["ragged_times"]:
                dense_avg = statistics.mean([t["time"] for t in results["dense_times"]])
                ragged_avg = statistics.mean(
                    [t["time"] for t in results["ragged_times"]]
                )
                report.append(f"- **Dense Attention**: {dense_avg*1000:.2f} ms (avg)")
                report.append(f"- **Ragged Attention**: {ragged_avg*1000:.2f} ms (avg)")
                report.append(
                    f"- **Performance Difference**: {((ragged_avg/dense_avg) - 1)*100:+.1f}%"
                )

        return "\n".join(report)


def run_comprehensive_benchmarks() -> Dict[str, Any]:
    """Run a comprehensive benchmark suite."""
    print("Running cb-lab comprehensive benchmarks...")
    benchmark = ScalabilityBenchmark(dim=16)

    # Test different scales
    request_counts = [1, 4, 8, 16, 32]
    sequence_lengths = [64, 128, 256, 512]
    prompt_lengths = [5, 10, 20, 30]

    # Run benchmarks
    scheduler_results = benchmark.benchmark_scheduler_scaling(
        request_counts=request_counts, prompt_lengths=prompt_lengths
    )

    cache_results = benchmark.benchmark_kv_cache_performance(
        sequence_lengths=sequence_lengths
    )

    attention_results = benchmark.benchmark_attention_mechanisms(
        sequence_lengths=[64, 128, 256], batch_sizes=[1, 4, 8]
    )

    # Generate plots and report
    benchmark.plot_scalability_results("benchmark_results.png")
    report = benchmark.generate_report()

    # Save report
    with open("benchmark_report.md", "w") as f:
        f.write(report)

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    print(report)

    return {
        "scheduler": scheduler_results,
        "kv_cache": cache_results,
        "attention": attention_results,
        "report": report,
    }


if __name__ == "__main__":
    results = run_comprehensive_benchmarks()
