#!/usr/bin/env python3
"""Interactive cb-lab demonstration system with real-time parameter adjustment."""

import argparse
import time
import json
from typing import Dict, Any, List, Optional
import sys

import torch

from cb_lab.core.scheduler import Scheduler
from cb_lab.core.request import Request
from cb_lab.core.kv_cache import DenseKVCache, PagedKVCache
from cb_lab.model.tiny_llm import TinyLLM
from cb_lab.monitoring.metrics import MetricsCollector
from cb_lab.exceptions import CBLabException


class InteractiveDemo:
    """Interactive demonstration system for cb-lab."""

    def __init__(self, dim: int = 16, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = TinyLLM(dim)
        self.scheduler: Optional[Scheduler] = None
        self.metrics = MetricsCollector()
        self.demo_requests: List[Request] = []
        self.step_count = 0

        # Default parameters
        self.params = {
            "max_tokens_per_step": 8,
            "prefill_chunk_size": 4,
            "block_size": 32,
            "cache_type": "dense",  # "dense" or "paged"
            "request_count": 3,
            "prompt_length_range": (5, 15),
            "max_new_tokens": 5,
            "enable_monitoring": True,
            "verbose": True,
        }

    def print_header(self):
        """Print demo header and instructions."""
        print("🚀 cb-lab Interactive Continuous Batching Demo")
        print("=" * 60)
        print("This interactive demo shows how continuous batching works.")
        print("You can adjust parameters in real-time to see their effects.\n")

    def print_current_config(self):
        """Print current configuration."""
        print("Current Configuration:")
        print(f"  Model dimension: {self.model.dim}")
        print(f"  Device: {self.device}")
        print(f"  Max tokens per step: {self.params['max_tokens_per_step']}")
        print(f"  Prefill chunk size: {self.params['prefill_chunk_size']}")
        print(f"  Cache type: {self.params['cache_type']}")
        print(f"  Block size: {self.params['block_size']}")
        print(f"  Request count: {self.params['request_count']}")
        print(f"  Prompt length range: {self.params['prompt_length_range']}")
        print(f"  Max new tokens per request: {self.params['max_new_tokens']}")
        print()

    def print_commands(self):
        """Print available commands."""
        print("Available Commands:")
        print("  r, run     - Run the demonstration")
        print("  s, step    - Run a single step")
        print("  c, config  - Configure parameters")
        print("  n, new     - Create new requests")
        print("  m, metrics - Show performance metrics")
        print("  v, verbose - Toggle verbose output")
        print("  h, help    - Show this help")
        print("  q, quit    - Exit the demo")
        print()

    def create_requests(self, count: Optional[int] = None) -> List[Request]:
        """Create test requests."""
        if count is None:
            count = self.params["request_count"]

        requests = []
        min_len, max_len = self.params["prompt_length_range"]

        for i in range(count):
            # Random prompt length within range
            prompt_len = torch.randint(min_len, max_len + 1, (1,)).item()
            prompt = torch.randn(prompt_len, self.model.dim, device=self.device)

            # Create cache based on type
            if self.params["cache_type"] == "paged":
                kv_cache = PagedKVCache(
                    block_size=self.params["block_size"], device=self.device
                )
            else:
                kv_cache = DenseKVCache(self.device)

            request = Request(
                req_id=f"demo_req_{i}",
                prompt=prompt,
                max_new_tokens=self.params["max_new_tokens"],
                kv_cache=kv_cache,
            )
            requests.append(request)

        return requests

    def run_demo(self):
        """Run the complete demonstration."""
        print("🔄 Running complete demonstration...\n")

        # Create new scheduler and requests
        self.scheduler = Scheduler(
            self.model,
            max_tokens_per_step=self.params["max_tokens_per_step"],
            prefill_chunk_size=self.params["prefill_chunk_size"],
        )

        self.demo_requests = self.create_requests()
        self.metrics.reset()

        # Add requests to scheduler and start tracking
        for req in self.demo_requests:
            self.scheduler.add_request(req)
            if self.params["enable_monitoring"]:
                self.metrics.start_request_tracking(req.req_id)

        self.step_count = 0

        # Run until all requests complete
        while self.scheduler.active:
            if self.params["verbose"]:
                input("Press Enter to continue to next step (or 'q' to quit)...")

            step_stats = self.scheduler.step()

            if self.params["enable_monitoring"]:
                self.metrics.record_step(**step_stats)

            self.step_count += 1

            if self.step_count > 50:  # Safety limit
                print("⚠️  Step limit reached, stopping demo")
                break

        print(f"\n✅ Demo completed in {self.step_count} steps")
        self.show_final_metrics()

    def run_single_step(self):
        """Run a single scheduler step."""
        if not self.scheduler:
            print("❌ No active scheduler. Run 'new' or 'run' first.")
            return

        if not self.scheduler.active:
            print("✅ All requests completed!")
            return

        step_stats = self.scheduler.step()
        self.step_count += 1

        if self.params["enable_monitoring"]:
            self.metrics.record_step(**step_stats)

        print(f"Step {self.step_count} completed:")
        print(f"  Decode tokens: {step_stats['decode_tokens']}")
        print(f"  Prefill tokens: {step_stats['prefill_tokens']}")
        print(f"  Active requests: {step_stats['active_requests']}")
        print(f"  Finished requests: {step_stats['finished_requests']}")
        print(f"  Step duration: {step_stats['step_duration']:.4f}s")

    def configure_parameters(self):
        """Interactive parameter configuration."""
        print("⚙️  Parameter Configuration")
        print("Enter new values (press Enter to keep current):\n")

        # Max tokens per step
        new_val = input(
            f"Max tokens per step [{self.params['max_tokens_per_step']}]: "
        ).strip()
        if new_val:
            try:
                self.params["max_tokens_per_step"] = int(new_val)
            except ValueError:
                print("⚠️  Invalid value, keeping current")

        # Prefill chunk size
        new_val = input(
            f"Prefill chunk size [{self.params['prefill_chunk_size']}]: "
        ).strip()
        if new_val:
            try:
                self.params["prefill_chunk_size"] = int(new_val)
            except ValueError:
                print("⚠️  Invalid value, keeping current")

        # Cache type
        new_val = (
            input(f"Cache type (dense/paged) [{self.params['cache_type']}]: ")
            .strip()
            .lower()
        )
        if new_val in ["dense", "paged"]:
            self.params["cache_type"] = new_val

        # Block size (for paged cache)
        if self.params["cache_type"] == "paged":
            new_val = input(f"Block size [{self.params['block_size']}]: ").strip()
            if new_val:
                try:
                    self.params["block_size"] = int(new_val)
                except ValueError:
                    print("⚠️  Invalid value, keeping current")

        # Request count
        new_val = input(f"Request count [{self.params['request_count']}]: ").strip()
        if new_val:
            try:
                self.params["request_count"] = int(new_val)
            except ValueError:
                print("⚠️  Invalid value, keeping current")

        # Prompt length range
        new_val = input(
            f"Prompt length range (min,max) [{self.params['prompt_length_range']}]: "
        ).strip()
        if new_val:
            try:
                min_len, max_len = map(int, new_val.split(","))
                self.params["prompt_length_range"] = (min_len, max_len)
            except ValueError:
                print("⚠️  Invalid format, keeping current")

        # Max new tokens
        new_val = input(
            f"Max new tokens per request [{self.params['max_new_tokens']}]: "
        ).strip()
        if new_val:
            try:
                self.params["max_new_tokens"] = int(new_val)
            except ValueError:
                print("⚠️  Invalid value, keeping current")

        print("\n✅ Configuration updated!")
        self.print_current_config()

    def create_new_requests(self):
        """Create a new scheduler and requests."""
        print("🆕 Creating new scheduler and requests...\n")

        self.scheduler = Scheduler(
            self.model,
            max_tokens_per_step=self.params["max_tokens_per_step"],
            prefill_chunk_size=self.params["prefill_chunk_size"],
        )

        self.demo_requests = self.create_requests()
        self.metrics.reset()
        self.step_count = 0

        # Add requests to scheduler
        for req in self.demo_requests:
            self.scheduler.add_request(req)
            if self.params["enable_monitoring"]:
                self.metrics.start_request_tracking(req.req_id)

        print(f"Created {len(self.demo_requests)} new requests:")
        for req in self.demo_requests:
            print(
                f"  {req.req_id}: prompt_len={req.prompt.size(0)}, max_tokens={req.max_new_tokens}"
            )
        print()

    def show_metrics(self):
        """Show current performance metrics."""
        print("📊 Performance Metrics\n")

        if not self.metrics.step_metrics:
            print("No metrics available. Run some steps first.")
            return

        # Step summary
        step_summary = self.metrics.get_step_summary()
        print("Step Summary:")
        print(f"  Total steps: {step_summary['total_steps']}")
        print(f"  Total decode tokens: {step_summary['total_decode_tokens']}")
        print(f"  Total prefill tokens: {step_summary['total_prefill_tokens']}")
        print(f"  Total tokens processed: {step_summary['total_tokens_processed']}")
        print(f"  Average step duration: {step_summary['avg_step_duration']:.4f}s")
        print(f"  Tokens per second: {step_summary['tokens_per_second']:.2f}")
        if step_summary["latest_gpu_memory_mb"] > 0:
            print(f"  GPU memory usage: {step_summary['latest_gpu_memory_mb']:.2f}MB")
        print()

        # Request summary
        if self.demo_requests:
            print("Request Status:")
            for req in self.demo_requests:
                metrics = self.metrics.get_request_summary(req.req_id)
                if metrics:
                    print(
                        f"  {req.req_id}: {metrics['status']}, "
                        f"{metrics['total_tokens']} tokens, "
                        f"{metrics['tokens_per_second']:.2f} tok/s"
                    )
                else:
                    status = "completed" if req.finished else "active"
                    progress = f"{len(req.generated_tokens)}/{req.max_new_tokens}"
                    print(f"  {req.req_id}: {status}, progress: {progress}")
            print()

    def toggle_verbose(self):
        """Toggle verbose output."""
        self.params["verbose"] = not self.params["verbose"]
        status = "enabled" if self.params["verbose"] else "disabled"
        print(f"🔊 Verbose output {status}")

    def save_config(self, filename: str = "demo_config.json"):
        """Save current configuration to file."""
        with open(filename, "w") as f:
            json.dump(self.params, f, indent=2)
        print(f"💾 Configuration saved to {filename}")

    def load_config(self, filename: str = "demo_config.json"):
        """Load configuration from file."""
        try:
            with open(filename, "r") as f:
                self.params.update(json.load(f))
            print(f"📁 Configuration loaded from {filename}")
            self.print_current_config()
        except FileNotFoundError:
            print(f"⚠️  Configuration file {filename} not found")
        except json.JSONDecodeError:
            print(f"⚠️  Invalid configuration file {filename}")

    def run_interactive_loop(self):
        """Run the main interactive loop."""
        self.print_header()
        self.print_current_config()
        self.print_commands()

        while True:
            try:
                command = input("\n> ").strip().lower()

                if command in ["q", "quit", "exit"]:
                    print("👋 Goodbye!")
                    break

                elif command in ["r", "run"]:
                    self.run_demo()

                elif command in ["s", "step"]:
                    self.run_single_step()

                elif command in ["c", "config", "configure"]:
                    self.configure_parameters()

                elif command in ["n", "new"]:
                    self.create_new_requests()

                elif command in ["m", "metrics", "stats"]:
                    self.show_metrics()

                elif command in ["v", "verbose"]:
                    self.toggle_verbose()

                elif command in ["h", "help", "?"]:
                    self.print_commands()

                elif command == "save":
                    self.save_config()

                elif command == "load":
                    self.load_config()

                elif command:
                    print(f"❓ Unknown command: {command}")
                    print("Type 'h' for help")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except CBLabException as e:
                print(f"❌ Error: {e}")
            except Exception as e:
                print(f"💥 Unexpected error: {e}")
                if self.params.get("verbose", False):
                    import traceback

                    traceback.print_exc()

    def show_final_metrics(self):
        """Show final metrics after demo completion."""
        print("\n" + "=" * 50)
        print("DEMO RESULTS SUMMARY")
        print("=" * 50)

        if self.metrics.step_metrics:
            step_summary = self.metrics.get_step_summary()
            print(f"Total Steps: {step_summary['total_steps']}")
            print(f"Total Tokens Processed: {step_summary['total_tokens_processed']}")
            print(
                f"Average Throughput: {step_summary['tokens_per_second']:.2f} tokens/sec"
            )
            print(f"Average Step Duration: {step_summary['avg_step_duration']:.4f}s")

            # Calculate efficiency metrics
            if self.demo_requests:
                total_possible_tokens = sum(
                    req.prompt.size(0) + req.max_new_tokens
                    for req in self.demo_requests
                )
                efficiency = (
                    step_summary["total_tokens_processed"] / total_possible_tokens
                ) * 100
                print(f"Processing Efficiency: {efficiency:.1f}%")

        print("\nCompleted Requests:")
        for req in self.demo_requests:
            if req.finished:
                print(
                    f"  ✅ {req.req_id}: generated {len(req.generated_tokens)} tokens"
                )
            else:
                print(
                    f"  ⏳ {req.req_id}: {len(req.generated_tokens)}/{req.max_new_tokens} tokens"
                )


def main():
    """Main entry point for interactive demo."""
    parser = argparse.ArgumentParser(description="cb-lab Interactive Demo")
    parser.add_argument("--dim", type=int, default=16, help="Model dimension")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--config", type=str, help="Load configuration from file")
    parser.add_argument(
        "--non-interactive", action="store_true", help="Run non-interactive demo"
    )

    args = parser.parse_args()

    # Create demo instance
    demo = InteractiveDemo(dim=args.dim, device=args.device)

    # Load config if specified
    if args.config:
        demo.load_config(args.config)

    if args.non_interactive:
        # Run non-interactive demo
        demo.run_demo()
    else:
        # Run interactive demo
        demo.run_interactive_loop()


if __name__ == "__main__":
    main()
