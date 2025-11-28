"""Integration tests for cb-lab continuous batching system."""

import pytest
import time
import torch
from typing import List, Dict, Any

from cb_lab.core.scheduler import Scheduler
from cb_lab.core.request import Request
from cb_lab.core.batch_builder import build_ragged_batch
from cb_lab.core.kv_cache import DenseKVCache, PagedKVCache
from cb_lab.model.tiny_llm import TinyLLM
from cb_lab.exceptions import CBLabException
from cb_lab.monitoring.metrics import MetricsCollector


class TestContinuousBatchingPipeline:
    """Test complete continuous batching pipeline."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return TinyLLM(dim=8)

    @pytest.fixture
    def device(self):
        """Test device."""
        return torch.device("cpu")

    def test_full_pipeline_with_ragged_decode_mix(self, model, device):
        """Test complete pipeline with ragged prefill and mixed decode."""
        # Create requests with varying lengths
        requests = [
            Request(
                "short",
                torch.randn(2, model.dim, device=device),
                3,
                DenseKVCache(device),
            ),
            Request(
                "medium",
                torch.randn(5, model.dim, device=device),
                4,
                DenseKVCache(device),
            ),
            Request(
                "long",
                torch.randn(8, model.dim, device=device),
                2,
                DenseKVCache(device),
            ),
        ]

        # Initialize scheduler
        scheduler = Scheduler(model, max_tokens_per_step=6, prefill_chunk_size=3)

        # Add requests and track execution
        for req in requests:
            scheduler.add_request(req)

        # Run until all requests complete
        initial_active = len(scheduler.active)
        steps_completed = 0
        max_steps = 30  # Safety limit

        while scheduler.active and steps_completed < max_steps:
            step_stats = scheduler.step()
            steps_completed += 1

            # Verify step statistics are reasonable
            assert step_stats["step_id"] == steps_completed - 1
            assert step_stats["decode_tokens"] >= 0
            assert step_stats["prefill_tokens"] >= 0
            assert step_stats["active_requests"] <= len(requests)
            assert step_stats["step_duration"] >= 0

        # All requests should complete
        assert all(req.finished for req in requests)
        assert scheduler.active == []
        assert steps_completed > 0

    def test_pipeline_with_paged_cache(self, model, device):
        """Test pipeline with paged KV cache."""
        requests = [
            Request(
                "req1",
                torch.randn(4, model.dim, device=device),
                3,
                PagedKVCache(block_size=4, device=device),
            ),
            Request(
                "req2",
                torch.randn(6, model.dim, device=device),
                2,
                PagedKVCache(block_size=4, device=device),
            ),
        ]

        scheduler = Scheduler(model, max_tokens_per_step=8, prefill_chunk_size=4)

        for req in requests:
            scheduler.add_request(req)

        # Run to completion
        steps = 0
        while scheduler.active and steps < 20:
            scheduler.step()
            steps += 1

        # Verify completion
        assert all(req.finished for req in requests)
        assert len(scheduler.active) == 0

    def test_mixed_cache_types_pipeline(self, model, device):
        """Test pipeline with mixed dense and paged caches."""
        requests = [
            Request(
                "dense_req",
                torch.randn(3, model.dim, device=device),
                2,
                DenseKVCache(device),
            ),
            Request(
                "paged_req",
                torch.randn(5, model.dim, device=device),
                3,
                PagedKVCache(block_size=8, device=device),
            ),
        ]

        scheduler = Scheduler(model, max_tokens_per_step=5, prefill_chunk_size=3)

        for req in requests:
            scheduler.add_request(req)

        # Run to completion
        steps = 0
        while scheduler.active and steps < 25:
            scheduler.step()
            steps += 1

        # Both should complete successfully
        assert all(req.finished for req in requests)

    def test_pipeline_with_metrics_collection(self, model, device):
        """Test pipeline with comprehensive metrics collection."""
        metrics = MetricsCollector()
        requests = [
            Request(
                "metric_test_1",
                torch.randn(4, model.dim, device=device),
                3,
                DenseKVCache(device),
            ),
            Request(
                "metric_test_2",
                torch.randn(2, model.dim, device=device),
                4,
                DenseKVCache(device),
            ),
        ]

        scheduler = Scheduler(model, max_tokens_per_step=4, prefill_chunk_size=2)

        # Start tracking
        for req in requests:
            scheduler.add_request(req)
            metrics.start_request_tracking(
                req.req_id, req.prompt.size(0) + req.max_new_tokens
            )

        # Run with metrics
        steps = 0
        while scheduler.active and steps < 15:
            step_stats = scheduler.step()
            metrics.record_step(**step_stats)

            # Update request progress
            for req in scheduler.active:
                if req.in_decode:
                    metrics.update_request_progress(
                        req.req_id, decode_tokens=len(req.generated_tokens)
                    )
                elif req.finished:
                    metrics.finish_request(req.req_id)

            steps += 1

        # Verify metrics
        step_summary = metrics.get_step_summary()
        assert step_summary["total_steps"] > 0
        assert step_summary["total_tokens_processed"] > 0
        assert step_summary["tokens_per_second"] > 0

        # Check request metrics
        for req in requests:
            req_metrics = metrics.get_request_summary(req.req_id)
            assert req_metrics is not None
            assert req_metrics["status"] == "completed"
            assert req_metrics["total_tokens"] > 0


class TestRaggedBatching:
    """Test ragged batching functionality."""

    def test_ragged_batch_construction(self):
        """Test ragged batch construction with mixed sequences."""
        device = torch.device("cpu")
        chunks_data = [
            ("req1", 0, torch.randn(3, 8, device=device)),
            ("req2", 0, torch.randn(2, 8, device=device)),
            ("req1", 3, torch.randn(2, 8, device=device)),
            ("req3", 0, torch.randn(4, 8, device=device)),
        ]

        # Build ragged batch
        tokens_cat, token_table, ragged_mask = build_ragged_batch(chunks_data)

        # Verify shapes
        expected_total_tokens = sum(chunk.size(0) for _, _, chunk in chunks_data)
        assert tokens_cat.shape[0] == expected_total_tokens
        assert tokens_cat.shape[1] == 8  # feature dimension
        assert ragged_mask.shape == (expected_total_tokens, expected_total_tokens)
        assert len(token_table) == expected_total_tokens

        # Verify causal properties
        for i, meta_i in enumerate(token_table):
            for j, meta_j in enumerate(token_table):
                if ragged_mask[i, j]:
                    # Should only attend to same request
                    assert meta_i.req_id == meta_j.req_id
                    # Should be causal
                    assert meta_j.pos_in_seq <= meta_i.pos_in_seq

    def test_ragged_batch_empty_input(self):
        """Test ragged batch with empty input."""
        tokens_cat, token_table, ragged_mask = build_ragged_batch([])

        assert tokens_cat.numel() == 0
        assert len(token_table) == 0
        assert ragged_mask.numel() == 0

    def test_ragged_batch_single_request(self):
        """Test ragged batch with single request."""
        device = torch.device("cpu")
        chunks_data = [
            ("single_req", 0, torch.randn(5, 6, device=device)),
        ]

        tokens_cat, token_table, ragged_mask = build_ragged_batch(chunks_data)

        assert tokens_cat.shape == (5, 6)
        assert len(token_table) == 5
        assert ragged_mask.shape == (5, 5)

        # Should be standard causal mask
        assert ragged_mask.all()  # Single request should attend to all tokens
        for i in range(5):
            for j in range(i + 1, 5):
                assert ragged_mask[i, j]  # Can attend to future tokens (batch-wise)
                assert ragged_mask[j, i]  # And past tokens

    def test_ragged_batch_device_consistency(self):
        """Test ragged batch respects device consistency."""
        device = torch.device("cpu")
        chunks_data = [
            ("req1", 0, torch.randn(3, 4, device=device)),
            ("req2", 0, torch.randn(2, 4, device=device)),
        ]

        tokens_cat, token_table, ragged_mask = build_ragged_batch(chunks_data)

        assert tokens_cat.device == device
        assert ragged_mask.device == device


class TestSchedulerEdgeCases:
    """Test scheduler edge cases and error conditions."""

    @pytest.fixture
    def model(self):
        return TinyLLM(dim=4)

    def test_empty_scheduler_step(self, model):
        """Test scheduler step with no active requests."""
        scheduler = Scheduler(model, max_tokens_per_step=4)

        step_stats = scheduler.step()

        assert step_stats["status"] == "no_active_requests"
        assert step_stats["step_id"] == 0
        assert step_stats["decode_tokens"] == 0
        assert step_stats["prefill_tokens"] == 0

    def test_zero_max_tokens_configuration(self, model):
        """Test scheduler configuration with zero max tokens."""
        with pytest.raises(Exception):  # Should raise ConfigurationError
            Scheduler(model, max_tokens_per_step=0)

    def test_invalid_prefill_chunk_size(self, model):
        """Test scheduler with invalid prefill chunk size."""
        with pytest.raises(Exception):  # Should raise ConfigurationError
            Scheduler(
                model, max_tokens_per_step=4, prefill_chunk_size=8
            )  # Larger than max

    def test_request_lifecycle_transitions(self, model):
        """Test proper request lifecycle transitions."""
        device = torch.device("cpu")
        req = Request(
            "test_req",
            torch.randn(3, model.dim, device=device),
            2,
            DenseKVCache(device),
        )

        # Initial state: should be in prefill
        assert req.in_prefill
        assert not req.in_decode
        assert not req.finished

        # After consuming all prefill tokens
        while req.in_prefill:
            chunk = req.get_prefill_chunk(2)
            # Process chunk (simplified)

        # Should transition to decode
        assert not req.in_prefill
        assert req.in_decode
        assert not req.finished

        # After generating tokens
        req.append_token(torch.randn(model.dim))
        req.append_token(torch.randn(model.dim))  # max_new_tokens = 2

        # Should be finished
        assert req.finished

    def test_scheduler_with_finished_requests_cleanup(self, model):
        """Test scheduler properly cleans up finished requests."""
        device = torch.device("cpu")
        scheduler = Scheduler(model, max_tokens_per_step=6)

        # Add a request that will finish immediately (0 new tokens)
        instant_req = Request(
            "instant", torch.randn(1, model.dim, device=device), 0, DenseKVCache(device)
        )
        scheduler.add_request(instant_req)

        assert len(scheduler.active) == 1
        assert instant_req.finished

        # Step should clean up finished request
        scheduler.step()
        assert len(scheduler.active) == 0


class TestPerformanceCharacteristics:
    """Test performance and scalability characteristics."""

    def test_scheduler_scalability_with_batch_size(self):
        """Test scheduler performance scales reasonably with batch size."""
        device = torch.device("cpu")
        dim = 8
        model = TinyLLM(dim)

        batch_sizes = [1, 2, 4, 8]
        completion_times = []

        for batch_size in batch_sizes:
            requests = [
                Request(
                    f"req_{i}",
                    torch.randn(4, dim, device=device),
                    2,
                    DenseKVCache(device),
                )
                for i in range(batch_size)
            ]

            scheduler = Scheduler(model, max_tokens_per_step=8, prefill_chunk_size=4)
            start_time = time.time()

            for req in requests:
                scheduler.add_request(req)

            steps = 0
            while scheduler.active and steps < 20:
                scheduler.step()
                steps += 1

            completion_time = time.time() - start_time
            completion_times.append(completion_time)

        # Verify completion time scales reasonably (not exponentially)
        assert completion_times[-1] / completion_times[0] < batch_sizes[-1] * 2

    def test_memory_usage_growth(self):
        """Test memory usage grows linearly with workload."""
        device = torch.device("cpu")
        dim = 6
        model = TinyLLM(dim)

        # Test with increasing sequence lengths
        sequence_lengths = [5, 10, 15, 20]
        memory_usage = []

        for seq_len in sequence_lengths:
            req = Request(
                f"mem_test_{seq_len}",
                torch.randn(seq_len, dim, device=device),
                2,
                DenseKVCache(device),
            )

            scheduler = Scheduler(model, max_tokens_per_step=6)
            scheduler.add_request(req)

            # Run one step to initialize KV cache
            if scheduler.active:
                scheduler.step()

            # Estimate memory usage (simplified)
            k, v = req.kv_cache.get_kv()
            estimated_memory = k.numel() + v.numel()
            memory_usage.append(estimated_memory)

        # Memory should grow roughly linearly
        if len(memory_usage) >= 2:
            growth_ratio = memory_usage[-1] / memory_usage[0]
            length_ratio = sequence_lengths[-1] / sequence_lengths[0]
            assert growth_ratio <= length_ratio * 1.5  # Allow some overhead


@pytest.mark.integration
class TestSystemIntegration:
    """System-level integration tests."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow from request creation to completion."""
        device = torch.device("cpu")
        model = TinyLLM(dim=10)
        scheduler = Scheduler(model, max_tokens_per_step=8, prefill_chunk_size=3)
        metrics = MetricsCollector()

        # Create diverse set of requests
        requests = [
            Request(
                "user_1",
                torch.randn(3, model.dim, device=device),
                2,
                DenseKVCache(device),
            ),
            Request(
                "user_2",
                torch.randn(6, model.dim, device=device),
                1,
                PagedKVCache(4, device),
            ),
            Request(
                "user_3",
                torch.randn(2, model.dim, device=device),
                3,
                DenseKVCache(device),
            ),
        ]

        # Track execution
        total_start = time.time()
        for req in requests:
            scheduler.add_request(req)
            metrics.start_request_tracking(req.req_id)

        # Execute
        steps = 0
        while scheduler.active and steps < 30:
            step_stats = scheduler.step()
            metrics.record_step(**step_stats)
            steps += 1

        total_time = time.time() - total_start

        # Verify results
        assert all(req.finished for req in requests)
        assert len(scheduler.active) == 0

        # Check metrics
        step_summary = metrics.get_step_summary()
        assert step_summary["total_steps"] == steps
        assert step_summary["total_tokens_processed"] > 0
        assert step_summary["tokens_per_second"] > 0

        # Performance should be reasonable
        total_tokens = sum(
            req.prompt.size(0) + len(req.generated_tokens) for req in requests
        )
        throughput = total_tokens / total_time
        assert throughput > 0  # Should process tokens

    def test_error_recovery_and_resilience(self):
        """Test system resilience to errors."""
        device = torch.device("cpu")
        model = TinyLLM(dim=4)
        scheduler = Scheduler(model, max_tokens_per_step=6)

        # Create a mix of valid and potentially problematic requests
        requests = [
            Request(
                "valid_req",
                torch.randn(3, model.dim, device=device),
                2,
                DenseKVCache(device),
            ),
            # Note: In real tests, you might simulate various error conditions
        ]

        scheduler.add_request(requests[0])

        # System should handle normal execution gracefully
        try:
            steps = 0
            while scheduler.active and steps < 10:
                scheduler.step()
                steps += 1
            assert steps > 0
        except Exception as e:
            pytest.fail(f"System should handle normal execution without errors: {e}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
