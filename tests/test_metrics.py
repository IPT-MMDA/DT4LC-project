"""Tests for metrics collection infrastructure.

Tests execution metrics, LLM metrics, and metrics collector.
"""

import time

from dta.dti.metrics import ExecutionMetrics, LLMMetrics, MetricsCollector, get_metrics_collector


class TestExecutionMetrics:
    """Tests for execution metrics."""

    def test_creation(self) -> None:
        """Test execution metrics creation."""
        metrics = ExecutionMetrics(
            plan_id="test-plan",
            start_time=time.time(),
            steps_total=5,
        )

        assert metrics.plan_id == "test-plan"
        assert metrics.status == "running"
        assert metrics.steps_completed == 0
        assert metrics.duration > 0


class TestLLMMetrics:
    """Tests for LLM metrics."""

    def test_creation(self) -> None:
        """Test LLM metrics creation."""
        metrics = LLMMetrics(
            provider="gemini",
            model="gemini-2.0-flash-exp",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.001,
            duration=2.5,
        )

        assert metrics.provider == "gemini"
        assert metrics.total_tokens == 150
        assert metrics.cost == 0.001


class TestMetricsCollector:
    """Tests for metrics collector."""

    def test_execution_lifecycle(self) -> None:
        """Test metrics collector execution lifecycle."""
        collector = MetricsCollector()

        # Start execution
        collector.start_execution("plan-1", steps_total=3)
        assert "plan-1" in collector.executions

        # Update progress
        collector.update_execution("plan-1", steps_completed=2)
        assert collector.executions["plan-1"].steps_completed == 2

        # Complete
        collector.complete_execution("plan-1", status="success")
        assert collector.executions["plan-1"].status == "success"

    def test_record_llm_call(self) -> None:
        """Test recording LLM calls."""
        collector = MetricsCollector()

        collector.record_llm_call(
            provider="gemini",
            model="gemini-2.0-flash-exp",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.001,
            duration=1.5,
        )

        stats = collector.get_stats()
        assert stats.total_llm_calls >= 1
        assert stats.total_llm_tokens >= 150

    def test_get_stats(self) -> None:
        """Test getting statistics."""
        collector = MetricsCollector()

        collector.start_execution("plan-1", steps_total=3)
        collector.complete_execution("plan-1", status="success")

        collector.record_llm_call(
            provider="gemini",
            model="gemini-2.0-flash-exp",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.001,
            duration=1.5,
        )

        stats = collector.get_stats()
        assert stats.total_executions == 1
        assert stats.successful_executions == 1
        assert stats.total_llm_calls == 1
        assert stats.total_llm_tokens == 150

    def test_global_metrics_collector(self) -> None:
        """Test global metrics collector."""
        collector = get_metrics_collector()
        collector.clear()

        collector.start_execution("test", 1)
        collector.complete_execution("test", "success")

        stats = collector.get_stats()
        assert stats.total_executions >= 1
