"""Metrics collection for performance monitoring.

Tracks execution times, LLM usage, and success rates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import threading
import time


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution."""

    plan_id: str
    start_time: float
    end_time: float | None = None
    status: str = "running"  # running, success, failed
    steps_completed: int = 0
    steps_total: int = 0
    error: str | None = None

    @property
    def duration(self) -> float:
        """Execution duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class LLMMetrics:
    """Metrics for LLM usage."""

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    duration: float


@dataclass
class MetricsStats:
    """Aggregated statistics."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    total_llm_calls: int = 0
    total_llm_tokens: int = 0
    total_llm_cost: float = 0.0
    llm_by_provider: dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.executions: dict[str, ExecutionMetrics] = {}
        self.llm_calls: list[LLMMetrics] = []
        self._execution_counts: defaultdict[str, int] = defaultdict(int)
        self._llm_counts: defaultdict[str, int] = defaultdict(int)

    def start_execution(self, plan_id: str, steps_total: int) -> None:
        """Record execution start.

        Args:
            plan_id: Plan identifier
            steps_total: Total number of steps
        """
        self.executions[plan_id] = ExecutionMetrics(
            plan_id=plan_id,
            start_time=time.time(),
            steps_total=steps_total,
        )

    def update_execution(self, plan_id: str, steps_completed: int) -> None:
        """Update execution progress.

        Args:
            plan_id: Plan identifier
            steps_completed: Number of steps completed
        """
        if plan_id in self.executions:
            self.executions[plan_id].steps_completed = steps_completed

    def complete_execution(self, plan_id: str, status: str = "success", error: str | None = None) -> None:
        """Record execution completion.

        Args:
            plan_id: Plan identifier
            status: Execution status (success/failed)
            error: Error message if failed
        """
        if plan_id in self.executions:
            metrics = self.executions[plan_id]
            metrics.end_time = time.time()
            metrics.status = status
            metrics.error = error

            # Update counts
            self._execution_counts["total"] += 1
            self._execution_counts[status] += 1

    def record_llm_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        duration: float,
    ) -> None:
        """Record LLM API call.

        Args:
            provider: LLM provider (gemini, ollama, etc.)
            model: Model name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            cost: Estimated cost in USD
            duration: Call duration in seconds
        """
        metrics = LLMMetrics(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            duration=duration,
        )
        self.llm_calls.append(metrics)

        # Update counts
        self._llm_counts["total"] += 1
        self._llm_counts[provider] += 1

    def get_stats(self) -> MetricsStats:
        """Get aggregated statistics.

        Returns:
            Aggregated metrics
        """
        total_exec = self._execution_counts["total"]
        successful = self._execution_counts["success"]
        failed = self._execution_counts["failed"]

        # Calculate execution stats
        completed_execs = [e for e in self.executions.values() if e.end_time is not None]
        total_time = sum(e.duration for e in completed_execs)
        avg_time = total_time / len(completed_execs) if completed_execs else 0.0

        # Calculate LLM stats
        total_tokens = sum(call.total_tokens for call in self.llm_calls)
        total_cost = sum(call.cost for call in self.llm_calls)

        # LLM by provider
        llm_by_provider: dict[str, int] = {}
        for call in self.llm_calls:
            llm_by_provider[call.provider] = llm_by_provider.get(call.provider, 0) + 1

        return MetricsStats(
            total_executions=total_exec,
            successful_executions=successful,
            failed_executions=failed,
            total_execution_time=total_time,
            avg_execution_time=avg_time,
            total_llm_calls=len(self.llm_calls),
            total_llm_tokens=total_tokens,
            total_llm_cost=total_cost,
            llm_by_provider=llm_by_provider,
        )

    def get_execution_metrics(self, plan_id: str) -> ExecutionMetrics | None:
        """Get metrics for specific execution.

        Args:
            plan_id: Plan identifier

        Returns:
            Execution metrics or None
        """
        return self.executions.get(plan_id)

    def clear(self) -> None:
        """Clear all metrics."""
        self.executions.clear()
        self.llm_calls.clear()
        self._execution_counts.clear()
        self._llm_counts.clear()


# Global metrics collector
_metrics_collector: MetricsCollector | None = None
_metrics_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector.

    Returns:
        Metrics collector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_collector_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    return _metrics_collector
