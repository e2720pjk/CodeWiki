"""
Performance metrics framework for CodeWiki documentation generation.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for documentation generation."""

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_modules: int = 0
    leaf_modules: int = 0
    parent_modules: int = 0
    successful_modules: int = 0
    failed_modules: int = 0
    api_calls: int = 0
    total_api_time: float = 0.0
    parallel_efficiency: float = 0.0
    concurrency_limit: int = 1
    total_module_processing_time: float = 0.0

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    _token_rate_history: List[float] = field(default_factory=list)
    _token_rate_window_size: int = 5
    _last_token_update: Optional[float] = None

    def start_timing(self) -> None:
        """Start timing."""
        self.start_time = time.time()
        self._last_token_update = self.start_time

    def end_timing(self) -> None:
        """End timing and calculate duration."""
        self.end_time = time.time()

    @property
    def total_time(self) -> float:
        """Get total generation time."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def avg_time_per_module(self) -> float:
        """Get average time per module."""
        if self.total_modules == 0:
            return 0.0
        return self.total_time / self.total_modules

    @property
    def avg_api_time(self) -> float:
        """Get average API call time."""
        if self.api_calls == 0:
            return 0.0
        return self.total_api_time / self.api_calls

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        total_attempts = self.successful_modules + self.failed_modules
        if total_attempts == 0:
            return 100.0
        return (self.successful_modules / total_attempts) * 100

    def calculate_parallel_efficiency(self, sequential_time_estimate: float) -> None:
        """Calculate parallel efficiency compared to sequential processing."""
        if self.total_time == 0 or sequential_time_estimate == 0:
            self.parallel_efficiency = 0.0
            return

        # Efficiency = sequential_time / (parallel_time * concurrency_limit)
        theoretical_parallel_time = sequential_time_estimate / self.concurrency_limit
        self.parallel_efficiency = (
            theoretical_parallel_time / self.total_time if self.total_time > 0 else 0.0
        )

    def record_module_success(self) -> None:
        """Record a successful module processing."""
        self.successful_modules += 1

    def record_module_failure(self) -> None:
        """Record a failed module processing."""
        self.failed_modules += 1

    def record_api_call(self, api_time: float) -> None:
        """Record an API call with its duration."""
        self.api_calls += 1
        self.total_api_time += api_time

    def record_token_usage(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int, api_time: float
    ) -> None:
        """
        Record token usage from an API call.

        Args:
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            total_tokens: Total tokens used (prompt + completion)
            api_time: Time taken for API call in seconds
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens

        current_time = time.time()
        if self._last_token_update is not None and api_time > 0:
            instant_rate = completion_tokens / api_time

            self._token_rate_history.append(instant_rate)
            if len(self._token_rate_history) > self._token_rate_window_size:
                self._token_rate_history.pop(0)

            self._last_token_update = current_time

    def get_current_token_rate(self) -> float:
        """
        Get the current token generation rate (tokens per second).

        Returns:
            Token rate as tokens/second, or 0 if no data available
        """
        if not self._token_rate_history:
            return 0.0

        weights = list(range(1, len(self._token_rate_history) + 1))
        weighted_sum = sum(rate * weight for rate, weight in zip(self._token_rate_history, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "total_time_seconds": round(self.total_time, 2),
                "total_modules": self.total_modules,
                "leaf_modules": self.leaf_modules,
                "parent_modules": self.parent_modules,
                "concurrency_limit": self.concurrency_limit,
            },
            "performance": {
                "avg_time_per_module_seconds": round(self.avg_time_per_module, 2),
                "parallel_efficiency": round(self.parallel_efficiency, 2),
                "success_rate_percent": round(self.success_rate, 1),
            },
            "api_stats": {
                "total_api_calls": self.api_calls,
                "avg_api_time_seconds": round(self.avg_api_time, 2),
                "total_api_time_seconds": round(self.total_api_time, 2),
            },
            "token_stats": {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "avg_tokens_per_api_call": (
                    round(self.total_tokens / self.api_calls, 2) if self.api_calls > 0 else 0
                ),
                "current_token_rate_tps": round(self.get_current_token_rate(), 2),
                "avg_token_rate_tps": (
                    round(self.total_tokens / self.total_time, 2) if self.total_time > 0 else 0
                ),
            },
            "reliability": {
                "successful_modules": self.successful_modules,
                "failed_modules": self.failed_modules,
                "success_rate_percent": round(self.success_rate, 1),
            },
        }


class PerformanceTracker:
    """Global performance tracker for documentation generation."""

    def __init__(self):
        self.metrics: Optional[PerformanceMetrics] = None
        self.api_call_times: List[float] = []

    def start_tracking(
        self, total_modules: int, leaf_modules: int, concurrency_limit: int = 1
    ) -> None:
        """Start performance tracking."""
        self.metrics = PerformanceMetrics(
            total_modules=total_modules,
            leaf_modules=leaf_modules,
            parent_modules=total_modules - leaf_modules,
            concurrency_limit=concurrency_limit,
        )
        self.metrics.start_timing()
        logger.info(
            f"Started performance tracking for {total_modules} modules ({leaf_modules} leaf, {total_modules - leaf_modules} parent)"
        )

    def stop_tracking(self, sequential_time_estimate: Optional[float] = None) -> PerformanceMetrics:
        """Stop tracking and return metrics."""
        if self.metrics is None:
            raise RuntimeError("Performance tracking not started")

        self.metrics.end_timing()

        if sequential_time_estimate is not None:
            self.metrics.calculate_parallel_efficiency(sequential_time_estimate)

        logger.info(
            f"Performance tracking completed: {self.metrics.total_time:.2f}s total, "
            f"{self.metrics.successful_modules}/{self.metrics.total_modules} successful, "
            f"{self.format_tokens(self.metrics.total_tokens)} tokens used"
        )

        return self.metrics

    def record_module_processing(self, is_success: bool, processing_time: float) -> None:
        """Record module processing result."""
        if self.metrics is None:
            return

        # Record processing time
        self.metrics.total_module_processing_time += processing_time

        if is_success:
            self.metrics.record_module_success()
        else:
            self.metrics.record_module_failure()

    def record_api_call(self, api_time: float) -> None:
        """Record an API call."""
        if self.metrics is None:
            return

        self.metrics.record_api_call(api_time)
        self.api_call_times.append(api_time)

    def record_token_usage(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int, api_time: float
    ) -> None:
        """
        Record token usage from an API call.

        Args:
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            total_tokens: Total tokens used
            api_time: Time taken for API call
        """
        if self.metrics is None:
            return

        self.metrics.record_token_usage(prompt_tokens, completion_tokens, total_tokens, api_time)

    def get_current_token_rate(self) -> float:
        """
        Get the current token generation rate.

        Returns:
            Token rate as tokens/second
        """
        if self.metrics is None:
            return 0.0

        return self.metrics.get_current_token_rate()

    def get_total_tokens(self) -> int:
        """
        Get total tokens used so far.

        Returns:
            Total token count
        """
        if self.metrics is None:
            return 0

        return self.metrics.total_tokens

    @staticmethod
    def format_tokens(tokens: int) -> str:
        """
        Format token count with appropriate units.

        Args:
            tokens: Token count to format

        Returns:
            Formatted string (e.g., "1.5K", "2.3M")
        """
        if tokens < 1000:
            return f"{tokens}"
        elif tokens < 1_000_000:
            return f"{tokens / 1000:.1f}K"
        elif tokens < 1_000_000_000:
            return f"{tokens / 1_000_000:.2f}M"
        else:
            return f"{tokens / 1_000_000_000:.2f}G"

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current metrics (useful for progress reporting)."""
        return self.metrics


# Global performance tracker instance
performance_tracker = PerformanceTracker()
