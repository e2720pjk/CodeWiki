"""
Tests for token tracking functionality in PerformanceMetrics and PerformanceTracker.
"""

import time
from codewiki.src.be.performance_metrics import (
    PerformanceMetrics,
    PerformanceTracker,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics class token tracking."""

    def test_initial_metrics(self):
        """Test initial metrics state."""
        metrics = PerformanceMetrics()
        assert metrics.total_tokens == 0
        assert metrics.total_prompt_tokens == 0
        assert metrics.total_completion_tokens == 0
        assert metrics.get_current_token_rate() == 0.0

    def test_record_token_usage(self):
        """Test token usage recording."""
        metrics = PerformanceMetrics()

        metrics.record_token_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            api_time=10.0
        )

        assert metrics.total_prompt_tokens == 1000
        assert metrics.total_completion_tokens == 500
        assert metrics.total_tokens == 1500

        metrics.record_token_usage(
            prompt_tokens=2000,
            completion_tokens=1000,
            total_tokens=3000,
            api_time=20.0
        )

        assert metrics.total_prompt_tokens == 3000
        assert metrics.total_completion_tokens == 1500
        assert metrics.total_tokens == 4500

    def test_token_rate_calculation(self):
        """Test token rate calculation."""
        metrics = PerformanceMetrics()
        metrics.start_timing()

        metrics.record_token_usage(
            prompt_tokens=1000,
            completion_tokens=1000,
            total_tokens=2000,
            api_time=10.0
        )

        rate1 = metrics.get_current_token_rate()
        assert rate1 > 0

        metrics.record_token_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            api_time=5.0
        )

        rate2 = metrics.get_current_token_rate()
        assert rate2 > 0

    def test_token_rate_smoothing(self):
        """Test token rate smoothing."""
        metrics = PerformanceMetrics()
        metrics.start_timing()

        metrics.record_token_usage(
            prompt_tokens=1000,
            completion_tokens=1000,
            total_tokens=2000,
            api_time=10.0
        )

        metrics.record_token_usage(
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            api_time=1.0
        )

        rate = metrics.get_current_token_rate()
        assert rate > 0

    def test_to_dict_includes_tokens(self):
        """Test that to_dict includes token statistics."""
        metrics = PerformanceMetrics()
        metrics.record_token_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            api_time=10.0
        )

        metrics_dict = metrics.to_dict()

        assert "token_stats" in metrics_dict
        assert metrics_dict["token_stats"]["total_prompt_tokens"] == 1000
        assert metrics_dict["token_stats"]["total_completion_tokens"] == 500
        assert metrics_dict["token_stats"]["total_tokens"] == 1500


class TestPerformanceTracker:
    """Test PerformanceTracker class token tracking."""

    def test_token_usage_delegation(self):
        """Test that tracker delegates token recording to metrics."""
        tracker = PerformanceTracker()
        tracker.start_tracking(total_modules=10, leaf_modules=5)

        tracker.record_token_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            api_time=10.0
        )

        metrics = tracker.get_current_metrics()
        assert metrics.total_tokens == 1500

    def test_format_tokens(self):
        """Test token formatting."""
        assert PerformanceTracker.format_tokens(500) == "500"
        assert PerformanceTracker.format_tokens(1500) == "1.5K"
        assert PerformanceTracker.format_tokens(1500000) == "1.50M"
        assert PerformanceTracker.format_tokens(1500000000) == "1.50G"

    def test_get_total_tokens_before_tracking(self):
        """Test get_total_tokens before tracking starts."""
        tracker = PerformanceTracker()
        assert tracker.get_total_tokens() == 0

    def test_get_current_token_rate_before_tracking(self):
        """Test get_current_token_rate before tracking starts."""
        tracker = PerformanceTracker()
        assert tracker.get_current_token_rate() == 0.0

    def test_get_current_token_rate_during_tracking(self):
        """Test get_current_token_rate during tracking."""
        tracker = PerformanceTracker()
        tracker.start_tracking(total_modules=10, leaf_modules=5)

        tracker.record_token_usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            api_time=5.0
        )

        rate = tracker.get_current_token_rate()
        assert 90 < rate < 110


if __name__ == "__main__":
    test_metrics = TestPerformanceMetrics()

    print("Testing PerformanceMetrics...")
    test_metrics.test_initial_metrics()
    print("✓ test_initial_metrics passed")

    test_metrics.test_record_token_usage()
    print("✓ test_record_token_usage passed")

    test_metrics.test_token_rate_calculation()
    print("✓ test_token_rate_calculation passed")

    test_metrics.test_token_rate_smoothing()
    print("✓ test_token_rate_smoothing passed")

    test_metrics.test_to_dict_includes_tokens()
    print("✓ test_to_dict_includes_tokens passed")

    test_tracker = TestPerformanceTracker()

    print("\nTesting PerformanceTracker...")
    test_tracker.test_token_usage_delegation()
    print("✓ test_token_usage_delegation passed")

    test_tracker.test_format_tokens()
    print("✓ test_format_tokens passed")

    test_tracker.test_get_total_tokens_before_tracking()
    print("✓ test_get_total_tokens_before_tracking passed")

    test_tracker.test_get_current_token_rate_before_tracking()
    print("✓ test_get_current_token_rate_before_tracking passed")

    test_tracker.test_get_current_token_rate_during_tracking()
    print("✓ test_get_current_token_rate_during_tracking passed")

    print("\n✓ All tests passed!")
