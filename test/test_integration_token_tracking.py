"""
Integration tests for token tracking in documentation generation.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestTokenTrackingIntegration:
    """Integration tests for token tracking."""

    @pytest.mark.asyncio
    async def test_token_tracking_in_llm_call(self):
        """Test that token tracking works in LLM calls."""
        from codewiki.src.be.llm_services import call_llm_async_with_retry
        from codewiki.src.be.performance_metrics import performance_tracker

        config = Mock()
        config.main_model = "gpt-4"
        config.llm_base_url = "https://api.openai.com/v1"
        config.llm_api_key = "test-key"
        config.analysis_options.enable_llm_cache = False

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"

        mock_usage = Mock()
        mock_usage.prompt_tokens = 1000
        mock_usage.completion_tokens = 500
        mock_usage.total_tokens = 1500
        mock_response.usage = mock_usage

        with patch('codewiki.src.be.llm_services.client_manager') as mock_client_manager:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_manager.get_client = AsyncMock(return_value=mock_client)

            performance_tracker.start_tracking(total_modules=1, leaf_modules=1)

            result = await call_llm_async_with_retry("test prompt", config)

            assert result == "Test response"
            assert performance_tracker.get_total_tokens() == 1500
            assert performance_tracker.get_current_token_rate() > 0

    @pytest.mark.asyncio
    async def test_token_tracking_with_missing_usage(self):
        """Test token tracking handles missing usage information."""
        from codewiki.src.be.llm_services import call_llm_async_with_retry
        from codewiki.src.be.performance_metrics import performance_tracker

        config = Mock()
        config.main_model = "gpt-4"
        config.llm_base_url = "https://api.openai.com/v1"
        config.llm_api_key = "test-key"
        config.analysis_options.enable_llm_cache = False

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = None

        with patch('codewiki.src.be.llm_services.client_manager') as mock_client_manager:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_manager.get_client = AsyncMock(return_value=mock_client)

            performance_tracker.start_tracking(total_modules=1, leaf_modules=1)

            result = await call_llm_async_with_retry("test prompt", config)

            assert result == "Test response"
            assert performance_tracker.get_total_tokens() == 0


if __name__ == "__main__":
    test_class = TestTokenTrackingIntegration()

    print("Testing token tracking integration...")

    print("Note: These tests require pytest to run properly.")
    print("Use: pytest test/test_integration_token_tracking.py -v")
