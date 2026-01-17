"""
LLM service factory for creating configured LLM clients.
"""

import asyncio
import httpx
import logging
from typing import Dict, Optional
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModelSettings
from pydantic_ai.models.fallback import FallbackModel
from openai import OpenAI, AsyncOpenAI

from codewiki.src.config import Config


# Import will be done lazily to avoid circular imports
def get_performance_tracker():
    from codewiki.src.be.performance_metrics import performance_tracker

    return performance_tracker


# Import caching system
def get_llm_cache():
    from codewiki.src.be.caching import get_llm_cache

    return get_llm_cache()


logger = logging.getLogger(__name__)


def initialize_llm_cache(config: Config):
    """
    Initialize LLM cache with configuration-based size.

    Args:
        config: Configuration object containing cache_size in analysis_options
    """
    cache_size = None
    try:
        from codewiki.src.be.caching import get_llm_cache

        cache_size = config.analysis_options.cache_size
        llm_cache = get_llm_cache()
        llm_cache.max_size = cache_size

        logger.info(f"LLM cache initialized with max_size={cache_size}")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM cache: {e}")


def create_main_model(config: Config) -> OpenAIModel:
    """Create the main LLM model from configuration."""
    return OpenAIModel(
        model_name=config.main_model,
        provider=OpenAIProvider(base_url=config.llm_base_url, api_key=config.llm_api_key),
        settings=OpenAIModelSettings(temperature=0.0, max_tokens=config.max_tokens),
    )


def create_fallback_model(config: Config) -> OpenAIModel:
    """Create the fallback LLM model from configuration."""
    return OpenAIModel(
        model_name=config.fallback_model,
        provider=OpenAIProvider(base_url=config.llm_base_url, api_key=config.llm_api_key),
        settings=OpenAIModelSettings(temperature=0.0, max_tokens=config.max_tokens),
    )


def create_fallback_models(config: Config) -> FallbackModel:
    """Create fallback models chain from configuration."""
    main = create_main_model(config)
    fallback = create_fallback_model(config)
    return FallbackModel(main, fallback)


def create_openai_client(config: Config) -> OpenAI:
    """Create OpenAI client from configuration."""
    return OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)


class ClientManager:
    """
    Manages async OpenAI clients with connection pooling.

    Singleton Pattern: Creates one AsyncOpenAI client per unique (base_url, api_key) combination
    and reuses it across multiple requests. Client reuse is critical for:
      - Connection pooling: HTTP connections are kept alive and reused
      - Thread safety: AsyncOpenAI clients are safe for concurrent asyncio task access
      - Performance: Avoids TCP/TLS handshake overhead on each request

    HTTP Backend: Uses httpx with connection pooling limits appropriate for concurrency_limit=5:
      - max_keepalive_connections=20: Keeps 20 idle connections ready for reuse
      - max_connections=100: Maximum concurrent connections (far above default concurrency)
      Note: aiohttp backend would only provide benefits at concurrency_limit > 20
    """

    def __init__(self):
        self._clients: Dict[str, AsyncOpenAI] = {}
        self._lock = asyncio.Lock()

    async def get_client(self, config: Config) -> AsyncOpenAI:
        """
        Get or create async OpenAI client with connection pooling.

        Returns a cached AsyncOpenAI client for given configuration.
        If no client exists for this config, creates a new one with optimized
        connection pooling settings.

        Thread Safety: AsyncOpenAI clients are safe for concurrent access from
        multiple asyncio tasks. The lock ensures only one client instance is
        created per unique config key, avoiding race conditions during initialization.

        Args:
            config: Configuration containing llm_base_url and llm_api_key

        Returns:
            AsyncOpenAI client instance
        """
        client_key = f"{config.llm_base_url}_{config.llm_api_key[:8]}"

        async with self._lock:
            if client_key not in self._clients:
                self._clients[client_key] = AsyncOpenAI(
                    base_url=config.llm_base_url,
                    api_key=config.llm_api_key,
                    http_client=httpx.AsyncClient(
                        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
                    ),
                )
            return self._clients[client_key]

    async def cleanup_client(self, config: Config) -> bool:
        """
        Close and remove a specific client.

        Args:
            config: Configuration containing llm_base_url and llm_api_key

        Returns:
            True if client was found and removed, False otherwise
        """
        client_key = f"{config.llm_base_url}_{config.llm_api_key[:8]}"

        async with self._lock:
            if client_key in self._clients:
                await self._clients[client_key].close()
                del self._clients[client_key]
                return True
            return False

    async def cleanup(self) -> None:
        """
        Close all HTTP clients and clear cache.

        This should be called when shutting down the application to prevent
        resource leaks. All async OpenAI clients and their underlying httpx
        AsyncClient instances will be properly closed.
        """
        async with self._lock:
            for client in self._clients.values():
                await client.close()
            self._clients.clear()

    async def cleanup_inactive(self, inactive_seconds: int = 3600) -> int:
        """
        Remove clients that haven't been used recently.

        Note: Current implementation doesn't track last access time,
        so this is a placeholder for future enhancement.

        Args:
            inactive_seconds: Minimum seconds of inactivity before cleanup

        Returns:
            Number of clients removed (always 0 in current implementation)
        """
        # TODO: Track last access time and implement actual cleanup logic
        # For now, return 0 as we don't track access times
        return 0


# Global client manager
client_manager = ClientManager()


def call_llm(
    prompt: str, config: Config, model: Optional[str] = None, temperature: float = 0.0
) -> str:
    """
    Call LLM with the given prompt (synchronous version for backward compatibility).

    Args:
        prompt: The prompt to send
        config: Configuration containing LLM settings
        model: Model name (defaults to config.main_model)
        temperature: Temperature setting

    Returns:
        LLM response text
    """
    if model is None:
        model = config.main_model

    client = create_openai_client(config)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=config.max_tokens,
    )
    return response.choices[0].message.content or ""


async def call_llm_async_with_retry(
    prompt: str,
    config: Config,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """
    Call LLM asynchronously with connection pooling and retry logic.

    Retry Strategy: Implements exponential backoff for retryable errors:
      - Network errors (connection, timeout)
      - Rate limiting (HTTP 429)
      - Server errors (HTTP 502, 503, 504)
      - Temporary failures
    Non-retryable errors (e.g., authentication, invalid request) are raised immediately.

    Thread Safety: AsyncOpenAI client from ClientManager is safe for concurrent
    access from multiple asyncio tasks. Each task gets its own coroutine context,
    but the underlying HTTP connection pool is shared efficiently.

    Args:
        prompt: The prompt to send
        config: Configuration containing LLM settings
        model: Model name (defaults to config.main_model)
        temperature: Temperature setting
        max_tokens: Maximum tokens (defaults to config value)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff (seconds)

    Returns:
        LLM response text

    Raises:
        Exception: If all retries are exhausted or error is non-retryable
    """
    if model is None:
        model = config.main_model

    if max_tokens is None:
        max_tokens = config.max_tokens

    # Check cache first
    if config.analysis_options.enable_llm_cache:
        cached_response = await get_llm_cache().get(prompt, model, max_tokens)
        if cached_response is not None:
            logger.debug(f"Cache hit for LLM prompt: {model}")
            return cached_response

    client = await client_manager.get_client(config)
    last_exception = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            start_time = asyncio.get_event_loop().time()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            api_time = asyncio.get_event_loop().time() - start_time
            response_content = response.choices[0].message.content or ""

            # Record successful API call
            get_performance_tracker().record_api_call(api_time)

            # Extract and record token usage
            if hasattr(response, "usage") and response.usage is not None:
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

                get_performance_tracker().record_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    api_time=api_time,
                )

                rate = completion_tokens / api_time if api_time > 0 else float("inf")
                logger.debug(
                    f"Token usage: prompt={prompt_tokens}, "
                    f"completion={completion_tokens}, "
                    f"total={total_tokens}, "
                    f"rate={rate:.1f} t/s"
                )
            else:
                logger.warning(f"API response missing usage information for model {model}")

            # Cache successful response
            if config.analysis_options.enable_llm_cache:
                from codewiki.src.be.caching import cache_llm_response

                await cache_llm_response(prompt, model, response_content, max_tokens)

            return response_content

        except Exception as e:
            last_exception = e

            # Check if this is a retryable error
            if not _is_retryable_error(e):
                logger.error(f"Non-retryable error in LLM call: {e}")
                raise

            if attempt < max_retries:
                # Calculate exponential backoff delay
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.2f}s: {e}"
                )

                await asyncio.sleep(delay)
            else:
                logger.error(f"LLM call failed after {max_retries + 1} attempts: {e}")

    # All retries exhausted
    if last_exception is not None:
        raise last_exception
    else:
        raise RuntimeError("All retries exhausted but no exception was captured")


def _is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception is retryable.

    Args:
        exception: The exception to check

    Returns:
        True if the exception should be retried
    """
    error_str = str(exception).lower()

    # Network-related errors
    retryable_patterns = [
        "connection",
        "timeout",
        "network",
        "rate limit",
        "too many requests",
        "temporary",
        "service unavailable",
        "internal server error",
        "503",
        "502",
        "504",
        "429",
    ]

    return any(pattern in error_str for pattern in retryable_patterns)


async def call_llm_async(
    prompt: str,
    config: Config,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Call LLM asynchronously with connection pooling.

    Args:
        prompt: The prompt to send
        config: Configuration containing LLM settings
        model: Model name (defaults to config.main_model)
        temperature: Temperature setting
        max_tokens: Maximum tokens (defaults to config value)

    Returns:
        LLM response text
    """
    return await call_llm_async_with_retry(prompt, config, model, temperature, max_tokens)
