"""
LLM Response Caching System

Simple LLM response cache to avoid redundant API calls for identical prompts.
Provides LRU caching with configurable size limits.
"""

import hashlib
import logging
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMPromptCache:
    """
    Simple LLM response cache for identical prompts.

    Implements LRU (Least Recently Used) caching strategy
    with configurable size limits.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize LLM prompt cache.

        Args:
            max_size: Maximum number of cached responses
        """
        self.max_size = max_size
        self._cache: Dict[str, str] = {}
        self._access_order: list[str] = []
        self._lock = threading.RLock()

    def _generate_cache_key(self, prompt: str, model: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate cache key from prompt and parameters.

        Args:
            prompt: The prompt text
            model: Model name
            max_tokens: Maximum tokens for the request

        Returns:
            SHA256 hash as cache key
        """
        # Create normalized content for caching
        content_parts = [prompt.strip(), model]
        if max_tokens is not None:
            content_parts.append(str(max_tokens))

        content = "|".join(content_parts)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, prompt: str, model: str, max_tokens: Optional[int] = None) -> Optional[str]:
        """
        Get cached response if available.

        Args:
            prompt: The prompt text
            model: Model name
            max_tokens: Maximum tokens for the request

        Returns:
            Cached response or None if not found
        """
        key = self._generate_cache_key(prompt, model, max_tokens)

        with self._lock:
            if key in self._cache:
                # Move to end (LRU update)
                self._update_access_order(key)
                logger.debug(f"Cache hit for prompt: {key[:16]}...")
                return self._cache[key]

        return None

    def set(self, prompt: str, model: str, response: str, max_tokens: Optional[int] = None) -> None:
        """
        Cache response for prompt.

        Args:
            prompt: The prompt text
            model: Model name
            response: The response to cache
            max_tokens: Maximum tokens for the request
        """
        key = self._generate_cache_key(prompt, model, max_tokens)

        with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
                logger.debug(f"Evicted cache entry: {oldest_key[:16]}...")

            self._cache[key] = response
            self._update_access_order(key)

        logger.debug(f"Cached response for prompt: {key[:16]}...")

    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
        logger.debug("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0,
                "keys": list(self._cache.keys()),
            }


# Global cache instance
llm_cache = LLMPromptCache()


def get_llm_cache() -> LLMPromptCache:
    """
    Get the global LLM cache instance.

    Returns:
        Global LLMPromptCache instance
    """
    return llm_cache


def cache_llm_response(
    prompt: str, model: str, response: str, max_tokens: Optional[int] = None
) -> None:
    """
    Cache an LLM response.

    Args:
        prompt: The prompt text
        model: Model name
        response: The response to cache
        max_tokens: Maximum tokens for the request
    """
    llm_cache.set(prompt, model, response, max_tokens)


def get_cached_llm_response(
    prompt: str, model: str, max_tokens: Optional[int] = None
) -> Optional[str]:
    """
    Get cached LLM response.

    Args:
        prompt: The prompt text
        model: Model name
        max_tokens: Maximum tokens for the request

    Returns:
        Cached response or None if not found
    """
    return llm_cache.get(prompt, model, max_tokens)


def clear_llm_cache() -> None:
    """Clear the global LLM cache."""
    llm_cache.clear()
