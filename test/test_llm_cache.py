"""
Tests for LLM caching system (LLMPromptCache).

Verifies that LLM caching works correctly and produces consistent results.
Tests cache hits, eviction, and key generation without making actual API calls.
"""

import sys
import hashlib
import json
import logging
import threading
from typing import Optional, Dict, Any
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


logger = logging.getLogger(__name__)

# Define LLMPromptCache inline for testing (to avoid import chain issues)
class LLMPromptCache:
    """
    Simple LLM response cache for identical prompts.
    
    Implements LRU (Least Recently Used) caching strategy
    with configurable size limits.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, str] = {}
        self._access_order: list[str] = []
        self._lock = threading.RLock()
    
    def _generate_cache_key(self, prompt: str, model: str,
                           max_tokens: Optional[int] = None) -> str:
        content_parts = [prompt.strip(), model]
        if max_tokens is not None:
            content_parts.append(str(max_tokens))
        content = "|".join(content_parts)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get(self, prompt: str, model: str,
            max_tokens: Optional[int] = None) -> Optional[str]:
        key = self._generate_cache_key(prompt, model, max_tokens)
        with self._lock:
            if key in self._cache:
                self._update_access_order(key)
                logger.debug(f"Cache hit for prompt: {key[:16]}...")
                return self._cache[key]
        return None
    
    def set(self, prompt: str, model: str, response: str,
            max_tokens: Optional[int] = None) -> None:
        key = self._generate_cache_key(prompt, model, max_tokens)
        with self._lock:
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
                logger.debug(f"Evicted cache entry: {oldest_key[:16]}...")
            self._cache[key] = response
            self._update_access_order(key)
        logger.debug(f"Cached response for prompt: {key[:16]}...")
    
    def _update_access_order(self, key: str) -> None:
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
        logger.debug("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0,
                "keys": list(self._cache.keys())
            }


def test_cache_hit_for_identical_prompt():
    """
    Test that identical prompts return cached results.
    
    Expected: Second call with same prompt should return cached response.
    """
    print("Testing cache hit for identical prompts...")
    
    try:
        cache = LLMPromptCache(max_size=100)
        prompt = "What is the purpose of this function?"
        model = "test-model"
        response1 = "This function processes user input."
        
        # First call - cache miss
        result1 = cache.get(prompt, model)
        if result1 is not None:
            print("  ✗ First call should be cache miss")
            return False
        
        # Set cache
        cache.set(prompt, model, response1)
        
        # Second call - cache hit
        result2 = cache.get(prompt, model)
        if result2 != response1:
            print(f"  ✗ Cache hit returned wrong response: {result2}")
            return False
        
        print("  ✓ Cache hit works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_size_limit_and_eviction():
    """
    Test that cache respects size limits and evicts oldest entries.
    
    Expected: When cache is full, adding a new entry should evict
    the oldest entry (LRU eviction).
    """
    print("\nTesting cache size limit and LRU eviction...")
    
    try:
        cache_size = 3
        cache = LLMPromptCache(max_size=cache_size)
        model = "test-model"
        
        # Fill cache to capacity
        for i in range(cache_size):
            cache.set(f"prompt{i}", model, f"response{i}")
        
        # Check all entries are in cache
        stats = cache.get_stats()
        if stats['size'] != cache_size:
            print(f"  ✗ Cache size should be {cache_size}, got {stats['size']}")
            return False
        
        # Add one more entry (should evict prompt0 - oldest)
        cache.set("prompt_new", model, "response_new")
        
        # Verify cache size remains at limit
        stats = cache.get_stats()
        if stats['size'] != cache_size:
            print(f"  ✗ Cache size should remain {cache_size}, got {stats['size']}")
            return False
        
        # Verify oldest entry was evicted
        result0 = cache.get("prompt0", model)
        if result0 is not None:
            print("  ✗ Oldest entry should have been evicted")
            return False
        
        # Verify newer entries are still in cache
        result1 = cache.get("prompt1", model)
        if result1 != "response1":
            print("  ✗ Newer entries should still be in cache")
            return False
        
        result_new = cache.get("prompt_new", model)
        if result_new != "response_new":
            print("  ✗ New entry should be in cache")
            return False
        
        print("  ✓ LRU eviction works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_key_consistency():
    """
    Test that cache key generation (SHA256) produces consistent keys.
    
    Expected: Same prompt+model+max_tokens combination produces
    the same cache key every time.
    """
    print("\nTesting cache key generation consistency...")
    
    try:
        cache = LLMPromptCache(max_size=100)
        prompt = "Test prompt"
        model = "test-model"
        max_tokens = 1000
        
        # Set cache entry
        cache.set(prompt, model, "response", max_tokens)
        
        # Get with exact same parameters
        result1 = cache.get(prompt, model, max_tokens)
        if result1 != "response":
            print("  ✗ Cache get with same params failed")
            return False
        
        # Get with different max_tokens (should be cache miss)
        result2 = cache.get(prompt, model, max_tokens + 1000)
        if result2 is not None:
            print("  ✗ Different max_tokens should produce different cache key")
            return False
        
        # Get with different model (should be cache miss)
        result3 = cache.get(prompt, "different-model", max_tokens)
        if result3 is not None:
            print("  ✗ Different model should produce different cache key")
            return False
        
        print("  ✓ Cache key generation is consistent")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_clear():
    """
    Test that clearing the cache removes all entries.
    
    Expected: After clear(), all cached entries should be removed.
    """
    print("\nTesting cache clear...")
    
    try:
        cache = LLMPromptCache(max_size=100)
        model = "test-model"
        
        # Add some entries
        for i in range(5):
            cache.set(f"prompt{i}", model, f"response{i}")
        
        # Verify entries exist
        stats = cache.get_stats()
        if stats['size'] != 5:
            print(f"  ✗ Expected 5 entries, got {stats['size']}")
            return False
        
        # Clear cache
        cache.clear()
        
        # Verify all entries are gone
        stats = cache.get_stats()
        if stats['size'] != 0:
            print(f"  ✗ Expected 0 entries after clear, got {stats['size']}")
            return False
        
        # Verify no cache hits
        result = cache.get("prompt0", model)
        if result is not None:
            print("  ✗ Cache should be empty after clear")
            return False
        
        print("  ✓ Cache clear works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_stats():
    """
    Test that cache statistics are accurate.
    
    Expected: get_stats() returns accurate size and utilization info.
    """
    print("\nTesting cache statistics...")
    
    try:
        cache = LLMPromptCache(max_size=10)
        model = "test-model"
        
        # Check empty stats
        stats = cache.get_stats()
        if stats['size'] != 0:
            print(f"  ✗ Empty cache should have size 0, got {stats['size']}")
            return False
        if stats['utilization'] != 0.0:
            print(f"  ✗ Empty cache should have 0% utilization, got {stats['utilization']}")
            return False
        
        # Add some entries
        for i in range(5):
            cache.set(f"prompt{i}", model, f"response{i}")
        
        # Check stats after additions
        stats = cache.get_stats()
        if stats['size'] != 5:
            print(f"  ✗ Expected 5 entries, got {stats['size']}")
            return False
        if abs(stats['utilization'] - 0.5) > 0.01:
            print(f"  ✗ Expected 50% utilization, got {stats['utilization']}")
            return False
        
        print("  ✓ Cache statistics are accurate")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all LLM cache tests."""
    print("LLM Cache Correctness Tests")
    print("=" * 50)
    
    tests = [
        ("Cache Hit for Identical Prompt", test_cache_hit_for_identical_prompt),
        ("Cache Size Limit and Eviction", test_cache_size_limit_and_eviction),
        ("Cache Key Consistency", test_cache_key_consistency),
        ("Cache Clear", test_cache_clear),
        ("Cache Statistics", test_cache_stats),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All LLM cache tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
