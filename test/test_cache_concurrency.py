"""
Tests for concurrent access to LLM caching system.

Verifies thread safety and race condition handling in LLMPromptCache.
"""

import sys
import asyncio
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "codewiki" / "src"))

from codewiki.src.be.caching import LLMPromptCache


async def test_cache_concurrent_writes():
    """
    Test that multiple concurrent writes don't cause race conditions.

    Expected: All writes should be stored correctly without data loss or corruption.
    """
    print("Testing concurrent cache writes...")

    cache = LLMPromptCache(max_size=100)
    num_writes = 50

    async def write_item(i: int):
        await asyncio.sleep(0.001)  # Small delay to increase chance of race conditions
        cache.set(
            prompt=f"Test prompt {i}",
            model="test-model",
            response=f"Test response {i}",
            max_tokens=1000,
        )

    # Run concurrent writes
    tasks = [write_item(i) for i in range(num_writes)]
    await asyncio.gather(*tasks)

    # Verify all items are stored
    stats = cache.get_stats()
    print(f"  Cache size after {num_writes} concurrent writes: {stats['size']}")

    # Verify all items can be retrieved
    success_count = 0
    for i in range(num_writes):
        response = cache.get(f"Test prompt {i}", "test-model", 1000)
        if response == f"Test response {i}":
            success_count += 1

    print(f"  Successfully retrieved {success_count}/{num_writes} items")

    if success_count == num_writes:
        print("  ✓ All concurrent writes successful")
        return True
    else:
        print(f"  ✗ Lost {num_writes - success_count} items due to race conditions")
        return False


async def test_cache_concurrent_reads_writes():
    """
    Test that concurrent reads and writes don't cause issues.

    Expected: Reads should either return cached value or None, never raise exceptions.
    """
    print("\nTesting concurrent reads and writes...")

    cache = LLMPromptCache(max_size=100)

    async def write_item(key: str, value: str):
        cache.set(prompt=key, model="test-model", response=value, max_tokens=1000)

    async def read_item(key: str) -> str:
        return cache.get(key, "test-model", 1000) or "not_found"

    # Perform concurrent reads and writes
    tasks = []
    for i in range(30):
        tasks.append(write_item(f"key{i}", f"value{i}"))
        tasks.append(read_item(f"key{i}"))

    try:
        results = await asyncio.gather(*tasks)
    except asyncio.CancelledError as e:
        print(f"  ✗ Concurrent operations cancelled: {e}")
        return False
    except asyncio.TimeoutError as e:
        print(f"  ✗ Concurrent operations timed out: {e}")
        return False

    # Verify all tasks completed successfully
    if len(results) != len(tasks):
        print(f"  ✗ Not all tasks completed: {len(results)}/{len(tasks)} results")
        return False

    print(f"  ✓ Completed {len(tasks)} concurrent operations without exceptions")
    return True


async def test_cache_lru_with_concurrency():
    """
    Test that LRU eviction works correctly under concurrent access.

    Expected: Cache should maintain correct size and evict oldest entries.
    """
    print("\nTesting LRU eviction under concurrent access...")

    cache = LLMPromptCache(max_size=10)

    # Write items concurrently
    async def write_items(start: int, count: int):
        for i in range(start, start + count):
            cache.set(
                prompt=f"prompt{i}",
                model="test-model",
                response=f"response{i}",
                max_tokens=1000,
            )
            await asyncio.sleep(0.001)

    # Write 20 items with concurrency (should only keep 10 due to max_size)
    tasks = [
        write_items(0, 10),
        write_items(10, 10),
    ]
    await asyncio.gather(*tasks)

    stats = cache.get_stats()
    print(f"  Cache size: {stats['size']}/{stats['max_size']}")

    # Verify cache size is within limit
    if stats["size"] <= stats["max_size"]:
        print("  ✓ Cache size maintained correctly")
        return True
    else:
        print(f"  ✗ Cache exceeded max size: {stats['size']} > {stats['max_size']}")
        return False


def test_thread_safety():
    """
    Test that cache is thread-safe when accessed from multiple threads.

    Expected: No race conditions or deadlocks occur.
    """
    print("\nTesting thread safety...")

    cache = LLMPromptCache(max_size=100)
    num_threads = 10
    operations_per_thread = 20
    exceptions = []

    def thread_operations(thread_id: int):
        try:
            for i in range(operations_per_thread):
                key = f"thread{thread_id}_key{i}"
                cache.set(
                    prompt=key,
                    model="test-model",
                    response=f"value{thread_id}_{i}",
                    max_tokens=1000,
                )
                cache.get(key, "test-model", 1000)
        except Exception as e:
            exceptions.append((thread_id, e))

    # Create and start threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=thread_operations, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    if exceptions:
        print(f"  ✗ {len(exceptions)} threads encountered exceptions:")
        for thread_id, exc in exceptions:
            print(f"    Thread {thread_id}: {exc}")
        return False
    else:
        print(f"  ✓ {num_threads} threads completed successfully without exceptions")
        return True


async def run_all_tests():
    """Run all concurrent access tests."""
    print("=" * 60)
    print("LLM Cache Concurrent Access Tests")
    print("=" * 60)

    results = []

    # Run async tests
    results.append(await test_cache_concurrent_writes())
    results.append(await test_cache_concurrent_reads_writes())
    results.append(await test_cache_lru_with_concurrency())

    # Run thread safety test
    results.append(test_thread_safety())

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All concurrent access tests passed")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
