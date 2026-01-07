"""
Tests for concurrent access to LLM caching system.

Verifies async task safety and race condition handling in LLMPromptCache.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        await cache.set(
            prompt=f"Test prompt {i}",
            model="test-model",
            response=f"Test response {i}",
            max_tokens=1000,
        )

    # Run concurrent writes
    tasks = [write_item(i) for i in range(num_writes)]
    await asyncio.gather(*tasks)

    # Verify all items are stored
    stats = await cache.get_stats()
    print(f"  Cache size after {num_writes} concurrent writes: {stats['size']}")

    # Verify all items can be retrieved
    success_count = 0
    for i in range(num_writes):
        response = await cache.get(f"Test prompt {i}", "test-model", 1000)
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
        await cache.set(prompt=key, model="test-model", response=value, max_tokens=1000)

    async def read_item(key: str) -> str:
        return (await cache.get(key, "test-model", 1000)) or "not_found"

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
            await cache.set(
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

    stats = await cache.get_stats()
    print(f"  Cache size: {stats['size']}/{stats['max_size']}")

    # Verify cache size is within limit
    if stats["size"] <= stats["max_size"]:
        print("  ✓ Cache size maintained correctly")
        return True
    else:
        print(f"  ✗ Cache exceeded max size: {stats['size']} > {stats['max_size']}")
        return False


async def test_async_task_safety():
    """
    Test that cache is safe when accessed from multiple async tasks.

    Expected: No race conditions or deadlocks occur.
    """
    print("\nTesting async task safety...")

    cache = LLMPromptCache(max_size=100)
    num_tasks = 10
    operations_per_task = 20
    exceptions = []

    async def task_operations(task_id: int):
        try:
            for i in range(operations_per_task):
                key = f"task{task_id}_key{i}"
                await cache.set(
                    prompt=key,
                    model="test-model",
                    response=f"value{task_id}_{i}",
                    max_tokens=1000,
                )
                await cache.get(key, "test-model", 1000)
        except Exception as e:
            exceptions.append((task_id, e))

    # Run concurrent tasks
    tasks = [task_operations(i) for i in range(num_tasks)]
    await asyncio.gather(*tasks)

    if exceptions:
        print(f"  ✗ {len(exceptions)} tasks encountered exceptions:")
        for task_id, exc in exceptions:
            print(f"    Task {task_id}: {exc}")
        return False
    else:
        print(f"  ✓ {num_tasks} tasks completed successfully without exceptions")
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

    # Run async task safety test
    results.append(await test_async_task_safety())

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
