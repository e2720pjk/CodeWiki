"""
Performance measurement utilities for benchmarking CodeWiki operations.

Provides simple timing decorators and functions to collect and display
performance metrics for benchmarking parallel vs sequential operations.
"""

import time
from typing import Callable, Dict, List, Any, Optional
from functools import wraps
from collections import defaultdict


def timed(func: Callable) -> Callable:
    """
    Simple decorator to measure and log function execution time.
    
    Usage:
        @timed
        def my_function():
            ...
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that returns (result, elapsed_time)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        return result, elapsed_time
    return wrapper


class BenchmarkResult:
    """Stores results of a single benchmark run."""
    
    def __init__(self, name: str, elapsed_time: float, success: bool = True):
        self.name = name
        self.elapsed_time = elapsed_time
        self.success = success
    
    def __repr__(self):
        status = "✓" if self.success else "✗"
        return f"{status} {self.name}: {self.elapsed_time:.4f}s"


def benchmark_execution(
    functions: Dict[str, Callable],
    iterations: int = 1
) -> List[BenchmarkResult]:
    """
    Execute multiple functions and collect timing statistics.
    
    Each function is executed the specified number of times and
    average elapsed time is calculated.
    
    Args:
        functions: Dictionary mapping names to functions to benchmark
        iterations: Number of times to run each function (for averaging)
        
    Returns:
        List of BenchmarkResult objects
        
    Example:
        def process_sequential(files):
            ...
        
        def process_parallel(files):
            ...
            
        results = benchmark_execution({
            'sequential': lambda: process_sequential(files),
            'parallel': lambda: process_parallel(files),
        }, iterations=3)
    """
    results = []
    
    for name, func in functions.items():
        times = []
        success = True
        
        for i in range(iterations):
            try:
                start_time = time.perf_counter()
                func()
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)
            except Exception as e:
                print(f"  Error in {name} (iteration {i+1}): {e}")
                success = False
                break
        
        if times:
            avg_time = sum(times) / len(times)
            results.append(BenchmarkResult(name, avg_time, success))
        else:
            results.append(BenchmarkResult(name, 0.0, False))
    
    return results


def format_performance_report(
    results: List[BenchmarkResult],
    baseline_name: Optional[str] = None
) -> str:
    """
    Format timing results into a readable report.
    
    Args:
        results: List of BenchmarkResult objects
        baseline_name: Name of benchmark to use as baseline for comparison
        
    Returns:
        Formatted string with performance comparison
        
    Example:
        results = [
            BenchmarkResult('sequential', 5.2),
            BenchmarkResult('parallel', 2.1),
        ]
        print(format_performance_report(results, baseline_name='sequential'))
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Performance Benchmark Results")
    lines.append("=" * 60)
    
    baseline_time = None
    if baseline_name:
        for result in results:
            if result.name == baseline_name:
                baseline_time = result.elapsed_time
                break
    
    for result in results:
        status = "✓" if result.success else "✗"
        lines.append(f"{status} {result.name:20} {result.elapsed_time:.4f}s")
        
        if baseline_time and result.elapsed_time > 0:
            speedup = baseline_time / result.elapsed_time
            if result.name == baseline_name:
                lines.append(f"   (baseline)")
            elif speedup > 1:
                improvement = ((1 - (result.elapsed_time / baseline_time)) * 100)
                lines.append(f"   ({speedup:.2f}x faster, {improvement:.1f}% improvement)")
            else:
                slowdown = ((result.elapsed_time / baseline_time) - 1) * 100
                lines.append(f"   ({speedup:.2f}x slower, {slowdown:.1f}% regression)")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def compare_parallel_vs_sequential(
    sequential_func: Callable,
    parallel_func: Callable,
    *args,
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Convenience function to compare parallel vs sequential performance.
    
    Runs both functions and calculates speedup and improvement metrics.
    
    Args:
        sequential_func: Sequential version of the operation
        parallel_func: Parallel version of the operation
        *args: Arguments to pass to both functions
        iterations: Number of times to run each function
        
    Returns:
        Dictionary with comparison metrics
        
    Example:
        metrics = compare_parallel_vs_sequential(
            lambda: analyze_sequential(files),
            lambda: analyze_parallel(files),
            iterations=5
        )
        print(f"Speedup: {metrics['speedup']:.2f}x")
    """
    results = benchmark_execution({
        'sequential': lambda: sequential_func(*args),
        'parallel': lambda: parallel_func(*args),
    }, iterations=iterations)
    
    seq_time = next(r.elapsed_time for r in results if r.name == 'sequential')
    par_time = next(r.elapsed_time for r in results if r.name == 'parallel')
    
    return {
        'sequential_time': seq_time,
        'parallel_time': par_time,
        'speedup': seq_time / par_time if par_time > 0 else 0,
        'improvement_percent': ((seq_time - par_time) / seq_time * 100) if seq_time > 0 else 0,
        'results': results,
    }


if __name__ == "__main__":
    # Simple demonstration
    print("Performance utilities loaded successfully.")
    print("\nExample usage:")
    print("  results = benchmark_execution({'seq': func1, 'par': func2})")
    print("  print(format_performance_report(results, baseline_name='seq'))")
