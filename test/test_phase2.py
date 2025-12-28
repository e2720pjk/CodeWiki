#!/usr/bin/env python3
"""
Simple test script for Phase 2 implementation.
Tests the core functionality without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_thread_safe_parser():
    """
    Test ThreadSafeParserPool functionality.
    
    Validates that parsers can be retrieved for multiple languages.
    Note: Some parsers may not be available if their language grammars
    are not installed, so we expect at least 50% success rate.
    """
    print("Testing ThreadSafeParserPool...")
    
    try:
        from codewiki.src.be.dependency_analyzer.utils.thread_safe_parser import get_thread_safe_parser
        
        # Test getting parsers for different languages
        languages = ['javascript', 'typescript', 'java', 'c', 'cpp', 'csharp']
        results = {}
        
        for lang in languages:
            parser = get_thread_safe_parser(lang)
            results[lang] = parser is not None
            print(f"  {lang}: {'✓' if parser else '✗'}")
        
        success_rate = sum(results.values()) / len(results)
        print(f"  Success rate: {success_rate:.1%} ({sum(results.values())}/{len(results)})")
        
        return success_rate > 0.5  # At least half should work
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_call_graph_analyzer():
    """
    Test CallGraphAnalyzer parallel implementation.
    
    Validates:
    - CallGraphAnalyzer can be instantiated
    - Empty file list handling works correctly
    - Sequential fallback is triggered when appropriate
    - Parallel mode falls back to sequential for empty inputs
    """
    print("\nTesting CallGraphAnalyzer...")
    
    try:
        from codewiki.src.be.dependency_analyzer.analysis.call_graph_analyzer import CallGraphAnalyzer
        
        analyzer = CallGraphAnalyzer()
        print("  ✓ CallGraphAnalyzer instantiated")
        
        expected_keys = {'call_graph', 'functions', 'relationships', 'visualization'}
        
        # Test cases: (enable_parallel, expected_approach, test_name)
        test_cases = [
            (True, 'sequential', 'Parallel flag falls back to sequential for empty input'),
            (False, 'sequential', 'Sequential fallback test passed'),
        ]
        
        for enable_parallel, expected_approach, test_name in test_cases:
            result = analyzer.analyze_code_files([], '/tmp', enable_parallel=enable_parallel)
            actual_keys = set(result.keys())
            
            if not expected_keys.issubset(actual_keys):
                print(f"  ✗ Missing keys: {set(expected_keys) - actual_keys}")
                return False
            
            actual_approach = result['call_graph']['analysis_approach']
            if actual_approach == expected_approach:
                print(f"  ✓ {test_name}")
            else:
                print(f"  ✗ {test_name} (got '{actual_approach}' instead of '{expected_approach}')")
                return False
        
        # Validate result structure for empty file list
        result = analyzer.analyze_code_files([], '/tmp', enable_parallel=True)
        if expected_keys.issubset(set(result.keys())):
            print("  ✓ Empty file list test passed")
        else:
            print(f"  ✗ Missing keys: {set(expected_keys) - set(result.keys())}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """
    Test configuration compatibility.
    
    Validates that Configuration.from_dict() correctly handles all fields including:
    - API settings (base_url, main_model, cluster_model)
    - Token limits (max_tokens_per_module, max_tokens_per_leaf)
    - Parallel processing settings (enable_parallel_processing, concurrency_limit)
    """
    print("\nTesting Configuration...")
    
    try:
        from codewiki.cli.models.config import Configuration
        
        # Test from_dict with new fields
        config_data = {
            'base_url': 'https://api.test.com',
            'main_model': 'test-model',
            'cluster_model': 'test-model',
            'max_tokens_per_module': 40000,
            'max_tokens_per_leaf': 20000,
            'enable_parallel_processing': False,
            'concurrency_limit': 3
        }
        
        config = Configuration.from_dict(config_data)
        
        # Verify all fields are set correctly
        checks = [
            (config.base_url == 'https://api.test.com', 'base_url'),
            (config.main_model == 'test-model', 'main_model'),
            (config.cluster_model == 'test-model', 'cluster_model'),
            (config.max_tokens_per_module == 40000, 'max_tokens_per_module'),
            (config.max_tokens_per_leaf == 20000, 'max_tokens_per_leaf'),
            (not config.enable_parallel_processing, 'enable_parallel_processing'),
            (config.concurrency_limit == 3, 'concurrency_limit'),
        ]
        
        all_passed = True
        for check, field_name in checks:
            if check:
                print(f"  ✓ {field_name}")
            else:
                print(f"  ✗ {field_name}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("CodeWiki Phase 2 Implementation Tests")
    print("=" * 40)
    
    tests = [
        ("ThreadSafeParserPool", test_thread_safe_parser),
        ("CallGraphAnalyzer", test_call_graph_analyzer),
        ("Configuration", test_configuration),
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
    print("\n" + "=" * 40)
    print("Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All tests passed! Phase 2 implementation is working.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())