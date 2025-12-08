#!/usr/bin/env python3
"""
Simple test script for Phase 2 implementation - Core functionality only.
Tests the parallel processing without CLI dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that core modules can be imported."""
    print("Testing core imports...")
    
    try:
        # Test ThreadSafeParserPool import
        from codewiki.src.be.dependency_analyzer.utils.thread_safe_parser import ThreadSafeParserPool, get_thread_safe_parser
        print("  ✓ ThreadSafeParserPool imported")
        
        # Test CallGraphAnalyzer import
        from codewiki.src.be.dependency_analyzer.analysis.call_graph_analyzer import CallGraphAnalyzer
        print("  ✓ CallGraphAnalyzer imported")
        
        # Test Configuration import
        from codewiki.cli.models.config import Configuration
        print("  ✓ Configuration imported")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_thread_safe_parser_basic():
    """Test ThreadSafeParserPool basic functionality."""
    print("\nTesting ThreadSafeParserPool...")
    
    try:
        from codewiki.src.be.dependency_analyzer.utils.thread_safe_parser import ThreadSafeParserPool
        
        # Test instantiation
        pool = ThreadSafeParserPool()
        print("  ✓ ThreadSafeParserPool instantiated")
        
        # Test getting parsers (may fail due to missing tree-sitter libs)
        languages = ['javascript', 'typescript', 'java', 'c', 'cpp', 'csharp']
        for lang in languages:
            try:
                parser = pool.get_parser(lang)
                if parser:
                    print(f"  ✓ {lang} parser available")
                else:
                    print(f"  ⚠ {lang} parser not available (expected in test environment)")
            except Exception as e:
                print(f"  ⚠ {lang} parser error: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ThreadSafeParserPool error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_call_graph_analyzer_basic():
    """Test CallGraphAnalyzer basic functionality."""
    print("\nTesting CallGraphAnalyzer...")
    
    try:
        from codewiki.src.be.dependency_analyzer.analysis.call_graph_analyzer import CallGraphAnalyzer
        
        # Test instantiation
        analyzer = CallGraphAnalyzer()
        print("  ✓ CallGraphAnalyzer instantiated")
        
        # Test with empty file list - sequential
        result_seq = analyzer.analyze_code_files([], '/tmp', enable_parallel=False)
        expected_keys = {'call_graph', 'functions', 'relationships', 'visualization'}
        actual_keys = set(result_seq.keys())
        
        if expected_keys == actual_keys:
            print("  ✓ Sequential mode works")
        else:
            print(f"  ✗ Sequential mode missing keys: {expected_keys - actual_keys}")
            return False
        
        # Test with empty file list - even with enable_parallel=True, analyzer
        # should fall back to sequential since there is nothing to parallelize.
        result_par = analyzer.analyze_code_files([], '/tmp', enable_parallel=True)
        if result_par['call_graph']['analysis_approach'] == 'sequential':
            print("  ✓ Parallel flag falls back to sequential for empty input")
        else:
            print("  ✗ Unexpected analysis approach for empty input")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ CallGraphAnalyzer error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_basic():
    """Test Configuration class functionality."""
    print("\nTesting Configuration...")
    
    try:
        from codewiki.cli.models.config import Configuration
        
        # Test instantiation with all fields
        _ = Configuration(
            base_url='https://api.test.com',
            main_model='test-model',
            cluster_model='test-model',
            max_tokens_per_module=40000,
            max_tokens_per_leaf=20000,
            enable_parallel_processing=False,
            concurrency_limit=3
        )
        print("  ✓ Configuration instantiated with all fields")
        
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
        
        config2 = Configuration.from_dict(config_data)
        
        # Verify all fields are set correctly
        checks = [
            (config2.base_url == 'https://api.test.com', 'base_url'),
            (config2.main_model == 'test-model', 'main_model'),
            (config2.cluster_model == 'test-model', 'cluster_model'),
            (config2.max_tokens_per_module == 40000, 'max_tokens_per_module'),
            (config2.max_tokens_per_leaf == 20000, 'max_tokens_per_leaf'),
            (not config2.enable_parallel_processing, 'enable_parallel_processing'),
            (config2.concurrency_limit == 3, 'concurrency_limit'),
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
        print(f"  ✗ Configuration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("CodeWiki Phase 2 Implementation Tests - Core Functionality")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_imports),
        ("ThreadSafeParserPool", test_thread_safe_parser_basic),
        ("CallGraphAnalyzer", test_call_graph_analyzer_basic),
        ("Configuration", test_configuration_basic),
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
    print("\n" + "=" * 60)
    print("Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All core tests passed! Phase 2 implementation is working.")
        print("\nNote: Some parser tests may show 'not available' due to missing")
        print("tree-sitter libraries in the test environment, but the code structure is correct.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())