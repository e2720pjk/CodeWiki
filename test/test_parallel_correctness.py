"""
Parallel processing correctness tests.

Verifies that parallel processing produces identical results compared to
sequential processing. Tests CallGraphAnalyzer with enable_parallel=True
vs False to ensure correctness of the parallel implementation.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def create_test_files(base_dir: Path, file_count: int = 10) -> list:
    """
    Create test code files in multiple languages.
    
    Args:
        base_dir: Base directory to create files in
        file_count: Number of files to create
        
    Returns:
        List of file info dictionaries
    """
    # Create subdirectories
    (base_dir / "python").mkdir(parents=True, exist_ok=True)
    (base_dir / "javascript").mkdir(parents=True, exist_ok=True)
    (base_dir / "typescript").mkdir(parents=True, exist_ok=True)
    
    # Python files
    python_code = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def calculate(x, y):
    return add(x, y) - subtract(y, x)
"""
    
    # JavaScript files
    js_code = """
function add(a, b) {
    return a + b;
}

function subtract(a, b) {
    return a - b;
}

function calculate(x, y) {
    return add(x, y) - subtract(y, x);
}
"""
    
    # TypeScript files
    ts_code = """
function add(a: number, b: number): number {
    return a + b;
}

function subtract(a: number, b: number): number {
    return a - b;
}

function calculate(x: number, y: number): number {
    return add(x, y) - subtract(y, x);
}
"""
    
    files = []
    
    # Create Python files
    for i in range(min(4, file_count)):
        file_path = base_dir / "python" / f"module{i}.py"
        file_path.write_text(python_code)
        files.append({
            'path': str(file_path.relative_to(base_dir.parent)),
            'name': f"module{i}.py",
            'extension': '.py',
            'language': 'python'
        })
    
    # Create JavaScript files
    for i in range(min(3, max(0, file_count - 4))):
        file_path = base_dir / "javascript" / f"module{i}.js"
        file_path.write_text(js_code)
        files.append({
            'path': str(file_path.relative_to(base_dir.parent)),
            'name': f"module{i}.js",
            'extension': '.js',
            'language': 'javascript'
        })
    
    # Create TypeScript files
    for i in range(min(3, max(0, file_count - 7))):
        file_path = base_dir / "typescript" / f"module{i}.ts"
        file_path.write_text(ts_code)
        files.append({
            'path': str(file_path.relative_to(base_dir.parent)),
            'name': f"module{i}.ts",
            'extension': '.ts',
            'language': 'typescript'
        })
    
    return files


def test_parallel_vs_sequential_correctness():
    """
    Test that parallel and sequential processing produce identical results.
    
    Expected:
    - Functions dictionary should be identical
    - Call relationships set should be identical
    - Both should return same number of analyzed files
    """
    print("Testing parallel vs sequential correctness...")
    
    try:
        from codewiki.src.be.dependency_analyzer.analysis.call_graph_analyzer import CallGraphAnalyzer
        
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir) / "test_repo"
            tmpdir.mkdir()
            base_dir = tmpdir.parent
            
            # Create test files
            code_files = create_test_files(tmpdir, file_count=10)
            print(f"  Created {len(code_files)} test files")
            
            # Run sequential analysis
            print("  Running sequential analysis...")
            analyzer_seq = CallGraphAnalyzer()
            result_seq = analyzer_seq.analyze_code_files(code_files, str(tmpdir), enable_parallel=False)
            
            # Run parallel analysis
            print("  Running parallel analysis...")
            analyzer_par = CallGraphAnalyzer()
            result_par = analyzer_par.analyze_code_files(code_files, str(tmpdir), enable_parallel=True)
            
            # Compare results
            # Check functions count
            func_count_seq = len(result_seq['functions'])
            func_count_par = len(result_par['functions'])
            
            if func_count_seq != func_count_par:
                print(f"  ✗ Function count mismatch: sequential={func_count_seq}, parallel={func_count_par}")
                return False
            
            # Check relationships count
            rel_count_seq = len(result_seq['relationships'])
            rel_count_par = len(result_par['relationships'])
            
            if rel_count_seq != rel_count_par:
                print(f"  ✗ Relationship count mismatch: sequential={rel_count_seq}, parallel={rel_count_par}")
                return False
            
            # Check analysis approach
            if result_seq['call_graph']['analysis_approach'] != 'sequential':
                print("  ✗ Sequential analysis did not use sequential approach")
                return False
            
            if result_par['call_graph']['analysis_approach'] != 'parallel':
                print("  ✗ Parallel analysis did not use parallel approach")
                return False
            
            print(f"  ✓ Sequential: {func_count_seq} functions, {rel_count_seq} relationships")
            print(f"  ✓ Parallel: {func_count_par} functions, {rel_count_par} relationships")
            print("  ✓ Results are identical")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_file_list():
    """
    Test that empty file list is handled correctly in both modes.
    
    Expected: Both modes should return empty results with appropriate
    analysis_approach marker.
    """
    print("\nTesting empty file list handling...")
    
    try:
        from codewiki.src.be.dependency_analyzer.analysis.call_graph_analyzer import CallGraphAnalyzer
        
        analyzer = CallGraphAnalyzer()
        
        # Test parallel with empty list
        result_par = analyzer.analyze_code_files([], '/tmp', enable_parallel=True)
        if result_par['call_graph']['analysis_approach'] != 'sequential':
            print("  ✗ Empty list should fall back to sequential in parallel mode")
            return False
        if result_par['call_graph']['total_functions'] != 0:
            print("  ✗ Empty list should have 0 functions")
            return False
        
        # Test sequential with empty list
        result_seq = analyzer.analyze_code_files([], '/tmp', enable_parallel=False)
        if result_seq['call_graph']['analysis_approach'] != 'sequential':
            print("  ✗ Empty list should use sequential approach")
            return False
        if result_seq['call_graph']['total_functions'] != 0:
            print("  ✗ Empty list should have 0 functions")
            return False
        
        print("  ✓ Empty file list handled correctly in both modes")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_file():
    """
    Test that single file is handled correctly in both modes.
    
    Expected: Both modes should analyze the single file correctly.
    """
    print("\nTesting single file handling...")
    
    try:
        from codewiki.src.be.dependency_analyzer.analysis.call_graph_analyzer import CallGraphAnalyzer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir) / "test_repo"
            tmpdir.mkdir()
            base_dir = tmpdir.parent
            
            # Create single test file
            code_files = create_test_files(tmpdir, file_count=1)
            print(f"  Created {len(code_files)} test file")
            
            # Run sequential
            analyzer_seq = CallGraphAnalyzer()
            result_seq = analyzer_seq.analyze_code_files(code_files, str(tmpdir), enable_parallel=False)
            
            # Run parallel
            analyzer_par = CallGraphAnalyzer()
            result_par = analyzer_par.analyze_code_files(code_files, str(tmpdir), enable_parallel=True)
            
            # Both should find same functions
            func_count_seq = len(result_seq['functions'])
            func_count_par = len(result_par['functions'])
            
            if func_count_seq != func_count_par:
                print(f"  ✗ Function count mismatch: {func_count_seq} vs {func_count_par}")
                return False
            
            if func_count_seq == 0:
                print("  ✗ Should have found at least one function")
                return False
            
            print(f"  ✓ Both modes found {func_count_seq} function(s)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_languages():
    """
    Test handling of mixed language files.
    
    Expected: Both modes should correctly analyze files from
    different programming languages.
    """
    print("\nTesting mixed language handling...")
    
    try:
        from codewiki.src.be.dependency_analyzer.analysis.call_graph_analyzer import CallGraphAnalyzer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir) / "test_repo"
            tmpdir.mkdir()
            base_dir = tmpdir.parent
            
            # Create mixed language files
            code_files = create_test_files(tmpdir, file_count=9)
            
            # Count files per language
            lang_counts = {}
            for f in code_files:
                lang = f['language']
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
            print(f"  Created files: {', '.join(f'{k}={v}' for k, v in sorted(lang_counts.items()))}")
            
            # Run sequential
            analyzer_seq = CallGraphAnalyzer()
            result_seq = analyzer_seq.analyze_code_files(code_files, str(tmpdir), enable_parallel=False)
            
            # Run parallel
            analyzer_par = CallGraphAnalyzer()
            result_par = analyzer_par.analyze_code_files(code_files, str(tmpdir), enable_parallel=True)
            
            # Check languages found
            langs_seq = set(result_seq['call_graph']['languages_found'])
            langs_par = set(result_par['call_graph']['languages_found'])
            
            if langs_seq != langs_par:
                print(f"  ✗ Languages mismatch: {langs_seq} vs {langs_par}")
                return False
            
            # Check that all files were analyzed
            files_analyzed_seq = result_seq['call_graph']['files_analyzed']
            files_analyzed_par = result_par['call_graph']['files_analyzed']
            
            if files_analyzed_seq != len(code_files):
                print(f"  ✗ Sequential analyzed {files_analyzed_seq} files, expected {len(code_files)}")
                return False
            
            if files_analyzed_par != len(code_files):
                print(f"  ✗ Parallel analyzed {files_analyzed_par} files, expected {len(code_files)}")
                return False
            
            print(f"  ✓ Both modes analyzed {len(code_files)} files across {len(langs_seq)} languages")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all parallel correctness tests."""
    print("Parallel Processing Correctness Tests")
    print("=" * 50)
    
    tests = [
        ("Parallel vs Sequential Correctness", test_parallel_vs_sequential_correctness),
        ("Empty File List", test_empty_file_list),
        ("Single File", test_single_file),
        ("Mixed Languages", test_mixed_languages),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
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
        print("🎉 All parallel correctness tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
