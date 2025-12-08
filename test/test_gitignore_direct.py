#!/usr/bin/env python3
"""
Simple test script to verify .gitignore integration functionality.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import only the specific module we need, avoiding CLI imports
sys.path.insert(0, str(project_root / "codewiki" / "src" / "be" / "dependency_analyzer" / "analysis"))
from repo_analyzer import RepoAnalyzer

def test_gitignore_functionality():
    """Test that RepoAnalyzer respects .gitignore files when requested."""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_repo = Path(temp_dir) / "test_repo"
        test_repo.mkdir()
        
        # Create test files
        (test_repo / "main.py").write_text("print('Hello World')")
        (test_repo / "config.json").write_text('{"key": "value"}')
        (test_repo / "build").mkdir()
        (test_repo / "build" / "output.txt").write_text("build output")
        (test_repo / "node_modules").mkdir()
        (test_repo / "node_modules" / "package.json").write_text('{"name": "test"}')
        
        # Create .gitignore file
        gitignore_content = """
# Build artifacts
build/
*.log

# Dependencies
node_modules/

# Config files
config.json
"""
        (test_repo / ".gitignore").write_text(gitignore_content.strip())
        
        print("Test repository structure:")
        for root, dirs, files in os.walk(test_repo):
            level = root.replace(str(test_repo), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # Test 1: Without gitignore respect (default behavior)
        print("\n--- Test 1: Without --respect-gitignore (default) ---")
        analyzer_default = RepoAnalyzer(respect_gitignore=False, repo_path=str(test_repo))
        result_default = analyzer_default.analyze_repository_structure(str(test_repo))
        
        def count_files(tree):
            if tree["type"] == "file":
                return 1
            return sum(count_files(child) for child in tree.get("children", []))
        
        default_count = count_files(result_default["file_tree"])
        print(f"Files found without gitignore: {default_count}")
        
        # Test 2: With gitignore respect
        print("\n--- Test 2: With --respect-gitignore ---")
        analyzer_gitignore = RepoAnalyzer(respect_gitignore=True, repo_path=str(test_repo))
        result_gitignore = analyzer_gitignore.analyze_repository_structure(str(test_repo))
        
        gitignore_count = count_files(result_gitignore["file_tree"])
        print(f"Files found with gitignore: {gitignore_count}")
        
        # Verify results
        print(f"\n--- Results ---")
        print(f"Default behavior found {default_count} files")
        print(f"Gitignore behavior found {gitignore_count} files")
        
        # Should find fewer files when respecting gitignore
        if gitignore_count < default_count:
            print("✅ SUCCESS: Gitignore filtering is working!")
            print(f"   - Filtered out {default_count - gitignore_count} files")
            return True
        else:
            print("❌ FAILURE: Gitignore filtering is not working!")
            return False

def test_backward_compatibility():
    """Test that existing code still works without changes."""
    print("\n=== Testing Backward Compatibility ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_repo = Path(temp_dir) / "test_repo"
        test_repo.mkdir()
        
        # Create test files
        (test_repo / "main.py").write_text("print('Hello World')")
        (test_repo / "temp.txt").write_text("temporary")
        
        # Test old-style initialization (should still work)
        try:
            analyzer_old = RepoAnalyzer()  # No parameters
            result_old = analyzer_old.analyze_repository_structure(str(test_repo))
            print("✅ SUCCESS: Old-style RepoAnalyzer() initialization works")
            return True
        except Exception as e:
            print(f"❌ FAILURE: Old-style initialization failed: {e}")
            return False

if __name__ == "__main__":
    print("=== Testing .gitignore Integration ===")
    
    success = True
    
    # Test gitignore functionality
    if not test_gitignore_functionality():
        success = False
    
    # Test backward compatibility
    if not test_backward_compatibility():
        success = False
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)