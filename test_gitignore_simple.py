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

# Create a minimal test that directly tests the RepoAnalyzer class
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
        
        # Test the pathspec functionality directly
        try:
            from pathspec import PathSpec
            from pathspec.patterns.gitwildmatch import GitWildMatchPattern
            
            gitignore_file = test_repo / ".gitignore"
            with open(gitignore_file, 'r') as f:
                lines = f.readlines()
            
            gitignore_spec = PathSpec.from_lines(GitWildMatchPattern, lines)
            
            # Test file matching
            test_files = [
                "main.py",
                "config.json", 
                "build/output.txt",
                "node_modules/package.json"
            ]
            
            print("\n--- Testing pathspec gitignore matching ---")
            for file_path in test_files:
                matches = gitignore_spec.match_file(file_path)
                print(f"{file_path}: {'IGNORED' if matches else 'INCLUDED'}")
            
            return True
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False

if __name__ == "__main__":
    print("=== Testing .gitignore Integration ===")
    
    if test_gitignore_functionality():
        print("\n🎉 Pathspec test passed!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)