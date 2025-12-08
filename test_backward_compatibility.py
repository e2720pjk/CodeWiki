#!/usr/bin/env python3
"""
Test backward compatibility for RepoAnalyzer.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Test minimal imports to avoid dependency issues
def test_backward_compatibility():
    """Test that existing code still works without changes."""
    
    # Test 1: Old-style RepoAnalyzer initialization
    try:
        from codewiki.src.be.dependency_analyzer.analysis.repo_analyzer import RepoAnalyzer
        
        # Old-style initialization (should still work)
        analyzer_old = RepoAnalyzer()  # No parameters
        print("✅ SUCCESS: Old-style RepoAnalyzer() initialization works")
        
        # Test with include/exclude patterns (should still work)
        analyzer_patterns = RepoAnalyzer(
            include_patterns=["*.py"],
            exclude_patterns=["*.txt"]
        )
        print("✅ SUCCESS: RepoAnalyzer with patterns works")
        
        # Test new-style initialization with all parameters
        analyzer_new = RepoAnalyzer(
            include_patterns=["*.py"],
            exclude_patterns=["*.txt"],
            respect_gitignore=True,
            repo_path="/tmp"
        )
        print("✅ SUCCESS: RepoAnalyzer with new parameters works")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILURE: RepoAnalyzer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_options():
    """Test GenerationOptions backward compatibility."""
    try:
        from codewiki.cli.models.job import GenerationOptions
        
        # Test default initialization (should still work)
        options_old = GenerationOptions()
        print("✅ SUCCESS: GenerationOptions() default initialization works")
        
        # Test with existing parameters
        options_existing = GenerationOptions(
            create_branch=True,
            github_pages=True,
            no_cache=True,
            custom_output="custom"
        )
        print("✅ SUCCESS: GenerationOptions with existing parameters works")
        
        # Test with new parameter
        options_new = GenerationOptions(
            create_branch=True,
            github_pages=True,
            no_cache=True,
            custom_output="custom",
            respect_gitignore=True
        )
        print("✅ SUCCESS: GenerationOptions with respect_gitignore parameter works")
        
        # Verify default value
        if options_new.respect_gitignore == True:
            print("✅ SUCCESS: respect_gitignore parameter correctly set")
        else:
            print(f"❌ FAILURE: respect_gitignore parameter is {options_new.respect_gitignore}, expected True")
            return False
        
        # Verify default is False
        options_default = GenerationOptions()
        if options_default.respect_gitignore == False:
            print("✅ SUCCESS: respect_gitignore defaults to False")
        else:
            print(f"❌ FAILURE: respect_gitignore default is {options_default.respect_gitignore}, expected False")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ FAILURE: GenerationOptions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_backward_compatibility():
    """Test Config backward compatibility."""
    try:
        from codewiki.src.config import Config
        
        # Test old-style from_cli (should still work)
        config_old = Config.from_cli(
            repo_path="/tmp",
            output_dir="/tmp/docs",
            llm_base_url="http://localhost:8000",
            llm_api_key="test-key",
            main_model="test-model",
            cluster_model="test-cluster"
        )
        print("✅ SUCCESS: Config.from_cli() with old parameters works")
        
        # Test new-style from_cli with respect_gitignore
        config_new = Config.from_cli(
            repo_path="/tmp",
            output_dir="/tmp/docs",
            llm_base_url="http://localhost:8000",
            llm_api_key="test-key",
            main_model="test-model",
            cluster_model="test-cluster",
            respect_gitignore=True
        )
        print("✅ SUCCESS: Config.from_cli() with respect_gitignore parameter works")
        
        # Verify default value
        if config_new.respect_gitignore == True:
            print("✅ SUCCESS: Config respect_gitignore parameter correctly set")
        else:
            print(f"❌ FAILURE: Config respect_gitignore parameter is {config_new.respect_gitignore}, expected True")
            return False
        
        # Test default is False
        config_default = Config.from_cli(
            repo_path="/tmp",
            output_dir="/tmp/docs",
            llm_base_url="http://localhost:8000",
            llm_api_key="test-key",
            main_model="test-model",
            cluster_model="test-cluster"
        )
        if config_default.respect_gitignore == False:
            print("✅ SUCCESS: Config respect_gitignore defaults to False")
        else:
            print(f"❌ FAILURE: Config respect_gitignore default is {config_default.respect_gitignore}, expected False")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ FAILURE: Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Backward Compatibility ===")
    
    success = True
    
    # Test RepoAnalyzer backward compatibility
    if not test_backward_compatibility():
        success = False
    
    # Test GenerationOptions backward compatibility
    if not test_generation_options():
        success = False
    
    # Test Config backward compatibility
    if not test_config_backward_compatibility():
        success = False
    
    if success:
        print("\n🎉 All backward compatibility tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some backward compatibility tests failed!")
        sys.exit(1)