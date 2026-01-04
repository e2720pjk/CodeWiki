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

def test_analysis_options():
    """Test AnalysisOptions instantiation and default values."""
    try:
        from codewiki.cli.models.job import AnalysisOptions

        # Test default initialization
        analysis_opts = AnalysisOptions()
        print("✅ SUCCESS: AnalysisOptions() default initialization works")

        # Verify all analysis fields exist with correct defaults
        expected_defaults = {
            'respect_gitignore': False,
            'use_joern': False,
            'max_files': 100,
            'max_entry_points': 5,
            'max_connectivity_files': 10,
            'enable_parallel_processing': True,
            'concurrency_limit': 5,
            'enable_llm_cache': True,
            'agent_retries': 3,
        }

        for field, expected_value in expected_defaults.items():
            actual_value = getattr(analysis_opts, field)
            if actual_value == expected_value:
                print(f"✅ SUCCESS: {field} defaults to {expected_value}")
            else:
                print(f"❌ FAILURE: {field} default is {actual_value}, expected {expected_value}")
                return False

        # Test with custom values
        custom_opts = AnalysisOptions(
            respect_gitignore=True,
            use_joern=True,
            max_files=200,
            max_entry_points=10,
            max_connectivity_files=20,
        )

        if custom_opts.respect_gitignore == True:
            print("✅ SUCCESS: custom respect_gitignore correctly set")
        else:
            print(f"❌ FAILURE: respect_gitignore is {custom_opts.respect_gitignore}, expected True")
            return False

        if custom_opts.max_files == 200:
            print("✅ SUCCESS: custom max_files correctly set")
        else:
            print(f"❌ FAILURE: max_files is {custom_opts.max_files}, expected 200")
            return False

        return True

    except Exception as e:
        print(f"❌ FAILURE: AnalysisOptions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_options():
    """Test GenerationOptions backward compatibility (only generation workflow fields)."""
    try:
        from codewiki.cli.models.job import GenerationOptions

        # Test default initialization (should still work)
        options_old = GenerationOptions()
        print("✅ SUCCESS: GenerationOptions() default initialization works")

        # Test with existing generation workflow parameters
        options_existing = GenerationOptions(
            create_branch=True,
            github_pages=True,
            no_cache=True,
            custom_output="custom"
        )
        print("✅ SUCCESS: GenerationOptions with generation workflow parameters works")

        # Verify values are correctly set
        if options_existing.create_branch == True:
            print("✅ SUCCESS: create_branch parameter correctly set")
        else:
            print(f"❌ FAILURE: create_branch parameter is {options_existing.create_branch}, expected True")
            return False

        if options_existing.github_pages == True:
            print("✅ SUCCESS: github_pages parameter correctly set")
        else:
            print(f"❌ FAILURE: github_pages parameter is {options_existing.github_pages}, expected True")
            return False

        # Verify default values
        options_default = GenerationOptions()
        if options_default.create_branch == False:
            print("✅ SUCCESS: create_branch defaults to False")
        else:
            print(f"❌ FAILURE: create_branch default is {options_default.create_branch}, expected False")
            return False

        # Verify analysis fields are NOT in GenerationOptions
        if not hasattr(options_default, 'respect_gitignore'):
            print("✅ SUCCESS: respect_gitignore removed from GenerationOptions")
        else:
            print(f"❌ FAILURE: respect_gitignore still exists in GenerationOptions")
            return False

        return True

    except Exception as e:
        print(f"❌ FAILURE: GenerationOptions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_backward_compatibility():
    """Test Config backward compatibility with AnalysisOptions."""
    try:
        from codewiki.src.config import Config
        from codewiki.cli.models.job import AnalysisOptions

        # Test from_cli with default AnalysisOptions
        config_old = Config.from_cli(
            repo_path="/tmp",
            output_dir="/tmp/docs",
            llm_base_url="http://localhost:8000",
            llm_api_key="test-key",
            main_model="test-model",
            cluster_model="test-cluster",
        )
        print("✅ SUCCESS: Config.from_cli() with default AnalysisOptions works")

        # Test from_cli with custom AnalysisOptions
        custom_analysis_opts = AnalysisOptions(
            respect_gitignore=True,
            max_files=200,
            max_entry_points=10,
        )
        config_new = Config.from_cli(
            repo_path="/tmp",
            output_dir="/tmp/docs",
            llm_base_url="http://localhost:8000",
            llm_api_key="test-key",
            main_model="test-model",
            cluster_model="test-cluster",
            analysis_options=custom_analysis_opts,
        )
        print("✅ SUCCESS: Config.from_cli() with custom AnalysisOptions works")

        # Verify values accessible via config.analysis_options pattern
        if config_new.analysis_options.respect_gitignore == True:
            print("✅ SUCCESS: config.analysis_options.respect_gitignore correctly set")
        else:
            print(f"❌ FAILURE: config.analysis_options.respect_gitignore is {config_new.analysis_options.respect_gitignore}, expected True")
            return False

        if config_new.analysis_options.max_files == 200:
            print("✅ SUCCESS: config.analysis_options.max_files correctly set")
        else:
            print(f"❌ FAILURE: config.analysis_options.max_files is {config_new.analysis_options.max_files}, expected 200")
            return False

        # Verify default AnalysisOptions values
        if config_old.analysis_options.respect_gitignore == False:
            print("✅ SUCCESS: config.analysis_options.respect_gitignore defaults to False")
        else:
            print(f"❌ FAILURE: config.analysis_options.respect_gitignore default is {config_old.analysis_options.respect_gitignore}, expected False")
            return False

        # Verify individual fields removed from Config
        if not hasattr(config_old, 'respect_gitignore'):
            print("✅ SUCCESS: respect_gitignore removed from Config")
        else:
            print(f"❌ FAILURE: respect_gitignore still exists in Config")
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

    # Test AnalysisOptions
    if not test_analysis_options():
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