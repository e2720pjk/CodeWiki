#!/usr/bin/env python3
"""
Test refactoring: separation of AnalysisOptions from GenerationOptions
"""

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_analysis_options_separation():
    """Verify analysis and generation options are properly separated."""
    try:
        from codewiki.cli.models.job import AnalysisOptions, DocumentationJob, GenerationOptions

        job = DocumentationJob()

        # Separation of responsibilities
        assert hasattr(job, "analysis_options"), "DocumentationJob missing analysis_options"
        assert hasattr(job, "generation_options"), "DocumentationJob missing generation_options"
        print("✅ SUCCESS: DocumentationJob has both analysis_options and generation_options")

        # Correct fields exist on analysis_options
        assert hasattr(job.analysis_options, "respect_gitignore"), "analysis_options missing respect_gitignore"
        assert hasattr(job.analysis_options, "use_joern"), "analysis_options missing use_joern"
        assert hasattr(job.analysis_options, "max_files"), "analysis_options missing max_files"
        assert hasattr(job.analysis_options, "max_entry_points"), "analysis_options missing max_entry_points"
        assert hasattr(
            job.analysis_options, "max_connectivity_files"
        ), "analysis_options missing max_connectivity_files"
        assert hasattr(
            job.analysis_options, "enable_parallel_processing"
        ), "analysis_options missing enable_parallel_processing"
        assert hasattr(job.analysis_options, "concurrency_limit"), "analysis_options missing concurrency_limit"
        assert hasattr(job.analysis_options, "enable_llm_cache"), "analysis_options missing enable_llm_cache"
        assert hasattr(job.analysis_options, "agent_retries"), "analysis_options missing agent_retries"
        print("✅ SUCCESS: All analysis fields exist on analysis_options")

        # Correct fields exist on generation_options
        assert hasattr(job.generation_options, "create_branch"), "generation_options missing create_branch"
        assert hasattr(job.generation_options, "github_pages"), "generation_options missing github_pages"
        assert hasattr(job.generation_options, "no_cache"), "generation_options missing no_cache"
        assert hasattr(job.generation_options, "custom_output"), "generation_options missing custom_output"
        print("✅ SUCCESS: All generation fields exist on generation_options")

        # Fields that should NOT exist on generation_options
        assert not hasattr(
            job.generation_options, "respect_gitignore"
        ), "respect_gitignore should NOT be on generation_options"
        assert not hasattr(job.generation_options, "use_joern"), "use_joern should NOT be on generation_options"
        assert not hasattr(job.generation_options, "max_files"), "max_files should NOT be on generation_options"
        assert not hasattr(
            job.generation_options, "max_entry_points"
        ), "max_entry_points should NOT be on generation_options"
        assert not hasattr(
            job.generation_options, "max_connectivity_files"
        ), "max_connectivity_files should NOT be on generation_options"
        assert not hasattr(
            job.generation_options, "enable_parallel_processing"
        ), "enable_parallel_processing should NOT be on generation_options"
        assert not hasattr(
            job.generation_options, "concurrency_limit"
        ), "concurrency_limit should NOT be on generation_options"
        assert not hasattr(
            job.generation_options, "enable_llm_cache"
        ), "enable_llm_cache should NOT be on generation_options"
        assert not hasattr(job.generation_options, "agent_retries"), "agent_retries should NOT be on generation_options"
        print("✅ SUCCESS: Analysis fields correctly NOT on generation_options")

        # Fields that should NOT exist on analysis_options
        assert not hasattr(job.analysis_options, "create_branch"), "create_branch should NOT be on analysis_options"
        assert not hasattr(job.analysis_options, "github_pages"), "github_pages should NOT be on analysis_options"
        assert not hasattr(job.analysis_options, "no_cache"), "no_cache should NOT be on analysis_options"
        assert not hasattr(job.analysis_options, "custom_output"), "custom_output should NOT be on analysis_options"
        print("✅ SUCCESS: Generation fields correctly NOT on analysis_options")

        return True

    except Exception as e:
        print(f"❌ FAILURE: test_analysis_options_separation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_serialization_backward_compatibility():
    """Test that old serialized jobs can be deserialized with new structure."""
    try:
        from codewiki.cli.models.job import DocumentationJob

        # Create a DocumentationJob from old format (v0.1.0/v0.1.1-github)
        # Old format had only 4 fields in generation_options
        old_format_data = {
            "job_id": "test-job-id",
            "repository_path": "/tmp/test",
            "repository_name": "test",
            "output_directory": "/tmp/docs",
            "commit_hash": "abc123",
            "branch_name": None,
            "timestamp_start": "2024-01-01T00:00:00",
            "timestamp_end": None,
            "status": "pending",
            "error_message": None,
            "files_generated": [],
            "module_count": 0,
            "generation_options": {
                # Only the 4 original fields (v0.1.0/v0.1.1-github format)
                "create_branch": True,
                "github_pages": False,
                "no_cache": False,
                "custom_output": None,
                # No respect_gitignore, max_files, max_entry_points, max_connectivity_files
            },
            "llm_config": None,
            "statistics": {
                "total_files_analyzed": 0,
                "leaf_nodes": 0,
                "max_depth": 0,
                "total_tokens_used": 0,
            },
        }

        # Deserialize old format
        job = DocumentationJob.from_dict(old_format_data)
        print("✅ SUCCESS: Old format (v0.1.0/v0.1.1-github) deserializes successfully")

        # Verify generation_options has correct values
        assert job.generation_options.create_branch == True, "create_branch not correctly deserialized"
        assert job.generation_options.github_pages == False, "github_pages not correctly deserialized"
        print("✅ SUCCESS: GenerationOptions correctly deserialized from old format")

        # Verify analysis_options was created with defaults
        assert job.analysis_options is not None, "analysis_options not created"
        assert job.analysis_options.respect_gitignore == False, "respect_gitignore should default to False"
        assert job.analysis_options.max_files == 100, "max_files should default to 100"
        assert job.analysis_options.max_entry_points == 5, "max_entry_points should default to 5"
        assert job.analysis_options.max_connectivity_files == 10, "max_connectivity_files should default to 10"
        print("✅ SUCCESS: AnalysisOptions created with correct defaults from old format")

        # Serialize back to new format
        new_format_data = job.to_dict()
        print("✅ SUCCESS: Job serializes to new format")

        # Verify new format has both generation_options and analysis_options
        assert "generation_options" in new_format_data, "new format missing generation_options"
        assert "analysis_options" in new_format_data, "new format missing analysis_options"
        print("✅ SUCCESS: New format includes both generation_options and analysis_options")

        # Verify generation_options in new format only has 4 fields
        gen_opts = new_format_data["generation_options"]
        assert len(gen_opts) == 4, f"generation_options should have 4 fields, has {len(gen_opts)}"
        assert set(gen_opts.keys()) == {
            "create_branch",
            "github_pages",
            "no_cache",
            "custom_output",
        }, f"generation_options has wrong fields: {gen_opts.keys()}"
        print("✅ SUCCESS: generation_options in new format has correct fields")

        # Verify analysis_options in new format has all analysis fields
        analysis_opts = new_format_data["analysis_options"]
        assert "respect_gitignore" in analysis_opts, "analysis_options missing respect_gitignore"
        assert "max_files" in analysis_opts, "analysis_options missing max_files"
        assert "max_entry_points" in analysis_opts, "analysis_options missing max_entry_points"
        assert "max_connectivity_files" in analysis_opts, "analysis_options missing max_connectivity_files"
        print("✅ SUCCESS: analysis_options in new format has correct fields")

        # Deserialize new format to verify round-trip works
        job2 = DocumentationJob.from_dict(new_format_data)
        assert (
            job2.generation_options.create_branch == job.generation_options.create_branch
        ), "generation_options not preserved in round-trip"
        assert (
            job2.analysis_options.max_files == job.analysis_options.max_files
        ), "analysis_options not preserved in round-trip"
        print("✅ SUCCESS: Round-trip serialization/deserialization works correctly")

        return True

    except Exception as e:
        print(f"❌ FAILURE: test_serialization_backward_compatibility failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Testing Refactoring: AnalysisOptions Separation ===")

    success = True

    # Test 1: Verify separation of concerns
    if not test_analysis_options_separation():
        success = False

    # Test 2: Verify backward compatibility with old serialization format
    if not test_serialization_backward_compatibility():
        success = False

    if success:
        print("\n🎉 All refactoring tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some refactoring tests failed!")
        sys.exit(1)
