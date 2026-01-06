"""A/B comparison tests for CodeWiki documentation."""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.ab_testing.utils import (
    generate_report_for_version,
    calculate_metrics,
    generate_comparison_report,
)


@pytest.fixture
def temp_output_dirs():
    """
    Create temporary directories for baseline and current documentation output.

    Yields:
        Tuple of (baseline_output_dir, current_output_dir)
    """
    baseline_dir = tempfile.mkdtemp(prefix="codewiki_baseline_")
    current_dir = tempfile.mkdtemp(prefix="codewiki_current_")

    yield Path(baseline_dir), Path(current_dir)

    # Cleanup
    shutil.rmtree(baseline_dir, ignore_errors=True)
    shutil.rmtree(current_dir, ignore_errors=True)


@pytest.fixture
def repo_path():
    """Get the path to the CodeWiki repository."""
    return Path(__file__).parent.parent.parent


def test_ab_comparison(temp_output_dirs, repo_path, caplog):
    """
    Perform A/B comparison between baseline version and current HEAD.

    This test:
    1. Generates documentation for baseline version (v0.1.0)
    2. Generates documentation for current version (HEAD)
    3. Calculates comparison metrics
    4. Generates comparison report
    5. Asserts on critical metrics
    """
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    baseline_output_dir, current_output_dir = temp_output_dirs

    # Get version tags from environment or use defaults
    baseline_version = os.getenv("CODEWIKI_BASELINE_VERSION", "v0.1.0")
    current_version = os.getenv("CODEWIKI_CURRENT_VERSION", "HEAD")

    print(f"\n{'=' * 60}")
    print(f"A/B Comparison Test")
    print(f"{'=' * 60}")
    print(f"Baseline Version: {baseline_version}")
    print(f"Current Version: {current_version}")
    print(f"{'=' * 60}\n")

    # Step 1: Generate baseline documentation
    print(f"Generating baseline documentation for {baseline_version}...")
    baseline_docs_dir = generate_report_for_version(
        version_tag=baseline_version,
        output_dir=baseline_output_dir,
        repo_path=repo_path,
    )
    print(f"✓ Baseline documentation generated at {baseline_docs_dir}")

    # Step 2: Generate current documentation
    print(f"\nGenerating current documentation for {current_version}...")
    current_docs_dir = generate_report_for_version(
        version_tag=current_version,
        output_dir=current_output_dir,
        repo_path=repo_path,
    )
    print(f"✓ Current documentation generated at {current_docs_dir}")

    # Step 3: Calculate metrics
    print("\nCalculating comparison metrics...")
    metrics = calculate_metrics(
        baseline_dir=baseline_docs_dir,
        current_dir=current_docs_dir,
    )

    print("\n" + "=" * 60)
    print("Comparison Metrics:")
    print("=" * 60)
    print(f"Functional Correctness: {metrics.functional_correctness}")
    print(f"File Count Delta: {metrics.file_count_delta:+d}")
    print(f"Structure Compatibility: {metrics.structure_compatibility}")
    print(f"Documentation Coverage: {metrics.documentation_coverage:.2%}")
    print(f"Content Length Delta: {metrics.content_length_delta:+,} characters")
    print(f"Markdown Validity: {metrics.markdown_validity}")
    print(f"Total Modules: {metrics.total_modules}")
    print(f"Files (Baseline): {metrics.files_baseline}")
    print(f"Files (Current): {metrics.files_current}")
    print(
        f"Generation Time (Baseline): {metrics.generation_time_baseline:.2f}s ({metrics.files_per_second_baseline:.2f} files/s)"
    )
    print(
        f"Generation Time (Current): {metrics.generation_time_current:.2f}s ({metrics.files_per_second_current:.2f} files/s)"
    )
    print(f"Time Delta: {metrics.time_delta:+.2f}s ({metrics.time_delta_percent:+.1f}%)")
    print("=" * 60 + "\n")

    # Step 4: Generate comparison report
    report = generate_comparison_report(
        metrics=metrics,
        baseline_version=baseline_version,
        current_version=current_version,
    )

    # Save report to fixtures
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    report_path = fixtures_dir / "comparison_report.md"

    with open(report_path, "w") as f:
        f.write(report)

    print(f"✓ Comparison report saved to {report_path}")
    print("\n" + report)

    # Step 5: Assert on critical metrics
    print("\n" + "=" * 60)
    print("Assertions:")
    print("=" * 60)

    # Critical: Functional correctness must pass
    assert metrics.functional_correctness, (
        "Documentation generation failed for one or both versions"
    )
    print("✓ Functional correctness: PASS")

    # Critical: Structure compatibility must pass
    assert metrics.structure_compatibility, (
        "Module tree structure is not compatible between versions"
    )
    print("✓ Structure compatibility: PASS")

    # Optional: File count should not decrease significantly
    if metrics.file_count_delta < -10:
        pytest.fail(f"File count decreased significantly: {metrics.file_count_delta} files")
    print(f"✓ File count delta ({metrics.file_count_delta:+d}) within acceptable range")

    # Optional: Documentation coverage should be reasonable
    if metrics.documentation_coverage < 0.1:
        pytest.fail(f"Documentation coverage too low: {metrics.documentation_coverage:.2%}")
    print(f"✓ Documentation coverage ({metrics.documentation_coverage:.2%}) acceptable")

    # Optional: Performance should not degrade significantly (>20% slower)
    if metrics.generation_time_baseline > 0 and metrics.generation_time_current > 0:
        if metrics.time_delta_percent > 20:
            pytest.fail(
                f"Performance degraded significantly: {metrics.time_delta_percent:.1f}% slower "
                f"(baseline: {metrics.generation_time_baseline:.2f}s, current: {metrics.generation_time_current:.2f}s)"
            )
        print(f"✓ Performance change ({metrics.time_delta_percent:+.1f}%) within acceptable range")

    # Print report summary
    print("\n" + "=" * 60)
    print("All assertions passed!")
    print("=" * 60)


def test_generate_report_invalid_version(temp_output_dirs, repo_path):
    """Test that generate_report_for_version raises error for invalid version."""
    baseline_output_dir, _ = temp_output_dirs

    with pytest.raises(ValueError, match="Version tag.*not found"):
        generate_report_for_version(
            version_tag="invalid-version-tag-xyz123",
            output_dir=baseline_output_dir,
            repo_path=repo_path,
        )


def test_calculate_metrics_with_empty_dirs(temp_output_dirs):
    """Test calculate_metrics with empty directories."""
    baseline_dir, current_dir = temp_output_dirs

    # Create minimal metadata files
    metadata = {
        "generation_info": {
            "timestamp": "2025-01-01T00:00:00",
            "main_model": "test-model",
            "generator_version": "1.0.0",
            "repo_path": "/test",
            "commit_id": "abc123",
        },
        "statistics": {"total_components": 1, "leaf_nodes": 1, "max_depth": 1},
        "files_generated": ["overview.md"],
    }

    (baseline_dir / "metadata.json").write_text(str(metadata).replace("'", '"'), encoding="utf-8")
    (current_dir / "metadata.json").write_text(str(metadata).replace("'", '"'), encoding="utf-8")

    # Create minimal module trees
    module_tree = {}

    (baseline_dir / "module_tree.json").write_text(
        str(module_tree).replace("'", '"'), encoding="utf-8"
    )
    (current_dir / "module_tree.json").write_text(
        str(module_tree).replace("'", '"'), encoding="utf-8"
    )

    metrics = calculate_metrics(baseline_dir=baseline_dir, current_dir=current_dir)

    assert metrics is not None
    assert metrics.file_count_delta == 0
    assert metrics.total_modules == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
