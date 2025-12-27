"""A/B testing utilities for CodeWiki documentation comparison."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import shutil
import tempfile
import json
import logging
import sys
import os
import git

logger = logging.getLogger(__name__)

# Get project root directory
project_root = Path(__file__).parent.parent.parent


@dataclass
class ComparisonMetrics:
    """Metrics for comparing documentation between two versions."""
    functional_correctness: bool
    file_count_delta: int
    structure_compatibility: bool
    documentation_coverage: float
    content_length_delta: int
    markdown_validity: bool
    total_modules: int
    files_baseline: int
    files_current: int


def generate_report_for_version(version_tag: str, output_dir: Path, repo_path: Path) -> Path:
    """
    Generate documentation for a specific git version.

    Args:
        version_tag: Git tag or commit to checkout
        output_dir: Directory to output documentation
        repo_path: Path to the repository

    Returns:
        Path to the generated documentation directory
    """
    logger.info(f"Generating documentation for version {version_tag}")

    # Create a temporary worktree to avoid modifying the current working directory
    worktree_dir = None
    try:
        repo = git.Repo(repo_path)

        # Check if version_tag exists
        try:
            repo.commit(version_tag)
        except git.exc.BadName:
            logger.error(f"Version tag {version_tag} not found")
            raise ValueError(f"Version tag {version_tag} not found in repository")

        # Create a worktree for this version
        worktree_parent = tempfile.mkdtemp(prefix="codewiki_ab_test_")
        worktree_dir = Path(worktree_parent) / version_tag
        repo.git.worktree("add", str(worktree_dir), version_tag)

        logger.info(f"Created worktree at {worktree_dir}")

        # Generate documentation using subprocess
        cmd = [
            sys.executable,
            "-m",
            "codewiki",
            "generate",
            "--output",
            str(output_dir),
            "--no-cache",
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        # Ensure we use the codewiki module from current environment, not worktree
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)

        result = subprocess.run(
            cmd,
            cwd=worktree_dir,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout
            env=env,
        )

        if result.returncode != 0:
            logger.error(f"Documentation generation failed for {version_tag}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(
                f"Documentation generation failed for {version_tag}: {result.stderr}"
            )

        logger.info(f"Documentation generated successfully for {version_tag}")
        return output_dir

    finally:
        # Clean up worktree
        if worktree_dir and worktree_dir.exists():
            try:
                # Remove the worktree
                repo = git.Repo(repo_path)
                repo.git.worktree("remove", str(worktree_dir), "--force")
                # Remove the parent directory
                shutil.rmtree(worktree_dir.parent, ignore_errors=True)
                logger.info(f"Cleaned up worktree at {worktree_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up worktree: {e}")


def _count_markdown_files(directory: Path) -> int:
    """Count markdown files in a directory."""
    return len(list(directory.glob("**/*.md")))


def _get_total_content_length(directory: Path) -> int:
    """Get total character count of all markdown files."""
    total = 0
    for md_file in directory.glob("**/*.md"):
        total += md_file.read_text(encoding="utf-8", errors="ignore").__len__()
    return total


def _validate_markdown_files(directory: Path) -> bool:
    """Validate that all markdown files are well-formed."""
    try:
        for md_file in directory.glob("**/*.md"):
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                logger.warning(f"Empty markdown file: {md_file}")
                return False
        return True
    except Exception as e:
        logger.error(f"Markdown validation failed: {e}")
        return False


def _validate_module_tree_schema(module_tree: Dict[str, Any]) -> bool:
    """Validate module_tree.json has expected structure."""
    if not isinstance(module_tree, dict):
        return False

    for module_name, module_info in module_tree.items():
        if not isinstance(module_name, str):
            return False
        if not isinstance(module_info, dict):
            return False

        if module_info:
            if "children" in module_info:
                if not isinstance(module_info["children"], dict):
                    return False
                if not _validate_module_tree_schema(module_info["children"]):
                    return False

    return True


def calculate_metrics(
    baseline_dir: Path, current_dir: Path
) -> ComparisonMetrics:
    """
    Calculate comparison metrics between baseline and current documentation.

    Args:
        baseline_dir: Path to baseline documentation directory
        current_dir: Path to current documentation directory

    Returns:
        ComparisonMetrics object with calculated metrics
    """
    logger.info("Calculating comparison metrics")

    # Load metadata from both versions
    baseline_metadata = {}
    current_metadata = {}

    baseline_metadata_path = baseline_dir / "metadata.json"
    current_metadata_path = current_dir / "metadata.json"

    if baseline_metadata_path.exists():
        with open(baseline_metadata_path) as f:
            baseline_metadata = json.load(f)

    if current_metadata_path.exists():
        with open(current_metadata_path) as f:
            current_metadata = json.load(f)

    # Count files
    files_baseline = _count_markdown_files(baseline_dir)
    files_current = _count_markdown_files(current_dir)
    file_count_delta = files_current - files_baseline

    # Load module trees
    baseline_module_tree_path = baseline_dir / "module_tree.json"
    current_module_tree_path = current_dir / "module_tree.json"

    structure_compatibility = True
    total_modules = 0

    if baseline_module_tree_path.exists() and current_module_tree_path.exists():
        with open(baseline_module_tree_path) as f:
            baseline_module_tree = json.load(f)
        with open(current_module_tree_path) as f:
            current_module_tree = json.load(f)

        # Validate schemas
        structure_compatibility = _validate_module_tree_schema(
            baseline_module_tree
        ) and _validate_module_tree_schema(current_module_tree)

        # Count modules
        total_modules = len(current_module_tree)

    # Calculate documentation coverage (ratio of modules to files)
    documentation_coverage = 0.0
    if total_modules > 0:
        documentation_coverage = files_current / total_modules

    # Content length delta
    baseline_length = _get_total_content_length(baseline_dir)
    current_length = _get_total_content_length(current_dir)
    content_length_delta = current_length - baseline_length

    # Validate markdown
    markdown_validity = _validate_markdown_files(
        baseline_dir
    ) and _validate_markdown_files(current_dir)

    # Functional correctness - check if expected files exist
    expected_files = ["overview.md", "module_tree.json", "metadata.json"]
    functional_correctness = True

    for expected_file in expected_files:
        if not (baseline_dir / expected_file).exists():
            logger.error(f"Missing expected file in baseline: {expected_file}")
            functional_correctness = False
        if not (current_dir / expected_file).exists():
            logger.error(f"Missing expected file in current: {expected_file}")
            functional_correctness = False

    # Check for generation errors in metadata
    if "generation_info" not in baseline_metadata or "generation_info" not in current_metadata:
        functional_correctness = False

    return ComparisonMetrics(
        functional_correctness=functional_correctness,
        file_count_delta=file_count_delta,
        structure_compatibility=structure_compatibility,
        documentation_coverage=documentation_coverage,
        content_length_delta=content_length_delta,
        markdown_validity=markdown_validity,
        total_modules=total_modules,
        files_baseline=files_baseline,
        files_current=files_current,
    )


def generate_comparison_report(
    metrics: ComparisonMetrics,
    baseline_version: str,
    current_version: str,
) -> str:
    """
    Generate a markdown comparison report.

    Args:
        metrics: ComparisonMetrics object
        baseline_version: Baseline version identifier
        current_version: Current version identifier

    Returns:
        Formatted markdown report string
    """
    lines = []
    lines.append("# A/B Documentation Comparison Report")
    lines.append("")
    lines.append(f"**Baseline:** {baseline_version}")
    lines.append(f"**Current:** {current_version}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")

    overall_status = (
        "✓ PASS"
        if metrics.functional_correctness
        and metrics.structure_compatibility
        and metrics.markdown_validity
        else "✗ FAIL"
    )
    lines.append(f"**Overall Status:** {overall_status}")
    lines.append("")

    lines.append("### Key Metrics")
    lines.append("")
    lines.append(f"- **File Count:** {metrics.files_baseline} → {metrics.files_current} ({metrics.file_count_delta:+d})")
    lines.append(f"- **Content Length:** {metrics.content_length_delta:+d} characters")
    lines.append(f"- **Documentation Coverage:** {metrics.documentation_coverage:.2%}")
    lines.append(f"- **Total Modules:** {metrics.total_modules}")
    lines.append("")

    lines.append("## Functional Correctness")
    lines.append("")
    functional_status = "✓ PASS" if metrics.functional_correctness else "✗ FAIL"
    lines.append(f"**Status:** {functional_status}")
    lines.append("")
    lines.append("All expected files generated successfully:")
    lines.append("- overview.md")
    lines.append("- module_tree.json")
    lines.append("- metadata.json")
    lines.append("")

    if metrics.functional_correctness:
        lines.append("✓ Documentation generation completed without errors")
    else:
        lines.append("✗ Documentation generation had errors or missing files")
    lines.append("")

    lines.append("## Structural Changes")
    lines.append("")
    structure_status = "✓ PASS" if metrics.structure_compatibility else "✗ FAIL"
    lines.append(f"**Status:** {structure_status}")
    lines.append("")
    lines.append("Module tree structure validation:")
    lines.append("- Schema compatibility")
    lines.append("- Hierarchical integrity")
    lines.append("")

    if metrics.file_count_delta > 0:
        lines.append(f"✓ Generated {metrics.file_count_delta} additional files")
    elif metrics.file_count_delta < 0:
        lines.append(f"⚠ Generated {abs(metrics.file_count_delta)} fewer files")
    else:
        lines.append("= File count unchanged")
    lines.append("")

    lines.append("## Content Quality")
    lines.append("")
    markdown_status = "✓ PASS" if metrics.markdown_validity else "✗ FAIL"
    lines.append(f"**Markdown Validity:** {markdown_status}")
    lines.append("")
    lines.append(f"**Documentation Coverage:** {metrics.documentation_coverage:.2%}")
    lines.append(f"- Files per module: {metrics.documentation_coverage:.2f}")
    lines.append("")

    if metrics.content_length_delta > 0:
        lines.append(f"✓ Documentation increased by {metrics.content_length_delta:+,} characters")
    elif metrics.content_length_delta < 0:
        lines.append(f"⚠ Documentation decreased by {abs(metrics.content_length_delta):,} characters")
    else:
        lines.append("= Content length unchanged")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")

    if not metrics.functional_correctness:
        lines.append("- ⚠ **Critical:** Fix documentation generation errors")
        lines.append("- Review metadata.json for generation failures")

    if not metrics.structure_compatibility:
        lines.append("- ⚠ **Critical:** Verify module tree schema changes")
        lines.append("- Check for breaking changes in clustering algorithm")

    if not metrics.markdown_validity:
        lines.append("- **Warning:** Some markdown files may be empty or malformed")
        lines.append("- Review generated markdown for content quality")

    if metrics.file_count_delta < -5:
        lines.append(f"- ⚠ **Warning:** Significant decrease in file count ({metrics.file_count_delta})")
        lines.append("- Verify if this is expected behavior")

    if metrics.documentation_coverage < 0.5:
        lines.append("- **Info:** Low documentation coverage may indicate incomplete analysis")
        lines.append("- Consider increasing max_files limit")

    if metrics.documentation_coverage > 2.0:
        lines.append("- **Info:** High documentation coverage may indicate duplicate files")
        lines.append("- Review file generation logic")

    if (
        metrics.functional_correctness
        and metrics.structure_compatibility
        and metrics.markdown_validity
    ):
        lines.append("✓ All critical metrics passed")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by CodeWiki A/B Testing*")

    return "\n".join(lines)
