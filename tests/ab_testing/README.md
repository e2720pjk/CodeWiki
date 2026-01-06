# A/B Testing for CodeWiki Documentation

This directory contains A/B testing infrastructure to compare documentation reports between different git versions, ensuring backward compatibility and quality improvements.

## Directory Structure

- `utils.py` - Utility functions for generating documentation and calculating comparison metrics
- `test_ab_comparison.py` - Pytest test suite for A/B comparison
- `fixtures/` - Directory for test outputs and comparison reports

## Test Overview

### Unit Tests

- `test_calculate_metrics_with_empty_dirs()` - Tests metric calculation with minimal data
- `test_generate_report_invalid_version()` - Tests error handling for invalid git tags

### Integration Test

- `test_ab_comparison()` - Full A/B comparison test that:
  1. Generates documentation for baseline version (v0.1.0 by default)
  2. Generates documentation for current version (HEAD by default)
  3. Calculates comparison metrics
  4. Generates comparison report
  5. Asserts on critical metrics

## Metrics

The A/B test calculates the following metrics:

- **Functional Correctness**: Whether all expected files (overview.md, module_tree.json, metadata.json) are generated
- **File Count Delta**: Change in number of generated markdown files
- **Structure Compatibility**: Whether module_tree.json schema is preserved
- **Documentation Coverage**: Ratio of files to modules
- **Content Length Delta**: Total character change across all markdown files
- **Markdown Validity**: Whether all markdown files are well-formed

## Requirements

To run the full A/B comparison test, you need:

1. Git tags v0.1.0 and v0.1.1 (or other versions to compare)
2. LLM API configuration (see `codewiki config set` for setup)
3. Python dependencies installed (GitPython, pytest, pytest-cov)

## Running Tests

### Run all unit tests (no LLM required):

```bash
pytest tests/ab_testing/test_ab_comparison.py::test_calculate_metrics_with_empty_dirs -v
pytest tests/ab_testing/test_ab_comparison.py::test_generate_report_invalid_version -v
```

### Run full A/B comparison (requires LLM config):

```bash
pytest tests/ab_testing/test_ab_comparison.py::test_ab_comparison -v -s
```

### Custom versions via environment variables:

```bash
CODEWIKI_BASELINE_VERSION=v0.1.1 CODEWIKI_CURRENT_VERSION=HEAD pytest tests/ab_testing/test_ab_comparison.py::test_ab_comparison -v -s
```

## Output

The comparison report is saved to `tests/ab_testing/fixtures/comparison_report.md`.

## Troubleshooting

### LLM Configuration Error

If you see "Configuration not found or invalid", run:

```bash
codewiki config set --api-key <your-key> --base-url <api-url> --main-model <model> --cluster-model <model>
```

### Version Tag Not Found

Ensure git tags are fetched:

```bash
git fetch --tags
```

### Worktree Cleanup Issues

If worktree directories remain after failed tests, manually clean them:

```bash
git worktree list
git worktree remove <worktree-path> --force
```
