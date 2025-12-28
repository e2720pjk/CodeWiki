#!/usr/bin/env python3
"""
End-to-end CLI tests for file limit options.
"""

import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner
from codewiki.cli.main import cli
from codewiki.cli.models.job import GenerationOptions


class TestCLIFileLimits:
    """End-to-end CLI tests for file limit options."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_accepts_max_files_option(self):
        """Verify CLI accepts --max-files option."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a minimal test repository
            temp_path = Path(temp_dir)
            (temp_path / "main.py").write_text("def main(): pass\n")

            result = self.runner.invoke(cli, ['generate', '--help'])
            assert '--max-files' in result.output
            assert '--max-entry-points' in result.output
            assert '--max-connectivity-files' in result.output

    def test_generation_options_has_file_limits(self):
        """Verify GenerationOptions has file limit fields."""
        options = GenerationOptions(
            max_files=200,
            max_entry_points=10,
            max_connectivity_files=15
        )
        assert options.max_files == 200
        assert options.max_entry_points == 10
        assert options.max_connectivity_files == 15

    def test_generation_options_defaults_match_configured(self):
        """Verify default values match the configured defaults."""
        options = GenerationOptions()
        assert options.max_files == 100
        assert options.max_entry_points == 5
        assert options.max_connectivity_files == 10


if __name__ == "__main__":
    pytest.main([__file__])
