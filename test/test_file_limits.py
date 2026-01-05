#!/usr/bin/env python3
"""
Unit tests for file limit validation functionality.
"""

import os
import tempfile
from pathlib import Path
from typing import List

import pytest

from codewiki.src.be.dependency_analyzer.analysis.analysis_service import AnalysisService


class TestFileLimitValidation:
    """Test cases for file limit validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analysis_service = AnalysisService()

    def test_valid_file_limits(self):
        """Test that valid file limits pass validation."""
        # Should not raise any exceptions
        self.analysis_service._validate_file_limits(100, 5, 10)
        self.analysis_service._validate_file_limits(1, 1, 1)
        self.analysis_service._validate_file_limits(1000, 50, 100)

    def test_invalid_max_files_negative(self):
        """Test that negative max_files fails validation."""
        with pytest.raises(ValueError, match="max_files must be positive"):
            self.analysis_service._validate_file_limits(0, 5, 10)

        with pytest.raises(ValueError, match="max_files must be positive"):
            self.analysis_service._validate_file_limits(-10, 5, 10)

    def test_invalid_max_entry_points_negative(self):
        """Test that negative max_entry_points fails validation."""
        with pytest.raises(ValueError, match="max_entry_points must be positive"):
            self.analysis_service._validate_file_limits(100, 0, 10)

        with pytest.raises(ValueError, match="max_entry_points must be positive"):
            self.analysis_service._validate_file_limits(100, -5, 10)

    def test_invalid_max_connectivity_files_negative(self):
        """Test that negative max_connectivity_files fails validation."""
        with pytest.raises(ValueError, match="max_connectivity_files must be positive"):
            self.analysis_service._validate_file_limits(100, 5, 0)

        with pytest.raises(ValueError, match="max_connectivity_files must be positive"):
            self.analysis_service._validate_file_limits(100, 5, -10)

    def test_max_entry_points_exceeds_max_files(self):
        """Test that max_entry_points exceeding max_files fails validation."""
        with pytest.raises(ValueError, match="max_entry_points cannot exceed max_files"):
            self.analysis_service._validate_file_limits(10, 15, 5)

        with pytest.raises(ValueError, match="max_entry_points cannot exceed max_files"):
            self.analysis_service._validate_file_limits(5, 10, 10)

    def test_max_connectivity_files_exceeds_max_files(self):
        """Test that max_connectivity_files exceeding max_files fails validation."""
        with pytest.raises(ValueError, match="max_connectivity_files cannot exceed max_files"):
            self.analysis_service._validate_file_limits(10, 5, 15)

        with pytest.raises(ValueError, match="max_connectivity_files cannot exceed max_files"):
            self.analysis_service._validate_file_limits(5, 5, 10)

    def test_max_files_too_large(self):
        """Test that max_files exceeding reasonable limit fails validation."""
        with pytest.raises(ValueError, match="max_files cannot exceed 10000"):
            self.analysis_service._validate_file_limits(10001, 5, 10)

        with pytest.raises(ValueError, match="max_files cannot exceed 10000"):
            self.analysis_service._validate_file_limits(50000, 5, 10)

    def test_edge_case_equal_values(self):
        """Test edge cases where values are equal."""
        # All equal values should be valid
        self.analysis_service._validate_file_limits(5, 5, 5)

        # max_entry_points equals max_files should be valid
        self.analysis_service._validate_file_limits(10, 10, 5)

        # max_connectivity_files equals max_files should be valid
        self.analysis_service._validate_file_limits(10, 5, 10)


class TestFileLimitIntegration:
    """Integration tests for actual file limiting behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analysis_service = AnalysisService()

    def _create_test_files(self, temp_dir: Path, count: int, prefix: str = "file") -> List[Path]:
        """Create test files in the given directory."""
        files = []
        for i in range(count):
            file_path = temp_dir / f"{prefix}_{i}.py"
            file_path.write_text(f"def func_{i}():\n    pass\n")
            files.append(file_path)
        return files

    def test_max_files_actually_limits_files(self):
        """Verify that max_files actually limits the number of files analyzed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create 50 test files
            self._create_test_files(temp_path, 50)

            # Analyze with max_files=10
            result = self.analysis_service.analyze_local_repository(
                str(temp_path), max_files=10, max_entry_points=5, max_connectivity_files=10
            )

            # Verify only 10 files were analyzed
            assert result["summary"]["total_files"] == 10
            assert len(result["nodes"]) <= 10

    def test_max_entry_points_actually_limits_entry_points(self):
        """Verify that max_entry_points actually limits entry points."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create potential entry point files
            entry_point_names = ["main.py", "app.py", "index.py", "server.py", "start.py", "run.py"]
            for name in entry_point_names:
                (temp_path / name).write_text(f"# {name}\ndef main():\n    pass\n")

            # Create additional non-entry point files
            self._create_test_files(temp_path, 20, prefix="other")

            # Analyze with max_entry_points=3
            result = self.analysis_service.analyze_local_repository(
                str(temp_path), max_files=100, max_entry_points=3, max_connectivity_files=10
            )

            # Verify entry points are limited
            assert len(result["entry_points"]) <= 3
            assert result["summary"]["total_entry_points"] == len(result["entry_points"])

            # Verify entry_points is a subset of analyzed files
            entry_point_paths = {ep["path"] for ep in result["entry_points"]}
            assert entry_point_paths

    def test_max_connectivity_files_actually_limits_connectivity_files(self):
        """Verify that max_connectivity_files actually limits connectivity files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create source directories with files
            for dir_name in ["src", "lib", "app", "core"]:
                dir_path = temp_path / dir_name
                dir_path.mkdir()
                self._create_test_files(dir_path, 10, prefix=f"{dir_name}_file")

            # Create some root files
            self._create_test_files(temp_path, 5, prefix="root")

            # Analyze with max_connectivity_files=10
            result = self.analysis_service.analyze_local_repository(
                str(temp_path), max_files=100, max_entry_points=5, max_connectivity_files=10
            )

            # Verify connectivity files are limited
            assert len(result["connectivity_files"]) <= 10
            assert result["summary"]["total_connectivity_files"] == len(result["connectivity_files"])

            # Verify connectivity_files are from source directories
            connectivity_paths = [cf["path"] for cf in result["connectivity_files"]]
            assert any("src/" in p or "lib/" in p or "app/" in p or "core/" in p for p in connectivity_paths)

    def test_all_limits_respected_together(self):
        """Verify that all three limits are respected when used together."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create entry point files
            for name in ["main.py", "app.py", "index.py", "server.py", "start.py"]:
                (temp_path / name).write_text(f"# {name}\ndef main():\n    pass\n")

            # Create source directory files
            for dir_name in ["src", "lib"]:
                dir_path = temp_path / dir_name
                dir_path.mkdir()
                self._create_test_files(dir_path, 30, prefix=f"{dir_name}_file")

            # Create additional files
            self._create_test_files(temp_path, 50, prefix="other")

            # Analyze with all three limits
            result = self.analysis_service.analyze_local_repository(
                str(temp_path), max_files=50, max_entry_points=5, max_connectivity_files=10
            )

            # Verify all limits are respected
            assert result["summary"]["total_files"] <= 50
            assert len(result["entry_points"]) <= 5
            assert len(result["connectivity_files"]) <= 10

            # Verify entry_points and connectivity_files are subsets of analyzed files
            all_analyzed_files = set()
            for node in result["nodes"]:
                if "file_path" in node:
                    all_analyzed_files.add(node["file_path"])

            # Verify validation rules (entry/connectivity counts don't exceed max_files)
            assert len(result["entry_points"]) <= result["summary"]["total_files"]
            assert len(result["connectivity_files"]) <= result["summary"]["total_files"]

    @pytest.mark.parametrize(
        "file_count,limit",
        [
            (10, 10),
            (50, 25),
            (100, 50),
        ],
    )
    def test_scalability_at_different_scales(self, file_count, limit):
        """Test file limiting works with various project sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create specified number of files
            self._create_test_files(temp_path, file_count)

            # Analyze with limit
            result = self.analysis_service.analyze_local_repository(str(temp_path), max_files=limit)

            # Verify result matches expectation
            if file_count >= limit:
                assert result["summary"]["total_files"] == limit
            else:
                assert result["summary"]["total_files"] == file_count

    def test_backward_compatibility_existing_fields(self):
        """Verify backward compatibility: existing fields are present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some test files
            self._create_test_files(temp_path, 10)

            # Analyze
            result = self.analysis_service.analyze_local_repository(str(temp_path), max_files=10)

            # Verify existing fields are present
            assert "nodes" in result
            assert "relationships" in result
            assert "summary" in result
            assert "total_files" in result["summary"]
            assert "total_nodes" in result["summary"]
            assert "total_relationships" in result["summary"]

            # Verify new fields are present
            assert "entry_points" in result
            assert "connectivity_files" in result
            assert "total_entry_points" in result["summary"]
            assert "total_connectivity_files" in result["summary"]


if __name__ == "__main__":
    pytest.main([__file__])
