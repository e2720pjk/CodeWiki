#!/usr/bin/env python3
"""
Unit tests for file limit validation functionality.
"""

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


if __name__ == "__main__":
    pytest.main([__file__])