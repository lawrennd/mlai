#!/usr/bin/env python3
"""
Tests for utility functions in mlai.

This module tests the utility functions that will be moved to utils.py
in the refactoring process.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import mlai modules
import mlai.mlai as mlai

class TestUtilityFunctions:
    """Test additional utility functions."""
    
    def test_load_pgm(self):
        """Test load_pgm function."""
        # Create a simple test PGM file
        with tempfile.NamedTemporaryFile(suffix='.pgm', delete=False) as f:
            f.write(b'P5\n2 2\n255\n\x00\xFF\xFF\x00')
            pgm_file = f.name
        
        try:
            result = mlai.load_pgm(pgm_file)
            assert result.shape == (2, 2)
            assert result.dtype == np.uint8
        finally:
            os.unlink(pgm_file)
    
    def test_contour_data(self):
        """Test contour_data function."""
        # Mock model and data
        model = MagicMock()
        data = {'X': np.array([[1, 2], [3, 4]]), 'Y': np.array([1, 2])}  # Proper data structure
        length_scales = np.array([0.1, 0.5, 1.0])
        log_SNRs = np.array([0, 1, 2])
        
        result = mlai.contour_data(model, data, length_scales, log_SNRs)
        assert len(result) == 3  # Should return X, Y, Z for contour plot
        # Just check that we get arrays back
        assert all(isinstance(arr, np.ndarray) for arr in result) 

    
    def test_filename_join_no_directory(self):
        """Test filename_join with no directory specified."""
        result = mlai.filename_join("test.png")
        assert result == "test.png"
    
    def test_filename_join_with_directory(self):
        """Test filename_join with directory specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = mlai.filename_join("test.png", temp_dir)
            expected = os.path.join(temp_dir, "test.png")
            assert result == expected
    
    def test_filename_join_creates_directory(self):
        """Test filename_join creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new_subdir")
            result = mlai.filename_join("test.png", new_dir)
            expected = os.path.join(new_dir, "test.png")
            assert result == expected
            assert os.path.exists(new_dir)
    
    def test_write_animation(self):
        """Test write_animation function."""
        mock_anim = MagicMock()
        with tempfile.TemporaryDirectory() as temp_dir:
            mlai.write_animation(mock_anim, "test.gif", temp_dir, fps=10)
            expected_path = os.path.join(temp_dir, "test.gif")
            mock_anim.save.assert_called_once_with(expected_path, fps=10)
    
    def test_write_animation_html(self):
        """Test write_animation_html function."""
        mock_anim = MagicMock()
        mock_anim.to_jshtml.return_value = "<html>test</html>"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mlai.write_animation_html(mock_anim, "test.html", temp_dir)
            expected_path = os.path.join(temp_dir, "test.html")
            assert os.path.exists(expected_path)
            
            with open(expected_path, 'r') as f:
                content = f.read()
            assert content == "<html>test</html>"
    
    @patch('matplotlib.pyplot.savefig')
    def test_write_figure_current_figure(self, mock_savefig):
        """Test write_figure with current figure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mlai.write_figure("test.png", directory=temp_dir)
            expected_path = os.path.join(temp_dir, "test.png")
            mock_savefig.assert_called_once_with(expected_path, transparent=True)
    
    @patch('matplotlib.figure.Figure.savefig')
    def test_write_figure_specific_figure(self, mock_savefig):
        """Test write_figure with specific figure."""
        mock_figure = MagicMock()
        with tempfile.TemporaryDirectory() as temp_dir:
            mlai.write_figure("test.png", figure=mock_figure, directory=temp_dir)
            expected_path = os.path.join(temp_dir, "test.png")
            mock_figure.savefig.assert_called_once_with(expected_path, transparent=True)
    
    def test_write_figure_custom_kwargs(self):
        """Test write_figure with custom kwargs."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with tempfile.TemporaryDirectory() as temp_dir:
                mlai.write_figure("test.png", directory=temp_dir, dpi=300, transparent=False)
                expected_path = os.path.join(temp_dir, "test.png")
                mock_savefig.assert_called_once_with(expected_path, dpi=300, transparent=False)


class TestUtilityFunctionEdgeCases:
    """Test edge cases for utility functions."""
    
    def test_filename_join_edge_cases(self):
        """Test filename_join with edge cases."""
        # Test with empty filename
        result = mlai.filename_join("")
        assert result == ""
        
        # Test with None directory
        result = mlai.filename_join("test.png", None)
        assert result == "test.png"
        
        # Test with empty directory - this should work without creating directory
        with tempfile.TemporaryDirectory() as temp_dir:
            result = mlai.filename_join("test.png", temp_dir)
            assert result == os.path.join(temp_dir, "test.png")
    
    def test_write_figure_edge_cases(self):
        """Test write_figure with edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with custom kwargs that override defaults
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                mlai.write_figure("test.png", directory=temp_dir, transparent=False, dpi=300)
                expected_path = os.path.join(temp_dir, "test.png")
                mock_savefig.assert_called_once_with(expected_path, transparent=False, dpi=300) 

if __name__ == '__main__':
    unittest.main()
