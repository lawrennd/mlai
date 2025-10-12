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
    """Test utility functions for file operations and plotting."""
    
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


if __name__ == '__main__':
    unittest.main()
