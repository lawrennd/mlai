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
    
    def test_write_figure_caption(self):
        """Test write_figure_caption function (lines 165-167)."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test parameters
            counter = 1
            caption = "Test caption for figure"
            filestub = "test_figure"
            ext = "svg"
            
            # Mock the write_figure function to avoid actual file writing
            with patch('mlai.mlai.write_figure') as mock_write_figure:
                mlai.write_figure_caption(counter, caption, filestub, ext=ext, directory=temp_dir)
                
                # Check that write_figure was called with correct parameters
                expected_filename = f"{filestub}_{counter:0>3}.{ext}"
                mock_write_figure.assert_called_once_with(expected_filename, directory=temp_dir, frameon=None)
                
                # Check that caption file was created
                caption_file = os.path.join(temp_dir, f"{filestub}_{counter:0>3}.md")
                assert os.path.exists(caption_file)
                
                # Check caption content
                with open(caption_file, 'r') as f:
                    content = f.read()
                    assert content == caption
    
    def test_finite_difference_gradient(self):
        """Test finite_difference_gradient function (lines 1539-1591)."""
        # Test with a scalar-valued function
        def scalar_func(x):
            return x[0]**2 + x[1]**2
        
        x = np.array([1.0, 2.0])
        h = 1e-6
        
        gradient = mlai.finite_difference_gradient(scalar_func, x, h)
        
        # Check that it's finite
        assert np.all(np.isfinite(gradient))
        
        # Check that it has the right shape (same as input)
        assert gradient.shape == x.shape
        
        # Check approximate values (derivatives of x[0]**2 + x[1]**2)
        # At x=[1,2]: d/dx[0] = 2*x[0] = 2, d/dx[1] = 2*x[1] = 4
        expected = np.array([2.0, 4.0])
        np.testing.assert_allclose(gradient, expected, rtol=1e-3)
        
        # Test with an array-valued function (should sum the output)
        def array_func(x):
            return np.array([x[0]**2, x[1]**2])  # Returns array, should be summed
        
        gradient_array = mlai.finite_difference_gradient(array_func, x, h)
        
        # Should still be 1D gradient (sum of array outputs)
        assert gradient_array.shape == x.shape
        assert np.all(np.isfinite(gradient_array))
    
    def test_verify_gradient_implementation(self):
        """Test verify_gradient_implementation function (lines 1683-1684)."""
        # Test with matching gradients
        analytical = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.0, 2.0, 3.0])
        
        result = mlai.verify_gradient_implementation(analytical, numerical)
        assert result is True
        
        # Test with different gradients - use more relaxed tolerance
        analytical = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.1, 2.1, 3.1])  # Different values
        
        result = mlai.verify_gradient_implementation(analytical, numerical, rtol=1e-1, atol=1e-1)
        assert result is True  # Should pass with relaxed tolerance
    
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
    
    def test_load_pgm_invalid_file(self):
        """Test load_pgm with invalid file (lines 2251-2252)."""
        import pytest
        # Test with non-PGM file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'This is not a PGM file')
            invalid_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Not a raw PGM file"):
                mlai.load_pgm(invalid_file)
        finally:
            os.unlink(invalid_file)
    
    def test_generate_swiss_roll(self):
        """Test generate_swiss_roll function (lines 3417-3423)."""
        X, t = mlai.generate_swiss_roll(n_points=100, noise=0.01)
        
        # Check shape
        assert X.shape == (100, 3)  # 3D coordinates
        assert t.shape == (100,)   # 1D parameter
        
        # Check that coordinates are finite
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(t))
        
        # Check that it's actually 3D (not all zeros)
        assert not np.allclose(X, 0)
        
        # Test with different parameters
        X2, t2 = mlai.generate_swiss_roll(n_points=50, noise=0.1)
        assert X2.shape == (50, 3)
        assert t2.shape == (50,)
    
    
    
    def test_radial_multivariate(self):
        """Test radial_multivariate function (lines 3132-3154)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        num_basis = 3
        
        result = mlai.radial_multivariate(X, num_basis, random_state=42)
        
        # Check shape
        assert result.shape == (3, 3)  # n_samples x num_basis
        
        # Check that it's finite
        assert np.all(np.isfinite(result))
        
        # Test with different parameters
        result2 = mlai.radial_multivariate(X, num_basis, width=1.0, random_state=42)
        assert result2.shape == (3, 3)
        assert np.all(np.isfinite(result2))
    
    def test_radial_single_basis(self):
        """Test radial function with single basis (lines 783-785)."""
        X = np.array([[0.5], [1.0], [1.5]])
        data_limits = [0.0, 2.0]
        
        # Test with num_basis=1 to trigger the else branch
        result = mlai.radial(X, num_basis=1, data_limits=data_limits)
        
        assert result.shape == (X.shape[0], 1)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Gaussian values should be non-negative

    def test_finite_difference_jacobian(self):
        """Test finite_difference_jacobian function (lines 1617-1640)."""
        def test_func(x):
            """Test function: f(x) = [x[0]^2, x[1]^3]"""
            return np.array([x[0]**2, x[1]**3])
        
        x = np.array([2.0, 3.0])
        jacobian = mlai.finite_difference_jacobian(test_func, x)
        
        # Check shape: 2 outputs, 2 inputs
        assert jacobian.shape == (2, 2)
        
        # Check that it's finite
        assert np.all(np.isfinite(jacobian))
        
        # Check approximate values (should be close to [[4, 0], [0, 27]])
        assert abs(jacobian[0, 0] - 4.0) < 0.1  # derivative of x^2 at x=2
        assert abs(jacobian[1, 1] - 27.0) < 0.1  # derivative of x^3 at x=3
        assert abs(jacobian[0, 1]) < 0.1  # cross-derivative should be ~0
        assert abs(jacobian[1, 0]) < 0.1  # cross-derivative should be ~0
    
    def test_dist2(self):
        """Test dist2 function (lines 3184, 3191-3192)."""
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        result = mlai.dist2(X1, X2)
        
        # Check shape
        assert result.shape == (2, 2)  # X1.shape[0] x X2.shape[0]
        
        # Check that distances are non-negative
        assert np.all(result >= 0)
        
        # Check that it's finite
        assert np.all(np.isfinite(result))
    
    
    
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
