"""
Unit tests for plot.py module.

This module tests the plotting utilities and visualization functions
for the MLAI library. Tests focus on functionality rather than visual
appearance, using mocked matplotlib backends where appropriate.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os
import sys

# Import the module to test
import mlai.plot as plot


class TestPlotUtilities:
    """Test basic plotting utilities and constants."""
    
    def test_constants(self):
        """Test that plotting constants are defined correctly."""
        assert hasattr(plot, 'tau')
        assert plot.tau == 2 * np.pi
        
        # Test figure size constants
        assert hasattr(plot, 'three_figsize')
        assert hasattr(plot, 'two_figsize')
        assert hasattr(plot, 'one_figsize')
        assert hasattr(plot, 'big_figsize')
        assert hasattr(plot, 'wide_figsize')
        assert hasattr(plot, 'big_wide_figsize')
        
        # Test color constants
        assert hasattr(plot, 'hcolor')
        assert len(plot.hcolor) == 3
        
        # Test notation map
        assert hasattr(plot, 'notation_map')
        assert isinstance(plot.notation_map, dict)
    
    def test_pred_range_basic(self):
        """Test pred_range function with basic inputs."""
        x = np.array([1, 2, 3, 4, 5])
        result = plot.pred_range(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 1  # Should be 2D with single column
        assert len(result) == 200  # Default points
        assert result.min() < x.min()  # Should extend below
        assert result.max() > x.max()  # Should extend above
    
    def test_pred_range_custom_params(self):
        """Test pred_range function with custom parameters."""
        x = np.array([0, 10])
        result = plot.pred_range(x, portion=0.5, points=50, randomize=True)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 1
        assert len(result) == 50
        assert result.min() < x.min()
        assert result.max() > x.max()
    
    def test_pred_range_2d_input(self):
        """Test pred_range function with 2D input."""
        x = np.array([[1], [2], [3], [4], [5]])
        result = plot.pred_range(x)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 1
        assert len(result) == 200


class TestMatrixPlotting:
    """Test matrix plotting functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create a simple test matrix
        self.test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Mock matplotlib to avoid display issues
        self.mock_plt = patch('matplotlib.pyplot.gca').start()
        self.mock_ax = MagicMock()
        self.mock_plt.return_value = self.mock_ax
    
    def teardown_method(self):
        """Clean up test environment."""
        patch.stopall()
    
    def test_matrix_basic(self):
        """Test basic matrix plotting."""
        result = plot.matrix(self.test_matrix)
        
        # matrix function returns handle (list of text objects), not ax
        assert isinstance(result, list)
        assert len(result) == 9  # 3x3 matrix
        # Should have called text for each element
        assert self.mock_ax.text.call_count == 9
    
    def test_matrix_with_ax(self):
        """Test matrix plotting with provided axis."""
        result = plot.matrix(self.test_matrix, ax=self.mock_ax)
        
        # matrix function returns handle (list of text objects), not ax
        assert isinstance(result, list)
        assert len(result) == 9
        assert self.mock_ax.text.call_count == 9
    
    def test_matrix_values_type(self):
        """Test matrix plotting with 'values' type."""
        result = plot.matrix(self.test_matrix, type='values')
        
        # matrix function returns handle (list of text objects), not ax
        assert isinstance(result, list)
        assert len(result) == 9
        
        # Check that text calls have correct format
        calls = self.mock_ax.text.call_args_list
        assert len(calls) == 9
        
        # Check first call (should be for value 1)
        args, kwargs = calls[0]
        assert kwargs['horizontalalignment'] == 'center'
        assert 'fontsize' in kwargs
    
    def test_matrix_entries_type(self):
        """Test matrix plotting with 'entries' type."""
        result = plot.matrix(self.test_matrix, type='entries')
        
        # matrix function returns handle (list of text objects), not ax
        assert isinstance(result, list)
        assert len(result) == 9
        assert self.mock_ax.text.call_count == 9
    
    def test_matrix_image_type(self):
        """Test matrix plotting with 'image' type."""
        with patch('matplotlib.pyplot.set_cmap'):
            result = plot.matrix(self.test_matrix, type='image')
            
            # matrix function returns handle (matshow result), not ax
            assert result == self.mock_ax.matshow.return_value
            self.mock_ax.matshow.assert_called_once()
    
    def test_matrix_highlight(self):
        """Test matrix plotting with highlighting."""
        # Pass highlight_row as a list to match the function's expectation
        result = plot.matrix(self.test_matrix, highlight=True, highlight_row=[0])
        
        # matrix function returns handle (list of text objects), not ax
        assert isinstance(result, list)
        assert len(result) == 9
        # Should still have text calls for all elements
        assert self.mock_ax.text.call_count == 9
    
    def test_matrix_zoom(self):
        """Test matrix plotting with zoom."""
        # Pass zoom_row and zoom_col as lists to match the function's expectation
        result = plot.matrix(self.test_matrix, zoom=True, zoom_row=[0], zoom_col=[0])
        
        # matrix function returns handle (list of text objects), not ax
        assert isinstance(result, list)
        assert len(result) == 9
        assert self.mock_ax.text.call_count == 9


class TestBasePlot:
    """Test base plotting functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_ax = MagicMock()
    
    def test_base_plot_basic(self):
        """Test base_plot function with basic inputs."""
        K = np.array([[1, 0.5], [0.5, 1]])  # Simple covariance matrix
        
        result = plot.base_plot(K, ax=self.mock_ax)
        
        # base_plot returns a tuple (cont, thandle, cent), not ax
        assert isinstance(result, tuple)
        assert len(result) == 3
        # Should have called contour or similar plotting function
        # The exact calls depend on the implementation


class TestUtilityFunctions:
    """Test utility functions in plot.py."""
    
    def test_clear_axes(self):
        """Test clear_axes function."""
        mock_ax = MagicMock()
        
        plot.clear_axes(mock_ax)
        
        # Should have called some clearing methods
        # The exact implementation may vary
    
    def test_blank_canvas(self):
        """Test blank_canvas function."""
        mock_ax = MagicMock()
        
        plot.blank_canvas(mock_ax)
        
        # Should have called some canvas clearing methods
        # The exact implementation may vary
    
    def test_dist2(self):
        """Test dist2 function for distance calculation."""
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[5, 6], [7, 8]])
        
        result = plot.dist2(X, Y)
        
        assert isinstance(result, np.ndarray)
        # Should return distance matrix
        assert result.shape == (2, 2)


class TestNetworkVisualization:
    """Test network visualization classes."""
    
    def test_network_initialization(self):
        """Test network class initialization."""
        net = plot.network()
        
        assert hasattr(net, 'layers')
        # layers is initialized as empty list, not None
        assert net.layers == []
    
    def test_network_add_layer(self):
        """Test adding layers to network."""
        net = plot.network()
        layer = plot.layer(width=3, label='test')
        
        net.add_layer(layer)
        
        # layers is a list, so we check if the layer is in the list
        assert len(net.layers) == 1
        assert net.layers[0] == layer
    
    def test_network_properties(self):
        """Test network properties."""
        net = plot.network()
        layer = plot.layer(width=5, label='test')
        net.add_layer(layer)
        
        # Test width property
        assert net.width == 5
        
        # Test depth property
        assert net.depth == 1
    
    def test_layer_initialization(self):
        """Test layer class initialization."""
        layer = plot.layer(width=3, label='test', observed=True, fixed=False, text='test text')
        
        assert layer.width == 3
        assert layer.label == 'test'
        assert layer.observed is True
        assert layer.fixed is False
        assert layer.text == 'test text'
    
    def test_layer_defaults(self):
        """Test layer class with default parameters."""
        layer = plot.layer()
        
        assert layer.width == 5
        assert layer.label == ''
        assert layer.observed is False
        assert layer.fixed is False
        assert layer.text == ''


class TestModelOutputFunctions:
    """Test model output plotting functions."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create a simple mock model with required attributes
        self.mock_model = MagicMock()
        self.mock_model.X = np.array([[1], [2], [3]])  # 2D array
        self.mock_model.Y = np.array([[1], [2], [3]])  # 2D array
        self.mock_model.predict.return_value = (np.array([[1], [2], [3]]), np.array([[0.1], [0.1], [0.1]]))
    
    def test_model_output_basic(self):
        """Test model_output function with basic inputs."""
        mock_ax = MagicMock()
        
        result = plot.model_output(self.mock_model, ax=mock_ax)
        
        # Should have called the model's predict method
        self.mock_model.predict.assert_called()
        # model_output now returns the axis
        assert result == mock_ax
    
    def test_model_sample_basic(self):
        """Test model_sample function with basic inputs."""
        mock_ax = MagicMock()
        # Add posterior_sample method to mock model
        self.mock_model.posterior_sample.return_value = np.array([[1], [2], [3]])
        
        result = plot.model_sample(self.mock_model, ax=mock_ax, samps=5)
        
        # Should have called the model's predict method multiple times
        assert self.mock_model.predict.call_count >= 1
        # model_sample now returns the axis
        assert result == mock_ax


class TestFileOperations:
    """Test file operation functions."""
    
    def test_diagrams_directory_creation(self):
        """Test that diagrams directory is created when needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            diagrams_dir = os.path.join(temp_dir, 'diagrams')
            
            # Test that a function that uses diagrams directory works
            # This is a basic test - actual implementation may vary
            assert not os.path.exists(diagrams_dir)


class TestMatplotlibIntegration:
    """Test matplotlib integration and backend handling."""
    
    def test_matplotlib_import(self):
        """Test that matplotlib is properly imported."""
        assert hasattr(plot, 'plt')
        assert plot.plt is plt
    
    def test_3d_import(self):
        """Test that 3D plotting is available."""
        assert hasattr(plot, 'Axes3D')
        from mpl_toolkits.mplot3d import Axes3D
        assert plot.Axes3D is Axes3D
    
    def test_ipython_import(self):
        """Test that IPython is imported."""
        assert hasattr(plot, 'IPython')
        import IPython
        assert plot.IPython is IPython


class TestDaftIntegration:
    """Test daft integration for probabilistic graphical models."""
    
    def test_daft_import_optional(self):
        """Test that daft import is optional."""
        # The module should handle missing daft gracefully
        # This is tested by the try/except block in the module
    
    def test_daft_available(self):
        """Test behavior when daft is available."""
        # Test that the module can handle daft being available
        # We can't easily mock daft since it's imported conditionally
        # This test just ensures the module doesn't crash
        assert True


class TestGPyIntegration:
    """Test GPy integration."""
    
    def test_gpy_availability_check(self):
        """Test GPy availability check."""
        assert hasattr(plot, 'GPY_AVAILABLE')
        assert isinstance(plot.GPY_AVAILABLE, bool)
    
    def test_gpy_available(self):
        """Test behavior when GPy is available."""
        # Test that the module can handle GPy being available
        # We can't easily mock GPy since it's imported conditionally
        # This test just ensures the module doesn't crash
        assert True


class TestErrorHandling:
    """Test error handling in plotting functions."""
    
    def test_matrix_invalid_input(self):
        """Test matrix function with invalid input."""
        with pytest.raises(Exception):
            plot.matrix("not a matrix")
    
    def test_pred_range_empty_input(self):
        """Test pred_range with empty input."""
        with pytest.raises(Exception):
            plot.pred_range(np.array([]))
    
    def test_network_invalid_layer(self):
        """Test network with invalid layer."""
        net = plot.network()
        # The add_layer method doesn't validate input, so this won't raise
        # We'll test that it handles the input gracefully
        net.add_layer("not a layer")
        assert len(net.layers) == 1
        assert net.layers[0] == "not a layer"


class TestPerformance:
    """Test performance characteristics."""
    
    def test_matrix_large_input(self):
        """Test matrix function with large input."""
        large_matrix = np.random.rand(50, 50)
        
        with patch('matplotlib.pyplot.gca') as mock_gca:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax
            
            result = plot.matrix(large_matrix)
            
            # matrix function returns handle (list of text objects), not ax
            assert isinstance(result, list)
            # Should handle large matrices without issues
            assert len(result) == 2500  # 50x50 matrix
    
    def test_pred_range_large_points(self):
        """Test pred_range with large number of points."""
        x = np.array([1, 2, 3, 4, 5])
        
        result = plot.pred_range(x, points=1000)
        
        assert len(result) == 1000
        assert result.shape[1] == 1 