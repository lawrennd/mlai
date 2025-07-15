import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import mlai


@pytest.mark.unit
def test_ax_default_function_exists():
    """Test that ax_default function exists and is callable."""
    if not mlai.GPY_AVAILABLE:
        pytest.skip("GPy not available, skipping gp_tutorial tests")
    from mlai import gp_tutorial
    assert hasattr(gp_tutorial, 'ax_default')
    assert callable(gp_tutorial.ax_default)


@pytest.mark.unit
def test_meanplot_function_exists():
    """Test that meanplot function exists and is callable."""
    if not mlai.GPY_AVAILABLE:
        pytest.skip("GPy not available, skipping gp_tutorial tests")
    from mlai import gp_tutorial
    assert hasattr(gp_tutorial, 'meanplot')
    assert callable(gp_tutorial.meanplot)


@pytest.mark.unit
def test_gpplot_function_exists():
    """Test that gpplot function exists and is callable."""
    if not mlai.GPY_AVAILABLE:
        pytest.skip("GPy not available, skipping gp_tutorial tests")
    from mlai import gp_tutorial
    assert hasattr(gp_tutorial, 'gpplot')
    assert callable(gp_tutorial.gpplot)


@pytest.mark.unit
def test_ax_default_signature():
    """Test that ax_default has the expected parameters."""
    if not mlai.GPY_AVAILABLE:
        pytest.skip("GPy not available, skipping gp_tutorial tests")
    from mlai import gp_tutorial
    import inspect
    sig = inspect.signature(gp_tutorial.ax_default)
    params = list(sig.parameters.keys())
    assert 'fignum' in params
    assert 'ax' in params


@pytest.mark.unit
def test_meanplot_signature():
    """Test that meanplot has the expected parameters."""
    if not mlai.GPY_AVAILABLE:
        pytest.skip("GPy not available, skipping gp_tutorial tests")
    from mlai import gp_tutorial
    import inspect
    sig = inspect.signature(gp_tutorial.meanplot)
    params = list(sig.parameters.keys())
    assert 'x' in params
    assert 'mu' in params
    assert 'color' in params
    assert 'ax' in params


@pytest.mark.unit
def test_gpplot_signature():
    """Test that gpplot has the expected parameters."""
    if not mlai.GPY_AVAILABLE:
        pytest.skip("GPy not available, skipping gp_tutorial tests")
    from mlai import gp_tutorial
    import inspect
    sig = inspect.signature(gp_tutorial.gpplot)
    params = list(sig.parameters.keys())
    assert 'x' in params
    assert 'mu' in params
    assert 'lower' in params
    assert 'upper' in params
    assert 'edgecol' in params
    assert 'fillcol' in params
    assert 'ax' in params


@pytest.mark.unit
@patch('mlai.gp_tutorial.plt')
def test_ax_default_creates_new_figure(mock_plt):
    """Test ax_default creates a new figure when ax is None."""
    if not mlai.GPY_AVAILABLE:
        pytest.skip("GPy not available, skipping gp_tutorial tests")
    from mlai import gp_tutorial
    # Mock the matplotlib components
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.figure.return_value = mock_fig
    mock_fig.add_subplot.return_value = mock_ax
    
    # Test the function
    fig, ax = gp_tutorial.ax_default(fignum=1, ax=None)
    
    # Verify the function was called correctly
    mock_plt.figure.assert_called_once_with(1)
    mock_fig.add_subplot.assert_called_once_with(111)
    assert fig is mock_fig
    assert ax is mock_ax


@pytest.mark.unit
@patch('mlai.gp_tutorial.plt')
def test_ax_default_uses_existing_axis(mock_plt):
    """Test ax_default uses existing axis when provided."""
    if not mlai.GPY_AVAILABLE:
        pytest.skip("GPy not available, skipping gp_tutorial tests")
    from mlai import gp_tutorial
    # Mock the matplotlib components
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax.figure = mock_fig
    
    # Test the function
    fig, ax = gp_tutorial.ax_default(fignum=1, ax=mock_ax)
    
    # Verify the function returns the existing axis
    assert fig is mock_fig
    assert ax is mock_ax
    # Should not create a new figure
    mock_plt.figure.assert_not_called()


@pytest.mark.unit
@patch('mlai.gp_tutorial.ax_default')
@patch('mlai.gp_tutorial.plt')
def test_meanplot_calls_ax_default_and_plot(mock_plt, mock_ax_default):
    """Test meanplot calls ax_default and plots correctly."""
    if not mlai.GPY_AVAILABLE:
        pytest.skip("GPy not available, skipping gp_tutorial tests")
    from mlai import gp_tutorial
    # Mock the components
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax_default.return_value = (mock_fig, mock_ax)
    mock_line = MagicMock()
    mock_ax.plot.return_value = [mock_line]
    
    # Test data
    x = np.array([1, 2, 3])
    mu = np.array([0.5, 1.0, 1.5])
    
    # Test the function
    result = gp_tutorial.meanplot(x, mu, color='red', linewidth=2)
    
    # Verify ax_default was called
    mock_ax_default.assert_called_once_with(None, None)
    # Verify plot was called with correct parameters
    mock_ax.plot.assert_called_once_with(x, mu, color='red', linewidth=2)
    # Verify result
    assert result == [mock_line]


@pytest.mark.unit
@patch('mlai.gp_tutorial.meanplot')
@patch('mlai.gp_tutorial.ax_default')
@patch('mlai.gp_tutorial.plt')
def test_gpplot_creates_all_plot_elements(mock_plt, mock_ax_default, mock_meanplot):
    """Test gpplot creates all expected plot elements."""
    if not mlai.GPY_AVAILABLE:
        pytest.skip("GPy not available, skipping gp_tutorial tests")
    from mlai import gp_tutorial
    # Mock the components
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax_default.return_value = (mock_fig, mock_ax)
    mock_ax.fill.return_value = MagicMock()
    mock_meanplot.return_value = [MagicMock()]
    
    # Test data
    x = np.array([1, 2, 3])
    mu = np.array([0.5, 1.0, 1.5])
    lower = np.array([0.0, 0.5, 1.0])
    upper = np.array([1.0, 1.5, 2.0])
    
    # Test the function
    result = gp_tutorial.gpplot(x, mu, lower, upper)
    
    # Verify ax_default was called
    mock_ax_default.assert_called_once_with(None, None)
    # Verify meanplot was called multiple times (for mean, upper, lower)
    assert mock_meanplot.call_count >= 3
    # Verify fill was called
    mock_ax.fill.assert_called_once()
    # Verify result is a list
    assert isinstance(result, list) 