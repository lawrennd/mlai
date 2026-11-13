"""Unit tests for mlai.gp_tutorial (no GPy required)."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import mlai.gp_tutorial as gp_tutorial


@pytest.mark.unit
def test_ax_default_function_exists():
    assert callable(gp_tutorial.ax_default)


@pytest.mark.unit
def test_meanplot_function_exists():
    assert callable(gp_tutorial.meanplot)


@pytest.mark.unit
def test_gpplot_function_exists():
    assert callable(gp_tutorial.gpplot)


@pytest.mark.unit
@patch('mlai.gp_tutorial.plt')
def test_ax_default_creates_new_figure(mock_plt):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.figure.return_value = mock_fig
    mock_fig.add_subplot.return_value = mock_ax

    fig, ax = gp_tutorial.ax_default(fignum=1, ax=None)

    mock_plt.figure.assert_called_once_with(1)
    mock_fig.add_subplot.assert_called_once_with(111)
    assert fig is mock_fig
    assert ax is mock_ax


@pytest.mark.unit
@patch('mlai.gp_tutorial.plt')
def test_ax_default_uses_existing_axis(mock_plt):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax.figure = mock_fig

    fig, ax = gp_tutorial.ax_default(fignum=1, ax=mock_ax)

    assert fig is mock_fig
    assert ax is mock_ax
    mock_plt.figure.assert_not_called()


@pytest.mark.unit
@patch('mlai.gp_tutorial.ax_default')
def test_meanplot_calls_ax_default_and_plot(mock_ax_default):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax_default.return_value = (mock_fig, mock_ax)
    mock_line = MagicMock()
    mock_ax.plot.return_value = [mock_line]

    x = np.array([1.0, 2.0, 3.0])
    mu = np.array([0.5, 1.0, 1.5])
    result = gp_tutorial.meanplot(x, mu, color='red', linewidth=2)

    mock_ax_default.assert_called_once_with(None, None)
    mock_ax.plot.assert_called_once_with(x, mu, color='red', linewidth=2)
    assert result == [mock_line]


@pytest.mark.unit
@patch('mlai.gp_tutorial.meanplot')
@patch('mlai.gp_tutorial.ax_default')
def test_gpplot_creates_all_plot_elements(mock_ax_default, mock_meanplot):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax_default.return_value = (mock_fig, mock_ax)
    mock_ax.fill.return_value = MagicMock()
    mock_meanplot.return_value = [MagicMock()]

    x = np.array([1.0, 2.0, 3.0])
    mu = np.array([0.5, 1.0, 1.5])
    lower = np.array([0.0, 0.5, 1.0])
    upper = np.array([1.0, 1.5, 2.0])

    result = gp_tutorial.gpplot(x, mu, lower, upper)

    mock_ax_default.assert_called_once_with(None, None)
    assert mock_meanplot.call_count >= 3
    mock_ax.fill.assert_called_once()
    assert isinstance(result, list)


@pytest.mark.unit
def test_gpplot_real_axes_draws_fill():
    """Exercise the real matplotlib path for coverage."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 5)
    mu = np.sin(x)
    lower = mu - 0.1
    upper = mu + 0.1
    plots = gp_tutorial.gpplot(x, mu, lower, upper, ax=ax)
    assert isinstance(plots, list)
    assert len(plots) >= 3
    plt.close(fig)
