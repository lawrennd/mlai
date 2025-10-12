"""
Shared pytest fixtures for MLAI testing.
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Automatically clean up matplotlib state after each test."""
    yield
    plt.close('all')


@pytest.fixture(scope="session")
def random_seed():
    """Set a fixed random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_data_2d():
    """Generate sample 2D data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def sample_data_1d():
    """Generate sample 1D data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def mock_matplotlib_backend():
    """Use non-interactive matplotlib backend for testing."""
    with patch('matplotlib.pyplot.show'):
        plt.switch_backend('Agg')
        yield
        plt.close('all')


@pytest.fixture
def temp_plot_dir(tmp_path):
    """Create a temporary directory for saving test plots."""
    plot_dir = tmp_path / "test_plots"
    plot_dir.mkdir()
    return plot_dir 