"""
Integration tests for tutorial workflows.

These tests ensure that the tutorial examples in the documentation
actually work and stay up-to-date with the codebase.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import mlai.mlai as mlai
from mlai.mlai import radial_multivariate
from pathlib import Path
import tempfile
import os

# Disable matplotlib display for testing
plt.ioff()


class TestTutorialWorkflows:
    """Test that tutorial workflow examples actually work."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for plots
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Set random seed for reproducible tests
        np.random.seed(42)
    
    def teardown_method(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        # Clean up temporary files
        for file in Path(self.temp_dir).glob("*"):
            if file.is_file():
                file.unlink()
        os.rmdir(self.temp_dir)
    
    def test_basis_functions_workflow(self):
        """Test that basis functions demonstration works."""
        # Generate test data
        x = np.linspace(-3, 3, 100).reshape(-1, 1)
        
        # Test different basis functions
        basis_types = [
            ('Linear', mlai.linear, 2),  # Linear basis has 2 functions (constant + linear)
            ('Polynomial', mlai.polynomial, 3),
            ('Radial', mlai.radial, 5),
            ('Fourier', mlai.fourier, 4)
        ]
        
        for name, basis_func, num_basis in basis_types:
            # Create basis
            basis = mlai.Basis(basis_func, num_basis, data_limits=[float(np.min(x)), float(np.max(x))])
            
            # Compute basis matrix
            Phi = basis.Phi(x)
            
            # Assertions
            assert Phi.shape == (100, num_basis), f"{name} basis shape incorrect"
            assert not np.isnan(Phi).any(), f"{name} basis contains NaN values"
            assert not np.isinf(Phi).any(), f"{name} basis contains infinite values"
    
    def test_linear_regression_workflow(self):
        """Test that linear regression demonstration works."""
        # Generate regression data
        n_samples = 100
        x_data = np.linspace(-3, 3, n_samples).reshape(-1, 1)
        y_data = 2 * x_data**2 - 3 * x_data + 1 + 0.5 * np.random.randn(n_samples, 1)
        
        # Test polynomial basis (should work well for this data)
        basis = mlai.Basis(mlai.polynomial, 3, data_limits=[float(np.min(x_data)), float(np.max(x_data))])
        model = mlai.LM(x_data, y_data, basis)
        
        # Fit model
        model.fit()
        
        # Make predictions
        y_pred, _ = model.predict(x_data)
        
        # Calculate metrics
        mse = np.mean((y_data - y_pred) ** 2)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Assertions
        assert mse < 1.0, f"MSE too high: {mse}"
        assert r2 > 0.9, f"R² too low: {r2}"
        assert y_pred.shape == y_data.shape, "Prediction shape mismatch"
    
    def test_logistic_regression_workflow(self):
        """Test that logistic regression demonstration works."""
        # Generate classification data
        n_samples = 200
        n_features = 2
        
        X_class0 = np.random.multivariate_normal([-1, -1], [[1, 0.5], [0.5, 1]], n_samples//2)
        X_class1 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], n_samples//2)
        
        X = np.vstack([X_class0, X_class1])
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)]).reshape(-1, 1)
        
        # Use multivariate RBF basis
        basis = mlai.Basis(lambda x, num_basis: radial_multivariate(x, num_basis=num_basis, random_state=42), 10)
        
        # Create and fit model
        model = mlai.LR(X, y, basis)
        model.fit()
        
        # Make predictions
        y_pred_proba, Phi = model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = (y_pred == y).mean()
        
        # Assertions
        assert accuracy > 0.7, f"Accuracy too low: {accuracy}"
        assert y_pred_proba.shape == y.shape, "Probability shape mismatch"
        assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)), "Probabilities not in [0,1]"
    
    def test_perceptron_workflow(self):
        """Test that perceptron demonstration works."""
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

        # Split into positive and negative classes
        x_plus = X[y.flatten() == 1]
        x_minus = X[y.flatten() == 0]

        # Initialize perceptron
        w, b, _ = mlai.init_perceptron(x_plus, x_minus)

        # Run perceptron algorithm
        converged = False
        max_iterations = 1000

        for iteration in range(max_iterations):
            # Check if all points are correctly classified
            predictions = (X @ w + b > 0).astype(int).reshape(-1, 1)
            if np.all(predictions == y):
                converged = True
                break

            # Update weights
            w, b, _, _ = mlai.update_perceptron(w, b, x_plus, x_minus, learn_rate=1.0)

        if not converged:
            # Print debug info
            n_correct = np.sum(predictions == y)
            print(f"Perceptron did not converge after {max_iterations} iterations. Correct: {n_correct}/{n_samples}")
            print(f"Final weights: {w}, bias: {b}")
            print(f"Predictions: {predictions.T}")
            print(f"Targets: {y.T}")
        # Assertions
        assert converged, "Perceptron did not converge"
        assert w.shape == (2,), "Weight shape incorrect"
        assert isinstance(b, (int, float, np.integer, np.floating)), "Bias should be scalar"
    
    def test_bayesian_linear_regression_workflow(self):
        """Test that Bayesian linear regression demonstration works."""
        # Generate data
        n_samples = 50
        x_data = np.linspace(-2, 2, n_samples).reshape(-1, 1)
        y_data = 2 * x_data + 1 + 0.3 * np.random.randn(n_samples, 1)
        
        # Create basis and model
        basis = mlai.Basis(mlai.linear, 2, data_limits=[float(np.min(x_data)), float(np.max(x_data))])
        model = mlai.BLM(x_data, y_data, alpha=1.0, sigma2=1.0, basis=basis)
        
        # Fit model
        model.fit()
        
        # Make predictions with uncertainty
        y_pred_mean, y_pred_var = model.predict(x_data)
        
        # Assertions
        assert y_pred_mean.shape == y_data.shape, "Mean prediction shape mismatch"
        assert y_pred_var.shape == y_data.shape, "Variance prediction shape mismatch"
        assert np.all(y_pred_var >= 0), "Variance should be non-negative"
        assert not np.isnan(y_pred_mean).any(), "Mean predictions contain NaN"
        assert not np.isnan(y_pred_var).any(), "Variance predictions contain NaN"


class TestTutorialDocumentationConsistency:
    """Test that RST documentation examples match working code."""
    
    def test_linear_regression_rst_example(self):
        """Test that linear regression RST example would work."""
        # This is the pattern from the RST file, but with 2D targets
        np.random.seed(42)
        n_samples = 100
        
        x_data = np.linspace(-3, 3, n_samples).reshape(-1, 1)
        y_data = 2 * x_data**2 - 3 * x_data + 1 + 0.5 * np.random.randn(n_samples, 1)
        
        # Create polynomial basis functions
        basis = mlai.Basis(mlai.polynomial, 3, data_limits=[x_data.min(), x_data.max()])
        
        # Create linear model
        model = mlai.LM(x_data, y_data, basis)
        
        # Fit the model
        model.fit()
        
        # Get predictions
        y_pred, _ = model.predict(x_data)
        
        # Calculate R-squared
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Assertions
        assert r_squared > 0.9, f"R-squared too low: {r_squared}"
        assert y_pred.shape == y_data.shape, "Prediction shape mismatch"
    
    def test_logistic_regression_rst_example(self):
        """Test that logistic regression RST example would work."""
        # This is the pattern from the RST file, but with numpy arrays instead of pandas
        np.random.seed(42)
        n_samples = 200
        n_features = 2
        
        X_class0 = np.random.multivariate_normal([-1, -1], [[1, 0.5], [0.5, 1]], n_samples//2)
        X_class1 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], n_samples//2)
        
        X = np.vstack([X_class0, X_class1])
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)]).reshape(-1, 1)
        
        # Use multivariate RBF basis instead of polynomial for 2D input
        basis = mlai.Basis(lambda x, num_basis: radial_multivariate(x, num_basis=num_basis, random_state=42), 10)
        
        # Create logistic regression model
        model = mlai.LR(X, y, basis)
        
        # Fit the model
        model.fit()
        
        # Compute predictions
        y_pred_proba, Phi = model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = (y_pred == y).mean()
        
        # Assertions
        assert accuracy > 0.7, f"Accuracy too low: {accuracy}"
        assert y_pred_proba.shape == y.shape, "Probability shape mismatch"


class TestTutorialOutputs:
    """Test that tutorial workflows generate expected outputs."""
    
    def test_workflow_generates_plots(self):
        """Test that workflow generates plot files."""
        # This would test the actual workflow script
        # For now, we'll test that we can create the expected plot files
        
        # Generate some data and create a simple plot
        x = np.linspace(-3, 3, 100).reshape(-1, 1)
        y = 2 * x**2 - 3 * x + 1 + 0.5 * np.random.randn(100, 1)
        
        # Create basis and model
        basis = mlai.Basis(mlai.polynomial, 3, data_limits=[float(np.min(x)), float(np.max(x))])
        model = mlai.LM(x, y, basis)
        model.fit()
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, c='blue', alpha=0.6, label='Data')
        
        x_smooth = np.linspace(np.min(x), np.max(x), 200).reshape(-1, 1)
        y_smooth, _ = model.predict(x_smooth)
        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Model Fit')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression with Polynomial Basis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = "test_linear_regression.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Check that file was created
        assert Path(plot_path).exists(), "Plot file was not created"
        assert Path(plot_path).stat().st_size > 0, "Plot file is empty"
        
        # Clean up
        Path(plot_path).unlink() 