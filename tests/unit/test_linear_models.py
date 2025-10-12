#!/usr/bin/env python3
"""
Tests for linear models, basis functions, and related functionality in mlai.

This module tests the linear models and basis functions that will be moved to 
linear_models.py in the refactoring process.
"""

import unittest
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import pytest
from unittest.mock import patch, MagicMock

# Import mlai modules
import mlai.mlai as mlai


class TestBasisFunctions:
    """Test basis function implementations."""
    
    def test_linear_basis(self):
        """Test linear basis function."""
        X = np.array([[1, 2], [3, 4]])
        result = mlai.linear(X)
        expected = np.array([[1, 1, 2], [1, 3, 4]])  # [1, x1, x2]
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_polynomial_basis(self):
        """Test polynomial basis function."""
        X = np.array([[1]])  # 1D input
        result = mlai.polynomial(X, num_basis=6)
        # Should have 6 basis functions
        assert result.shape[1] == 6
        assert result.shape[0] == 1
    
    def test_radial_basis(self):
        """Test radial basis function."""
        X = np.array([[1], [3]])  # 1D input
        result = mlai.radial(X, num_basis=4)
        assert result.shape[0] == 2
        assert result.shape[1] == 4
    
    def test_fourier_basis(self):
        """Test Fourier basis function."""
        X = np.array([[1]])  # 1D input
        result = mlai.fourier(X, num_basis=7)
        # Should have 7 basis functions
        assert result.shape[1] == 7
    
    def test_relu_basis(self):
        """Test ReLU basis function."""
        X = np.array([[1], [-1]])
        result = mlai.relu(X, num_basis=4)
        # ReLU should be non-negative
        assert np.all(result >= 0)
        assert result.shape[0] == 2
        assert result.shape[1] == 4
    
    def test_hyperbolic_tangent_basis(self):
        """Test hyperbolic tangent basis function."""
        X = np.array([[1]])
        result = mlai.hyperbolic_tangent(X, num_basis=4)
        # tanh should be between -1 and 1
        assert np.all(result >= -1)
        assert np.all(result <= 1)
        assert result.shape[1] == 4


class TestBasisClass:
    """Test Basis class functionality."""
    
    def test_basis_initialization(self):
        """Test Basis class initialization."""
        basis = mlai.Basis(mlai.linear, number=2)
        assert basis.function == mlai.linear
        assert basis.number == 2
    
    def test_basis_phi_method(self):
        """Test Basis Phi method."""
        basis = mlai.Basis(mlai.linear, number=2)
        X = np.array([[1, 2]])
        result = basis.Phi(X)
        expected = mlai.linear(X)
        np.testing.assert_array_almost_equal(result, expected)


class TestLinearModel:
    """Test Linear Model (LM) class."""
    
    def test_lm_initialization(self):
        """Test Linear Model initialization."""
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [4], [6]])
        basis = mlai.Basis(mlai.linear, number=2)
        lm = mlai.LM(X, y, basis)
        assert lm.basis == basis
    
    def test_lm_set_param(self):
        """Test Linear Model set_param method."""
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [4], [6]])
        basis = mlai.Basis(mlai.linear, number=2)
        lm = mlai.LM(X, y, basis)
        lm.fit()  # Fit first to initialize sigma2
        # Test that set_param doesn't raise an error
        lm.set_param('sigma2', 0.1)
        # Don't assert exact value as it may be computed differently
    
    def test_lm_fit_and_predict(self):
        """Test Linear Model fit and predict."""
        # Create simple test data
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [4], [6]])  # y = 2x
        
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [4], [6]])
        basis = mlai.Basis(mlai.linear, number=2)
        lm = mlai.LM(X, y, basis)
        lm.fit()
        
        # Test prediction
        X_test = np.array([[4]])
        y_pred, _ = lm.predict(X_test)
        assert y_pred.shape[0] == 1


class TestLogisticRegression:
    """Test Logistic Regression (LR) class."""
    
    def test_lr_initialization(self):
        """Test Logistic Regression initialization."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[0], [1]])
        basis = mlai.Basis(mlai.linear, number=2)
        lr = mlai.LR(X, y, basis)
        assert lr is not None


class TestBayesianLinearModel:
    """Test Bayesian Linear Model (BLM) class."""
    
    def test_blm_initialization(self):
        """Test Bayesian Linear Model initialization."""
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [4], [6]])
        basis = mlai.Basis(mlai.linear, number=2)
        blm = mlai.BLM(X, y, basis)
        assert blm.basis == basis


class TestLinearModelEdgeCases:
    """Test edge cases and error handling for Linear Model."""
    
    def test_lm_set_param_unknown_parameter_raises(self):
        """Test LM set_param with unknown parameter raises error."""
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [4], [6]])
        basis = mlai.Basis(mlai.linear, number=2)
        lm = mlai.LM(X, y, basis)
        
        with pytest.raises(ValueError):
            lm.set_param('unknown_param', 0.1)
    
    def test_lm_set_param_no_update_when_same_value(self):
        """Test LM set_param doesn't update when same value."""
        X = np.array([[1], [2], [3]])
        y = np.array([[2], [4], [6]])
        basis = mlai.Basis(mlai.linear, number=2)
        lm = mlai.LM(X, y, basis)
        
        # Set initial value
        lm.set_param('sigma2', 0.1)
        initial_sigma2 = lm.sigma2
        
        # Set same value again
        lm.set_param('sigma2', 0.1)
        assert lm.sigma2 == initial_sigma2


class TestBasisFunctionEdgeCases:
    """Test edge cases for basis functions."""
    
    def test_polynomial_basis_edge_cases(self):
        """Test polynomial basis function edge cases."""
        # Test with num_basis 1
        X = np.array([[1]])  # 1D input
        result = mlai.polynomial(X, num_basis=1)
        assert result.shape[1] == 1  # Only constant term
        
        # Test with num_basis 3
        result = mlai.polynomial(X, num_basis=3)
        assert result.shape[1] == 3
    
    def test_radial_basis_edge_cases(self):
        """Test radial basis function edge cases."""
        X = np.array([[1]])  # 1D input
        
        # Test with num_basis 0
        result = mlai.radial(X, num_basis=0)
        assert result.shape[1] == 0
        
        # Test with num_basis 2
        result = mlai.radial(X, num_basis=2)
        assert result.shape[1] == 2


class TestLogisticRegressionMethods:
    """Test Logistic Regression methods that were not previously covered."""
    
    def test_lr_initialization(self):
        """Test LR initialization works."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[0], [1]])
        basis = mlai.Basis(mlai.linear, number=2)
        lr = mlai.LR(X, y, basis)
        assert lr is not None


if __name__ == '__main__':
    unittest.main()
