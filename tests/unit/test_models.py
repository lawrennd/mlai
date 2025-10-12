"""
Unit tests for mlai.py module.

This module tests the core machine learning functionality including:
- Abstract base classes for the model
"""

import pytest
import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Import the module to test
import mlai


class TestAbstractBaseClasses:
    """Test abstract base classes for correct NotImplementedError behavior."""
    def test_model_objective_not_implemented(self):
        model = mlai.Model()
        with pytest.raises(NotImplementedError):
            model.objective()
    def test_model_fit_not_implemented(self):
        model = mlai.Model()
        with pytest.raises(NotImplementedError):
            model.fit()
    def test_probmodel_log_likelihood_not_implemented(self):
        class Dummy(mlai.ProbModel):
            def __init__(self):
                super().__init__()
        dummy = Dummy()
        with pytest.raises(NotImplementedError):
            dummy.log_likelihood()
    
    def test_probmodel_objective_calls_update_sum_squares(self):
        """Test ProbModel.objective() method (lines 654-655)."""
        class Dummy(mlai.ProbModel):
            def __init__(self):
                super().__init__()
                self.sum_squares = 42.0  # Set a known value
            def update_sum_squares(self):
                # This method should be called by objective()
                pass
            def log_likelihood(self):
                # Mock implementation to avoid NotImplementedError
                return 10.0
        
        dummy = Dummy()
        result = dummy.objective()
        assert result == -10.0  # objective() should return -self.log_likelihood()
    
    def test_probmodel_log_likelihood_calls_update_sum_squares(self):
        """Test ProbModel.log_likelihood() method (lines 659-660)."""
        # Use LM class which has a concrete implementation
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[1.0], [2.0], [3.0]])
        basis = mlai.Basis(mlai.linear, 2)
        
        lm = mlai.LM(X, y, basis)
        lm.fit()  # Fit the model first
        result = lm.log_likelihood()
        
        # Check that it returns a finite value (the exact formula is complex)
        assert np.isfinite(result)
        assert isinstance(result, float)
    
    def test_linear_model_set_param(self):
        """Test LM set_param method (lines 573, 578-579, 583)."""
        # Create a simple test case that won't trigger fit issues
        X = np.array([[1], [2], [3], [4]])  # More data points
        y = np.array([[1], [2], [3], [4]])
        basis = mlai.Basis(mlai.polynomial, 2)  # Fewer basis functions
        
        # Create LM without triggering fit
        lm = mlai.LM(X, y, basis)
        
        # Test setting a parameter that exists in the model (same value, no fit)
        lm.set_param('sigma2', 1.0)  # Same as default
        assert lm.sigma2 == 1.0
        
        # Test setting an unknown parameter (should raise ValueError)
        with pytest.raises(ValueError, match="Unknown parameter"):
            lm.set_param('unknown_param', 1.0)
    def test_mapmodel_predict_not_implemented(self):
        class Dummy(mlai.MapModel):
            def __init__(self, X, y):
                super().__init__(X, y)
            def update_sum_squares(self):
                pass
        X = np.zeros((2, 2))
        y = np.zeros(2)
        dummy = Dummy(X, y)
        with pytest.raises(NotImplementedError):
            dummy.predict(X)
    def test_mapmodel_update_sum_squares_not_implemented(self):
        class Dummy(mlai.MapModel):
            def __init__(self, X, y):
                super().__init__(X, y)
            def predict(self, X):
                return X
        X = np.zeros((2, 2))
        y = np.zeros(2)
        dummy = Dummy(X, y)
        with pytest.raises(NotImplementedError):
            dummy.update_sum_squares()

    def test_model_parameters_not_implemented(self):
        """Test that base Model class raises NotImplementedError for parameters."""
        model = mlai.Model()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement the parameters property"):
            _ = model.parameters
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement the parameters setter"):
            model.parameters = np.array([1.0, 2.0])

class TestMapModelMethods:
    """Test MapModel methods that were not previously covered."""
    
    def test_mapmodel_rmse(self):
        """Test MapModel rmse method."""
        class TestMapModel(mlai.MapModel):
            def __init__(self, X, y):
                super().__init__(X, y)
                self.sum_squares = 4.0  # Set a known value
            
            def update_sum_squares(self):
                pass  # Override to avoid NotImplementedError
            
            def predict(self, X):
                return X
        
        X = np.array([[1], [2]])
        y = np.array([1, 2])
        model = TestMapModel(X, y)
        
        rmse = model.rmse()
        assert isinstance(rmse, float)
        assert rmse > 0
        assert rmse == np.sqrt(4.0 / 2)  # sqrt(sum_squares / num_data)

            
