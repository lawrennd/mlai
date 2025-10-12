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
import mlai.mlai as mlai


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
    
    def test_probmodel_objective_calls_log_likelihood(self):
        """Test ProbModel.objective() method (line 377)."""
        class Dummy(mlai.ProbModel):
            def __init__(self):
                super().__init__()
            def log_likelihood(self):
                return 42.0  # Return a known value
        
        dummy = Dummy()
        result = dummy.objective()
        assert result == -42.0  # objective() should return -log_likelihood()
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

            
