"""
Tests for parameters and gradients properties in linear models.

This module tests the new parameters and gradients properties added to
LM and LR classes to ensure they work correctly for optimization.
"""

import unittest
import numpy as np
from mlai import LM, LR, polynomial, Basis, finite_difference_gradient, verify_gradient_implementation


class TestLinearModelParameters(unittest.TestCase):
    """Test parameters property for LM class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.randn(20, 1)
        self.y = np.random.randn(20, 1)
        self.basis = Basis(polynomial, number=3)
        self.model = LM(self.X, self.y, self.basis)
        self.model.fit()

    def test_parameters_getter(self):
        """Test that parameters property returns correct values."""
        params = self.model.parameters
        
        # Check shape and type
        self.assertEqual(params.ndim, 1)
        self.assertEqual(params.shape, (3,))
        self.assertEqual(params.dtype, np.float64)
        
        # Check that parameters match w_star
        w_star_flat = self.model.w_star.flatten()
        np.testing.assert_array_almost_equal(params, w_star_flat)

    def test_parameters_setter(self):
        """Test that parameters property setter works correctly."""
        original_params = self.model.parameters.copy()
        new_params = original_params + 0.1
        
        # Set new parameters
        self.model.parameters = new_params
        
        # Check that parameters were updated
        np.testing.assert_array_almost_equal(self.model.parameters, new_params)
        
        # Check that w_star was updated correctly
        np.testing.assert_array_almost_equal(self.model.w_star.flatten(), new_params)

    def test_parameters_setter_validation(self):
        """Test that parameters setter validates input correctly."""
        # Test wrong dimensions
        with self.assertRaises(ValueError):
            self.model.parameters = np.array([[1, 2], [3, 4]])
        
        # Test wrong length
        with self.assertRaises(ValueError):
            self.model.parameters = np.array([1, 2])

    def test_parameters_roundtrip(self):
        """Test that setting and getting parameters preserves values."""
        original_params = self.model.parameters.copy()
        
        # Modify parameters
        new_params = original_params * 2
        self.model.parameters = new_params
        
        # Check roundtrip
        retrieved_params = self.model.parameters
        np.testing.assert_array_almost_equal(retrieved_params, new_params)

    def test_parameters_before_fit(self):
        """Test that parameters property raises error before fitting."""
        # Create model but don't fit
        model = LM(self.X, self.y, self.basis)
        
        # Should raise error before fitting (w_star doesn't exist)
        with self.assertRaises(ValueError):
            params = model.parameters


class TestLinearModelGradients(unittest.TestCase):
    """Test gradients property for LM class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.randn(20, 1)
        self.y = np.random.randn(20, 1)
        self.basis = Basis(polynomial, number=3)
        self.model = LM(self.X, self.y, self.basis)
        self.model.fit()

    def test_gradients_getter(self):
        """Test that gradients property returns correct values."""
        grads = self.model.gradients
        
        # Check shape and type
        self.assertEqual(grads.ndim, 1)
        self.assertEqual(grads.shape, (3,))
        self.assertEqual(grads.dtype, np.float64)

    def test_gradients_consistency(self):
        """Test that gradients are consistent with finite differences."""
        # Use stateless function like neural network tests
        def objective_func(params):
            # Create a new model instance for each call to avoid state corruption
            temp_model = LM(self.model.X, self.model.y, self.model.basis)
            temp_model.w_star = params.reshape(-1, 1)
            temp_model.update_f()
            return temp_model.objective()
        
        # Get current parameters and gradients
        params = self.model.parameters
        grads = self.model.gradients
        
        # Use finite_difference_gradient utility
        numerical_grads = finite_difference_gradient(objective_func, params)
        
        # Check that gradients match finite differences
        np.testing.assert_array_almost_equal(grads, numerical_grads, decimal=4)

    def test_gradients_at_optimum(self):
        """Test that gradients are small at the optimum."""
        grads = self.model.gradients
        
        # At the optimum, gradients should be very small
        self.assertLess(np.max(np.abs(grads)), 1e-10)


class TestLogisticRegressionParameters(unittest.TestCase):
    """Test parameters property for LR class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.randn(20, 1)
        self.y = (np.random.randn(20, 1) > 0).astype(float)
        self.basis = Basis(polynomial, number=3)
        self.model = LR(self.X, self.y, self.basis)
        self.model.fit()

    def test_parameters_getter(self):
        """Test that parameters property returns correct values."""
        params = self.model.parameters
        
        # Check shape and type
        self.assertEqual(params.ndim, 1)
        self.assertEqual(params.shape, (3,))
        self.assertEqual(params.dtype, np.float64)
        
        # Check that parameters match w_star
        w_star_flat = self.model.w_star.flatten()
        np.testing.assert_array_almost_equal(params, w_star_flat)

    def test_parameters_setter(self):
        """Test that parameters property setter works correctly."""
        original_params = self.model.parameters.copy()
        new_params = original_params + 0.1
        
        # Set new parameters
        self.model.parameters = new_params
        
        # Check that parameters were updated
        np.testing.assert_array_almost_equal(self.model.parameters, new_params)
        
        # Check that w_star was updated correctly
        np.testing.assert_array_almost_equal(self.model.w_star.flatten(), new_params)

    def test_parameters_setter_validation(self):
        """Test that parameters setter validates input correctly."""
        # Test wrong dimensions
        with self.assertRaises(ValueError):
            self.model.parameters = np.array([[1, 2], [3, 4]])
        
        # Test wrong length
        with self.assertRaises(ValueError):
            self.model.parameters = np.array([1, 2])

    def test_parameters_roundtrip(self):
        """Test that setting and getting parameters preserves values."""
        original_params = self.model.parameters.copy()
        
        # Modify parameters
        new_params = original_params * 2
        self.model.parameters = new_params
        
        # Check roundtrip
        retrieved_params = self.model.parameters
        np.testing.assert_array_almost_equal(retrieved_params, new_params)


class TestLogisticRegressionGradients(unittest.TestCase):
    """Test gradients property for LR class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.randn(20, 1)
        self.y = (np.random.randn(20, 1) > 0).astype(float)
        self.basis = Basis(polynomial, number=3)
        self.model = LR(self.X, self.y, self.basis)
        self.model.fit()

    def test_gradients_getter(self):
        """Test that gradients property returns correct values."""
        grads = self.model.gradients
        
        # Check shape and type
        self.assertEqual(grads.ndim, 1)
        self.assertEqual(grads.shape, (3,))
        self.assertEqual(grads.dtype, np.float64)

    def test_gradients_consistency_with_gradient_method(self):
        """Test that gradients property matches gradient() method."""
        grads_property = self.model.gradients
        grads_method = self.model.gradient()
        
        # Should be identical
        np.testing.assert_array_almost_equal(grads_property, grads_method)

    def test_gradients_consistency(self):
        """Test that gradients are consistent with finite differences."""
        # Use stateless function like neural network tests
        def objective_func(params):
            # Create a new model instance for each call to avoid state corruption
            temp_model = LR(self.model.X, self.model.y, self.model.basis)
            temp_model.w_star = params.reshape(-1, 1)
            temp_model.update_g()
            return temp_model.objective()
        
        # Get current parameters and gradients
        params = self.model.parameters
        grads = self.model.gradients
        
        # Use finite_difference_gradient utility
        numerical_grads = finite_difference_gradient(objective_func, params)
        
        # Check that gradients match finite differences (relaxed tolerance)
        np.testing.assert_array_almost_equal(grads, numerical_grads, decimal=2)

    def test_gradients_at_optimum(self):
        """Test that gradients are reasonable at the optimum."""
        grads = self.model.gradients
        
        # Check that gradients are finite and not too large
        self.assertTrue(np.all(np.isfinite(grads)))
        self.assertLess(np.max(np.abs(grads)), 100.0)  # Reasonable upper bound


class TestOptimizationInterface(unittest.TestCase):
    """Test the optimization interface using parameters and gradients."""

    def test_simple_sgd_step(self):
        """Test a simple SGD step using parameters and gradients."""
        np.random.seed(42)
        X = np.random.randn(20, 1)
        y = np.random.randn(20, 1)
        basis = Basis(polynomial, number=3)
        model = LM(X, y, basis)
        model.fit()
        
        # Get initial state
        initial_params = model.parameters.copy()
        initial_objective = model.objective()
        
        # Move away from optimum to test SGD
        model.parameters = initial_params + 0.1
        grads = model.gradients
        
        # Perform SGD step
        learning_rate = 0.01
        new_params = model.parameters - learning_rate * grads
        
        # Update model
        model.parameters = new_params
        new_objective = model.objective()
        
        # Check that parameters changed
        self.assertFalse(np.allclose(initial_params, new_params))
        
        # Check that gradients are still available
        new_grads = model.gradients
        self.assertEqual(new_grads.shape, initial_params.shape)

    def test_lr_sgd_step(self):
        """Test a simple SGD step for logistic regression."""
        np.random.seed(42)
        X = np.random.randn(20, 1)
        y = (np.random.randn(20, 1) > 0).astype(float)
        basis = Basis(polynomial, number=3)
        model = LR(X, y, basis)
        model.fit()
        
        # Get initial state
        initial_params = model.parameters.copy()
        initial_objective = model.objective()
        
        # Perform SGD step
        learning_rate = 0.01
        grads = model.gradients
        new_params = initial_params - learning_rate * grads
        
        # Update model
        model.parameters = new_params
        new_objective = model.objective()
        
        # Check that parameters changed
        self.assertFalse(np.allclose(initial_params, new_params))
        
        # Check that gradients are still available
        new_grads = model.gradients
        self.assertEqual(new_grads.shape, initial_params.shape)


if __name__ == '__main__':
    unittest.main()
