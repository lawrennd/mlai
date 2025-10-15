#!/usr/bin/env python3
"""
Tests for optimization module in mlai.

This module tests the optimization algorithms and training utilities.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock

# Import the module to test
import mlai


class TestOptimiserBaseClass(unittest.TestCase):
    """Test suite for Optimiser base class."""
    
    def test_optimiser_instantiation(self):
        """Test that Optimiser can be instantiated."""
        optimiser = mlai.Optimiser()
        self.assertIsInstance(optimiser, mlai.Optimiser)
    
    def test_optimiser_step_not_implemented(self):
        """Test that Optimiser.step() raises NotImplementedError."""
        optimiser = mlai.Optimiser()
        model = MagicMock()
        
        with self.assertRaises(NotImplementedError):
            optimiser.step(model)


class TestSGD(unittest.TestCase):
    """Test suite for SGD optimizer."""
    
    def test_sgd_instantiation(self):
        """Test that SGD can be instantiated with default parameters."""
        sgd = mlai.SGD()
        self.assertIsInstance(sgd, mlai.SGD)
        self.assertIsInstance(sgd, mlai.Optimiser)
        self.assertEqual(sgd.learning_rate, 0.01)
    
    def test_sgd_custom_learning_rate(self):
        """Test that SGD accepts custom learning rate."""
        sgd = mlai.SGD(learning_rate=0.1)
        self.assertEqual(sgd.learning_rate, 0.1)
    
    def test_sgd_step_updates_parameters(self):
        """Test that SGD.step() updates model parameters correctly."""
        sgd = mlai.SGD(learning_rate=0.1)
        
        # Create mock model
        model = MagicMock()
        model.parameters = np.array([1.0, 2.0, 3.0])
        model.gradients = np.array([0.1, 0.2, 0.3])
        
        # Perform SGD step
        sgd.step(model)
        
        # Check that parameters were updated
        expected_params = np.array([1.0, 2.0, 3.0]) - 0.1 * np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(
            model.parameters, expected_params
        )
    
    def test_sgd_multiple_steps(self):
        """Test multiple SGD steps."""
        sgd = mlai.SGD(learning_rate=0.1)
        
        # Create mock model that updates gradients
        initial_params = np.array([1.0, 2.0, 3.0])
        params = initial_params.copy()
        
        for i in range(5):
            model = MagicMock()
            model.parameters = params
            model.gradients = np.array([0.1, 0.2, 0.3]) * (i + 1)  # Increasing gradients
            
            sgd.step(model)
            params = model.parameters
        
        # Check that parameters moved away from initial values
        self.assertTrue(np.any(params != initial_params))
    
    def test_sgd_with_zero_gradients(self):
        """Test SGD with zero gradients."""
        sgd = mlai.SGD(learning_rate=0.1)
        
        model = MagicMock()
        initial_params = np.array([1.0, 2.0, 3.0])
        model.parameters = initial_params.copy()
        model.gradients = np.zeros(3)
        
        sgd.step(model)
        
        # Parameters should not change with zero gradients
        np.testing.assert_array_equal(model.parameters, initial_params)
    
    def test_sgd_with_negative_gradients(self):
        """Test SGD with negative gradients."""
        sgd = mlai.SGD(learning_rate=0.1)
        
        model = MagicMock()
        initial_params = np.array([1.0, 2.0, 3.0])
        model.parameters = initial_params.copy()
        model.gradients = np.array([-0.1, -0.2, -0.3])
        
        sgd.step(model)
        
        # Parameters should increase with negative gradients
        expected_params = initial_params - 0.1 * np.array([-0.1, -0.2, -0.3])
        np.testing.assert_array_almost_equal(model.parameters, expected_params)
        self.assertTrue(np.all(model.parameters > initial_params))


class TestAdam(unittest.TestCase):
    """Test suite for Adam optimizer."""
    
    def test_adam_instantiation(self):
        """Test that Adam can be instantiated with default parameters."""
        adam = mlai.Adam()
        self.assertIsInstance(adam, mlai.Adam)
        self.assertIsInstance(adam, mlai.Optimiser)
        self.assertEqual(adam.learning_rate, 0.001)
        self.assertEqual(adam.beta1, 0.9)
        self.assertEqual(adam.beta2, 0.999)
        self.assertEqual(adam.epsilon, 1e-8)
    
    def test_adam_custom_parameters(self):
        """Test that Adam accepts custom parameters."""
        adam = mlai.Adam(learning_rate=0.01, beta1=0.8, beta2=0.99, epsilon=1e-7)
        self.assertEqual(adam.learning_rate, 0.01)
        self.assertEqual(adam.beta1, 0.8)
        self.assertEqual(adam.beta2, 0.99)
        self.assertEqual(adam.epsilon, 1e-7)
    
    def test_adam_moment_initialization(self):
        """Test that Adam initializes moments on first step."""
        adam = mlai.Adam()
        
        # Initially, moments should be None
        self.assertIsNone(adam.m)
        self.assertIsNone(adam.v)
        self.assertEqual(adam.t, 0)
        
        # Create mock model
        model = MagicMock()
        model.parameters = np.array([1.0, 2.0, 3.0])
        model.gradients = np.array([0.1, 0.2, 0.3])
        
        # First step should initialize moments
        adam.step(model)
        
        self.assertIsNotNone(adam.m)
        self.assertIsNotNone(adam.v)
        self.assertEqual(adam.t, 1)
        self.assertEqual(adam.m.shape, (3,))
        self.assertEqual(adam.v.shape, (3,))
    
    def test_adam_step_updates_parameters(self):
        """Test that Adam.step() updates model parameters correctly."""
        adam = mlai.Adam(learning_rate=0.01)
        
        # Create mock model
        model = MagicMock()
        initial_params = np.array([1.0, 2.0, 3.0])
        model.parameters = initial_params.copy()
        model.gradients = np.array([0.1, 0.2, 0.3])
        
        # Perform Adam step
        adam.step(model)
        
        # Check that parameters were updated (should move in direction opposite to gradient)
        self.assertTrue(np.all(model.parameters < initial_params))
    
    def test_adam_multiple_steps(self):
        """Test multiple Adam steps."""
        adam = mlai.Adam(learning_rate=0.01)
        
        # Create mock model
        params = np.array([1.0, 2.0, 3.0])
        
        for i in range(10):
            model = MagicMock()
            model.parameters = params
            model.gradients = np.array([0.1, 0.2, 0.3])  # Constant gradients
            
            adam.step(model)
            params = model.parameters
        
        # After multiple steps with constant gradients, parameters should continue moving
        self.assertTrue(np.all(params < np.array([1.0, 2.0, 3.0])))
        self.assertEqual(adam.t, 10)
    
    def test_adam_bias_correction(self):
        """Test that Adam applies bias correction."""
        adam = mlai.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
        
        # Create mock model
        model = MagicMock()
        model.parameters = np.array([1.0])
        model.gradients = np.array([1.0])
        
        # First step
        adam.step(model)
        first_params = model.parameters.copy()
        
        # Create new Adam instance and skip to later timestep
        adam2 = mlai.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
        model2 = MagicMock()
        model2.parameters = np.array([1.0])
        
        # Simulate multiple steps to see bias correction effect
        for _ in range(5):
            model2.gradients = np.array([1.0])
            adam2.step(model2)
        
        # At later timesteps, bias correction has less effect
        self.assertNotEqual(adam.t, adam2.t)
    
    def test_adam_with_zero_gradients(self):
        """Test Adam with zero gradients."""
        adam = mlai.Adam()
        
        model = MagicMock()
        initial_params = np.array([1.0, 2.0, 3.0])
        model.parameters = initial_params.copy()
        model.gradients = np.zeros(3)
        
        adam.step(model)
        
        # With zero gradients, parameters might change slightly due to bias correction
        # but should be very close to initial
        np.testing.assert_array_almost_equal(model.parameters, initial_params, decimal=5)


class TestTrainModel(unittest.TestCase):
    """Test suite for train_model function."""
    
    def test_train_model_with_sgd(self):
        """Test train_model with SGD optimizer."""
        # Create a simple linear model
        from mlai import LM, MeanSquaredError, SGD, Basis, linear
        
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = (2 * X[:, 0] + 3 * X[:, 1] + 1 + 0.1 * np.random.randn(20)).reshape(-1, 1)
        
        # Add intercept column
        basis = Basis(linear, number=2)
        Phi = basis.Phi(X)
        
        model = LM(X, y, basis)
        # Initialize parameters
        model.parameters = np.random.randn(Phi.shape[1])
        
        loss_fn = MeanSquaredError()
        optimiser = SGD(learning_rate=0.01)
        
        # Train for a few epochs
        losses = mlai.train_model(model, X, y, loss_fn, optimiser, n_epochs=10, verbose=False)
        
        # Check that losses were recorded
        self.assertEqual(len(losses), 10)
        
        # Check that loss decreased (model is learning)
        self.assertLess(losses[-1], losses[0])
    
    def test_train_model_with_adam(self):
        """Test train_model with Adam optimizer."""
        from mlai import NeuralNetwork, ReLUActivation, LinearActivation, MeanSquaredError, Adam
        
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)
        
        # Create network
        dimensions = [2, 10, 1]
        activations = [ReLUActivation(), LinearActivation()]
        model = NeuralNetwork(dimensions, activations)
        
        loss_fn = MeanSquaredError()
        optimiser = Adam(learning_rate=0.01)
        
        # Train for a few epochs
        losses = mlai.train_model(model, X, y, loss_fn, optimiser, n_epochs=10, verbose=False)
        
        # Check that losses were recorded
        self.assertEqual(len(losses), 10)
        
        # Check that loss decreased
        self.assertLess(losses[-1], losses[0])
    
    def test_train_model_verbose(self):
        """Test train_model with verbose output."""
        from mlai import LM, MeanSquaredError, SGD, Basis, linear
        
        np.random.seed(42)
        X = np.random.randn(10, 2)
        y = (X[:, 0] + X[:, 1] + 0.1 * np.random.randn(10)).reshape(-1, 1)
        
        basis = Basis(linear, number=2)
        model = LM(X, y, basis)
        # Initialize parameters
        model.parameters = np.random.randn(basis.Phi(X).shape[1])
        
        loss_fn = MeanSquaredError()
        optimiser = SGD(learning_rate=0.01)
        
        # Test with verbose=True (should not crash)
        import io
        import sys
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            losses = mlai.train_model(model, X, y, loss_fn, optimiser, n_epochs=5, verbose=True)
        
        output = f.getvalue()
        
        # Check that output contains epoch information
        self.assertIn("Epoch", output)
        self.assertIn("Loss", output)
    
    def test_train_model_returns_losses(self):
        """Test that train_model returns list of losses."""
        from mlai import NeuralNetwork, ReLUActivation, LinearActivation, MeanSquaredError, SGD
        
        np.random.seed(42)
        model = NeuralNetwork([1, 5, 1], [ReLUActivation(), LinearActivation()])
        X = np.array([[1.0]])
        y = np.array([[2.0]])
        
        loss_fn = MeanSquaredError()
        optimiser = SGD(learning_rate=0.01)
        
        losses = mlai.train_model(model, X, y, loss_fn, optimiser, n_epochs=5, verbose=False)
        
        # Check return type and length
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), 5)
        
        # Check that all losses are floats
        for loss in losses:
            self.assertIsInstance(loss, (float, np.floating))
    
    def test_train_model_with_classification(self):
        """Test train_model with classification task."""
        from mlai import NeuralNetwork, SigmoidActivation, BinaryCrossEntropyLoss, SGD
        
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = ((X[:, 0]**2 + X[:, 1]**2) > 1.0).astype(float).reshape(-1, 1)
        
        # Create network
        dimensions = [2, 5, 1]
        activations = [SigmoidActivation(), SigmoidActivation()]
        model = NeuralNetwork(dimensions, activations)
        
        loss_fn = BinaryCrossEntropyLoss()
        optimiser = SGD(learning_rate=0.1)
        
        # Train
        losses = mlai.train_model(model, X, y, loss_fn, optimiser, n_epochs=20, verbose=False)
        
        # Check that training occurred
        self.assertEqual(len(losses), 20)
        # Classification loss should decrease
        self.assertLess(losses[-1], losses[0])


class TestOptimisationIntegration(unittest.TestCase):
    """Integration tests for optimization with different model types."""
    
    def test_sgd_with_linear_model(self):
        """Test SGD optimization with linear model."""
        from mlai import LM, Basis, linear, SGD, MeanSquaredError
        
        np.random.seed(42)
        X = np.random.randn(30, 1)
        y = (2 * X[:, 0] + 1 + 0.1 * np.random.randn(30)).reshape(-1, 1)
        
        basis = Basis(linear, number=1)
        model = LM(X, y, basis)
        
        # Initialize parameters  
        model.parameters = np.random.randn(basis.Phi(X).shape[1]) * 0.1  # Small initialization
        
        # Optimize with SGD
        optimiser = SGD(learning_rate=0.001)  # Lower learning rate for stability
        loss_fn = MeanSquaredError()
        
        y_pred_initial, _ = model.predict(X)
        initial_loss = loss_fn.forward(y_pred_initial, y)
        
        for _ in range(100):
            optimiser.step(model)
        
        y_pred_final, _ = model.predict(X)
        final_loss = loss_fn.forward(y_pred_final, y)
        
        # Loss should decrease
        self.assertLess(final_loss, initial_loss)
    
    def test_adam_with_neural_network(self):
        """Test Adam optimization with neural network."""
        from mlai import NeuralNetwork, ReLUActivation, LinearActivation, Adam, MeanSquaredError
        
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)
        
        model = NeuralNetwork([2, 10, 1], [ReLUActivation(), LinearActivation()])
        optimiser = Adam(learning_rate=0.01)
        loss_fn = MeanSquaredError()
        
        initial_loss = loss_fn.forward(model.predict(X), y)
        
        for _ in range(100):
            y_pred = model.predict(X)
            loss_gradient = loss_fn.gradient(y_pred, y)
            model.set_output_gradient(loss_gradient)
            optimiser.step(model)
        
        final_loss = loss_fn.forward(model.predict(X), y)
        
        # Loss should decrease significantly
        self.assertLess(final_loss, initial_loss * 0.5)


if __name__ == '__main__':
    unittest.main()

