#!/usr/bin/env python3
"""
Tests for loss functions in mlai.

This module tests the loss functions that have been moved to loss.py.
"""

import unittest
import numpy as np

# Import the module to test
import mlai


class TestLossFunctions(unittest.TestCase):
    """Test suite for loss functions."""
    
    def test_loss_functions_with_finite_differences(self):
        """Test loss function gradients using finite differences."""
        from mlai import (
            MeanSquaredError, MeanAbsoluteError, HuberLoss, 
            BinaryCrossEntropyLoss, CrossEntropyLoss,
            finite_difference_gradient, verify_gradient_implementation
        )
        
        # Test data
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_true = np.array([[1.1, 2.1], [2.9, 4.1], [5.1, 5.9]])
        
        # Test Mean Squared Error
        mse_loss = MeanSquaredError()
        def mse_func(pred):
            return mse_loss.forward(pred.reshape(y_pred.shape), y_true)
        
        numerical_grad = finite_difference_gradient(mse_func, y_pred.flatten())
        analytical_grad = mse_loss.gradient(y_pred, y_true).flatten()
        self.assertTrue(verify_gradient_implementation(analytical_grad, numerical_grad))
        
        # Test Mean Absolute Error
        mae_loss = MeanAbsoluteError()
        def mae_func(pred):
            return mae_loss.forward(pred.reshape(y_pred.shape), y_true)
        
        numerical_grad = finite_difference_gradient(mae_func, y_pred.flatten())
        analytical_grad = mae_loss.gradient(y_pred, y_true).flatten()
        self.assertTrue(verify_gradient_implementation(analytical_grad, numerical_grad))
        
        # Test Huber Loss
        huber_loss = HuberLoss(delta=1.0)
        def huber_func(pred):
            return huber_loss.forward(pred.reshape(y_pred.shape), y_true)
        
        numerical_grad = finite_difference_gradient(huber_func, y_pred.flatten())
        analytical_grad = huber_loss.gradient(y_pred, y_true).flatten()
        self.assertTrue(verify_gradient_implementation(analytical_grad, numerical_grad))
        
        # Test Binary Cross Entropy (single output column)
        bce_loss = BinaryCrossEntropyLoss()
        y_pred_bce = np.array([[0.8], [0.3], [0.9]])
        y_true_bce = np.array([[1.0], [0.0], [1.0]])
        
        def bce_func(pred):
            return bce_loss.forward(pred.reshape(-1, 1), y_true_bce)
        
        numerical_grad = finite_difference_gradient(bce_func, y_pred_bce.flatten())
        analytical_grad = bce_loss.gradient(y_pred_bce, y_true_bce).flatten()
        self.assertTrue(verify_gradient_implementation(analytical_grad, numerical_grad))

        # Multi-column BCE: forward uses np.mean over all elements, so the
        # gradient must divide by predictions.size (not just n_samples).
        y_pred_bce_multi = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.4]])
        y_true_bce_multi = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

        def bce_multi_func(pred):
            return bce_loss.forward(pred.reshape(y_pred_bce_multi.shape), y_true_bce_multi)

        numerical_grad = finite_difference_gradient(bce_multi_func, y_pred_bce_multi.flatten())
        analytical_grad = bce_loss.gradient(y_pred_bce_multi, y_true_bce_multi).flatten()
        self.assertTrue(verify_gradient_implementation(analytical_grad, numerical_grad))
        
        # Test Cross Entropy
        ce_loss = CrossEntropyLoss()
        y_pred_ce = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        y_true_ce = np.array([[0, 1], [1, 0], [0, 1]])
        
        def ce_func(pred):
            return ce_loss.forward(pred.reshape(-1, 2), y_true_ce)
        
        numerical_grad = finite_difference_gradient(ce_func, y_pred_ce.flatten())
        analytical_grad = ce_loss.gradient(y_pred_ce, y_true_ce).flatten()
        self.assertTrue(verify_gradient_implementation(analytical_grad, numerical_grad))


class TestActivationNumericalStability(unittest.TestCase):
    """Test activation functions at extreme input values."""

    def test_sigmoid_and_soft_relu_large_inputs(self):
        from mlai import SigmoidActivation, SoftReLUActivation

        x = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
        sigmoid = SigmoidActivation()
        soft_relu = SoftReLUActivation()

        s = sigmoid.forward(x)
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertAlmostEqual(s[0], 0.0)
        self.assertAlmostEqual(s[-1], 1.0)

        sr = soft_relu.forward(x)
        self.assertTrue(np.all(np.isfinite(sr)))
        # softplus(x) ≈ x for large positive x; ≈ 0 for large negative x
        self.assertAlmostEqual(sr[0], 0.0, places=10)
        self.assertAlmostEqual(sr[-1], 1000.0, places=6)

        self.assertTrue(np.all(np.isfinite(sigmoid.gradient(x))))
        self.assertTrue(np.all(np.isfinite(soft_relu.gradient(x))))


if __name__ == '__main__':
    unittest.main()

