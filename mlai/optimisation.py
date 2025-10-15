"""
Optimisation Module

This module contains optimisation utilities including:
- Gradient-based optimisation algorithms (SGD, Adam)
- Training utilities

This provides a unified interface for training models with different optimisers.
"""

import numpy as np

__all__ = [
    'Optimiser',
    'SGD',
    'Adam',
    'train_model'
]


class Optimiser:
    """
    Base class for optimisation algorithms.
    
    This class defines the interface that all optimisers must implement.
    optimisers update model parameters based on gradients computed from
    the loss function.
    
    Methods
    -------
    step(model)
        Perform one optimisation step, updating the model's parameters
    
    Examples
    --------
    >>> optimiser = SGD(learning_rate=0.01)
    >>> optimiser.step(model)  # Updates model.parameters
    """
    
    def __init__(self):
        pass
    
    def step(self, model):
        """
        Perform one optimisation step.
        
        :param model: Model with parameters and gradients properties
        :type model: Model
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclass must implement step()")


class SGD(Optimiser):
    """
    Stochastic Gradient Descent optimiser.
    
    Updates parameters using the rule:
    parameters = parameters - learning_rate * gradients
    
    Parameters
    ----------
    learning_rate : float
        Step size for parameter updates (default: 0.01)
    
    Examples
    --------
    >>> optimiser = SGD(learning_rate=0.01)
    >>> model.set_output_gradient(loss_gradient)
    >>> optimiser.step(model)
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize SGD optimiser.
        
        :param learning_rate: Learning rate for parameter updates
        :type learning_rate: float
        """
        super().__init__()
        self.learning_rate = learning_rate
    
    def step(self, model):
        """
        Perform one SGD optimisation step.
        
        :param model: Model with parameters and gradients properties
        :type model: Model
        """
        # Get current parameters and gradients
        params = model.parameters
        grads = model.gradients
        
        # Update parameters
        new_params = params - self.learning_rate * grads
        model.parameters = new_params


class Adam(Optimiser):
    """
    Adam (Adaptive Moment Estimation) optimiser.
    
    Combines momentum and RMSprop for adaptive learning rates.
    Updates use both first and second moment estimates of gradients.
    
    Parameters
    ----------
    learning_rate : float
        Step size for parameter updates (default: 0.001)
    beta1 : float
        Exponential decay rate for first moment (default: 0.9)
    beta2 : float
        Exponential decay rate for second moment (default: 0.999)
    epsilon : float
        Small constant for numerical stability (default: 1e-8)
    
    Examples
    --------
    >>> optimiser = Adam(learning_rate=0.001)
    >>> model.set_output_gradient(loss_gradient)
    >>> optimiser.step(model)
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimiser.
        
        :param learning_rate: Learning rate for parameter updates
        :type learning_rate: float
        :param beta1: Exponential decay rate for first moment
        :type beta1: float
        :param beta2: Exponential decay rate for second moment
        :type beta2: float
        :param epsilon: Small constant for numerical stability
        :type epsilon: float
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize moment estimates
        self.m = None  # First moment (mean of gradients)
        self.v = None  # Second moment (uncentered variance of gradients)
        self.t = 0     # Time step
    
    def step(self, model):
        """
        Perform one Adam optimisation step.
        
        :param model: Model with parameters and gradients properties
        :type model: Model
        """
        # Get current parameters and gradients
        params = model.parameters
        grads = model.gradients
        
        # Initialize moments on first step
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        # Increment time step
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        new_params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        model.parameters = new_params


def train_model(model, X, y, loss_fn, optimiser, n_epochs=100, verbose=True):
    """
    Train a model using a specified optimiser and loss function.
    
    This function provides a unified interface for training models
    with different optimisers and loss functions.
    
    Parameters
    ----------
    model : Model
        Model with predict(), parameters, gradients, and set_output_gradient()
    X : numpy.ndarray
        Training input data
    y : numpy.ndarray
        Training target data
    loss_fn : LossFunction
        Loss function with forward() and gradient() methods
    optimiser : Optimiser
        optimiser with step() method
    n_epochs : int
        Number of training epochs (default: 100)
    verbose : bool
        Whether to print progress (default: True)
    
    Returns
    -------
    list
        List of loss values for each epoch
    
    Examples
    --------
    >>> from mlai import NeuralNetwork, SGD, MeanSquaredError, train_model
    >>> model = NeuralNetwork([2, 10, 1], [ReLUActivation(), LinearActivation()])
    >>> optimiser = SGD(learning_rate=0.01)
    >>> loss_fn = MeanSquaredError()
    >>> losses = train_model(model, X_train, y_train, loss_fn, optimiser, n_epochs=100)
    """
    losses = []
    
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model.predict(X)
        
        # Compute loss
        loss = loss_fn.forward(y_pred, y)
        losses.append(loss)
        
        # Compute loss gradient
        loss_gradient = loss_fn.gradient(y_pred, y)
        
        # Set output gradient for the model
        model.set_output_gradient(loss_gradient)
        
        # optimisation step
        optimiser.step(model)
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    return losses
