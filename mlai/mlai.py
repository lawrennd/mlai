# Python code for MLAI lectures.

"""
Machine Learning and Adaptive Intelligence (MLAI) Core Module

This module provides the core machine learning functionality for the MLAI package,
designed for teaching and lecturing on machine learning fundamentals. The module
includes implementations of key algorithms with a focus on clarity and educational value.

Key Components:
-------------
- Linear Models (LM): Basic linear regression with various basis functions
- Bayesian Linear Models (BLM): Linear models with Bayesian inference
- Gaussian Processes (GP): Non-parametric Bayesian models
- Logistic Regression (LR): Binary classification
- Neural Networks: Simple feedforward networks with dropout
- Kernel Functions: Various covariance functions for GPs
- Perceptron: Basic binary classifier for teaching

Mathematical Focus:
------------------
The implementations emphasize mathematical transparency, with clear connections
between code and mathematical notation. Each class and function includes
mathematical explanations where relevant.

Educational Design:
------------------
- Simple, readable implementations suitable for teaching
- Clear separation of concepts
- Extensive use of mathematical notation in variable names
- Comprehensive examples and documentation

For detailed usage examples, see the tutorials in gp_tutorial.py, deepgp_tutorial.py,
and mountain_car.py.

Author: Neil D. Lawrence
License: MIT
"""

# import the time model to allow python to pause.
import time
import os
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML

import numpy as np
import scipy.linalg as la

from numpy import vstack


        
##########          Week 2          ##########
def init_perceptron(x_plus, x_minus, seed=1000001):
    """
    Initialise the perceptron algorithm with random weights and bias.
    
    The perceptron is a simple binary classifier that learns a linear decision boundary.
    This function initialises the weight vector w and bias b by randomly selecting
    a point from either the positive or negative class and setting the normal vector
    accordingly.
    
    Mathematical formulation:
    The perceptron decision function is f(x) = w^T x + b, where:
    - w is the weight vector (normal to the decision boundary)
    - b is the bias term
    - x is the input vector
    
    :param x_plus: Positive class data points, shape (n_plus, n_features)
    :type x_plus: numpy.ndarray
    :param x_minus: Negative class data points, shape (n_minus, n_features)
    :type x_minus: numpy.ndarray
    :param seed: Random seed for reproducible initialization
    :type seed: int, optional
    :returns: Initial weight vector, shape (n_features,)
    :rtype: numpy.ndarray
    :returns: Initial bias term
    :rtype: float
    :returns: The randomly selected point used for initialization
    :rtype: numpy.ndarray
    
    Examples:
        >>> x_plus = np.array([[1, 2], [2, 3], [3, 4]])
        >>> x_minus = np.array([[0, 0], [1, 0], [0, 1]])
        >>> w, b, x_select = init_perceptron(x_plus, x_minus)
        >>> print(f"Weight vector: {w}, Bias: {b}")
    """
    np.random.seed(seed=seed)
    # flip a coin (i.e. generate a random number and check if it is greater than 0.5)
    plus_portion = x_plus.shape[0]/(x_plus.shape[0] + x_minus.shape[0])
    choose_plus = np.random.rand(1)<plus_portion
    if choose_plus:
        # generate a random point from the positives
        index = np.random.randint(0, x_plus.shape[0])
        x_select = x_plus[index, :]
        w = x_plus[index, :].astype(float)  # set the normal vector to plus that point
        b = 1
    else:
        # generate a random point from the negatives
        index = np.random.randint(0, x_minus.shape[0])
        x_select = x_minus[index, :]
        w = -x_minus[index, :].astype(float)  # set the normal vector to minus that point.
        b = -1
    return w, b, x_select


def update_perceptron(w, b, x_plus, x_minus, learn_rate):
    """
    Update the perceptron weights and bias using stochastic gradient descent.
    
    This function implements one step of the perceptron learning algorithm.
    It randomly selects a training point and updates the weights if the point
    is misclassified.
    
    Mathematical formulation:
    The perceptron update rule is:
    - If y_i = +1 and w^T x_i + b ≤ 0: w ← w + η x_i, b ← b + η
    - If y_i = -1 and w^T x_i + b > 0: w ← w - η x_i, b ← b - η
    
    where η is the learning rate.
    
    :param w: Current weight vector, shape (n_features,)
    :type w: numpy.ndarray
    :param b: Current bias term
    :type b: float
    :param x_plus: Positive class data points, shape (n_plus, n_features)
    :type x_plus: numpy.ndarray
    :param x_minus: Negative class data points, shape (n_minus, n_features)
    :type x_minus: numpy.ndarray
    :param learn_rate: Learning rate (step size) for weight updates
    :type learn_rate: float
    :returns: Updated weight vector
    :rtype: numpy.ndarray
    :returns: Updated bias term
    :rtype: float
    :returns: The randomly selected point that was used for the update
    :rtype: numpy.ndarray
    :returns: True if weights were updated, False otherwise
    :rtype: bool
    
    Examples:
        >>> w = np.array([0.5, -0.3])
        >>> b = 0.1
        >>> x_plus = np.array([[1, 2], [2, 3]])
        >>> x_minus = np.array([[0, 0], [1, 0]])
        >>> w_new, b_new, x_select, updated = update_perceptron(w, b, x_plus, x_minus, 0.1)
    """
    # select a point at random from the data
    plus_portion = x_plus.shape[0]/(x_plus.shape[0] + x_minus.shape[0])
    choose_plus = np.random.rand(1)<plus_portion
    updated=False
    if choose_plus:
        # choose a point from the positive data
        index = np.random.randint(x_plus.shape[0])
        x_select = x_plus[index, :]
        if np.dot(w, x_select)+b <= 0.:
            # point is currently incorrectly classified
            w += learn_rate*x_select
            b += learn_rate
            updated=True
    else:
        # choose a point from the negative data
        index = np.random.randint(x_minus.shape[0])
        x_select = x_minus[index, :]
        if np.dot(w, x_select)+b > 0.:
            # point is currently incorrectly classified
            w -= learn_rate*x_select
            b -= learn_rate
            updated=True
    return w, b, x_select, updated

##########           Weeks 4 and 5           ##########
class Model(object):
    """
    Abstract base class for all machine learning models.
    
    This class defines the interface that all models in MLAI must implement.
    It provides a common structure for different types of models while
    allowing for specific implementations of objective functions and fitting procedures.
    
    Methods
    -------
    objective() : float
        Compute the objective function value (to be minimized)
    fit() : None
        Fit the model to the data (to be implemented by subclasses)
    
    Examples:
        >>> class MyModel(Model):
        ...     def objective(self):
        ...         return 0.0  # Implement objective
        ...     def fit(self):
        ...         pass  # Implement fitting
    """
    def __init__(self):
        pass
    
    def objective(self):
        """
        Compute the objective function value.
        
        This method should return a scalar value that represents
        the model's objective function (typically to be minimized).
        
        :returns: The objective function value
        :rtype: float
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def fit(self):
        """
        Fit the model to the data.
        
        This method should implement the model fitting procedure,
        updating the model parameters to minimize the objective function.
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

class ProbModel(Model):
    """
    Abstract base class for probabilistic models.
    
    This class extends the base Model class to provide a framework for
    probabilistic models that can compute log-likelihoods. The objective
    function is defined as the negative log-likelihood, which is commonly
    minimized in maximum likelihood estimation.
    
    Mathematical formulation:
    The objective function is: J(θ) = -log p(y|X, θ)
    where θ are the model parameters, y are the targets, and X are the inputs.
    
    Methods
    -------
    objective() : float
        Compute the negative log-likelihood
    log_likelihood() : float
        Compute the log-likelihood (to be implemented by subclasses)
    
    Examples:
        >>> class GaussianModel(ProbModel):
        ...     def log_likelihood(self):
        ...         return -0.5 * np.sum((y - μ)**2 / σ²)  # Gaussian log-likelihood
    """
    def __init__(self):
        Model.__init__(self)

    def objective(self):
        """
        Compute the negative log-likelihood.
        
        This is the standard objective function for probabilistic models,
        which is minimized during maximum likelihood estimation.
        
        :returns: The negative log-likelihood: -log p(y|X, θ)
        :rtype: float
        """
        return -self.log_likelihood()

    def log_likelihood(self):
        """
        Compute the log-likelihood of the data given the model parameters.
        
        This method should be implemented by subclasses to provide
        the specific log-likelihood computation for their model.
        
        :returns: The log-likelihood: log p(y|X, θ)
        :rtype: float
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

class MapModel(Model):
    """
    Abstract base class for models that provide a mapping from inputs X to outputs y.
    
    This class extends the base Model class to provide functionality for
    supervised learning models that map input features to target values.
    It includes methods for computing prediction errors and root mean square error.
    
    Mathematical formulation:
    The model provides a mapping: f: X → y
    where f is the learned function, X are the inputs, and y are the targets.
    
    :param X: Input features, shape (n_samples, n_features)
    :type X: numpy.ndarray
    :param y: Target values, shape (n_samples,)
    :type y: numpy.ndarray
    
    Methods
    -------
    predict(X) : numpy.ndarray
        Make predictions for new inputs
    rmse() : float
        Compute the root mean square error
    update_sum_squares() : None
        Update the sum of squared errors
    
    Examples:
        >>> class LinearModel(MapModel):
        ...     def predict(self, X):
        ...         return X @ self.w + self.b
        ...     def update_sum_squares(self):
        ...         self.sum_squares = np.sum((self.y - self.predict(self.X))**2)
    """
    def __init__(self, X, y):
        Model.__init__(self)
        self.X = X
        self.y = y
        self.num_data = y.shape[0]

    def update_sum_squares(self):
        """
        Update the sum of squared errors.
        
        This method should be implemented by subclasses to compute
        the sum of squared differences between predictions and targets.
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError
    
    def rmse(self):
        """
        Compute the root mean square error.
        
        The RMSE is defined as: √(Σ(y_i - ŷ_i)² / n)
        where y_i are the true values, ŷ_i are the predictions, and n is the number of samples.
        
        :returns: The root mean square error
        :rtype: float
        """
        self.update_sum_squares()
        return np.sqrt(self.sum_squares/self.num_data)

    def predict(self, X):
        """
        Make predictions for new input data.
        
        This method should be implemented by subclasses to provide
        the specific prediction function for their model.
        
        :param X: Input features, shape (n_samples, n_features)
        :type X: numpy.ndarray
        :returns: Predictions, shape (n_samples,)
        :rtype: numpy.ndarray
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    
class ProbMapModel(ProbModel, MapModel):
    """Probabilistic model that provides a mapping from X to y."""
    def __init__(self, X, y):
        ProbModel.__init__(self)
        MapModel.__init__(self, X, y)

    
class LM(ProbMapModel):
    """
    Linear Model with basis function expansion.
    
    This class implements a linear model that uses basis functions to transform
    the input features. The model assumes Gaussian noise and uses maximum
    likelihood estimation to fit the parameters.
    
    Mathematical formulation:
    The model is: y = Φ(X)w + ε, where ε ~ N(0, σ²I)
    - Φ(X) is the basis function matrix
    - w are the weights to be learned
    - σ² is the noise variance
    
    The maximum likelihood solution is: w* = (Φ^T Φ)^(-1) Φ^T y
    
    :param X: Input features, shape (n_samples, n_features)
    :type X: numpy.ndarray
    :param y: Target values, shape (n_samples, 1)
    :type y: numpy.ndarray
    :param basis: Basis function object that provides the feature transformation
    :type basis: Basis
    
    Attributes
    ----------
    w_star : numpy.ndarray
        Fitted weight vector
    sigma2 : float
        Estimated noise variance
    Phi : numpy.ndarray
        Basis function matrix Φ(X)
    name : str
        Model name based on the basis function
    objective_name : str
        Name of the objective function
    
    Examples:
        >>> from mlai import polynomial
        >>> basis = polynomial(num_basis=4)
        >>> model = LM(X, y, basis)
        >>> model.fit()
        >>> y_pred, _ = model.predict(X_test)
    """
    def __init__(self, X, y, basis):
        """
        Initialize a linear model with basis function expansion.
        
        :param X: Input features, shape (n_samples, n_features)
        :type X: numpy.ndarray
        :param y: Target values, shape (n_samples, 1)
        :type y: numpy.ndarray
        :param basis: Basis function object that provides the feature transformation
        :type basis: Basis
        """
        ProbModel.__init__(self)
        # Ensure y is 2D (n_samples, 1)
        if y.ndim  != 2 or y.shape[1] != 1:
            raise ValueError("y must be 2D with shape (n_samples, 1)")
        self.y = y
        self.num_data = self.y.shape[0]
        self.X = X
        self.sigma2 = 1.
        self.basis = basis
        self.Phi = self.basis.Phi(X)
        self.name = 'LM_'+self.basis.function.__name__
        self.objective_name = 'Sum of Square Training Error'

    def set_param(self, name, val, update_fit=True):
        """
        Set a model parameter to a given value.
        
        This method allows updating model parameters (including basis function
        parameters) and optionally refits the model if the parameter change
        affects the basis function matrix.
        
        :param name: Name of the parameter to set
        :type name: str
        :param val: New value for the parameter
        :type val: float
        :param update_fit: Whether to refit the model after setting the parameter.
            Default is True.
        :type update_fit: bool, optional
        
        :raises ValueError: If the parameter name is not recognized
        
        Examples:
            >>> model.set_param('sigma2', 0.5)
            >>> model.set_param('num_basis', 6)  # Basis function parameter
        """
        if name in self.__dict__:
            if self.__dict__[name] == val:
                update_fit=False
            else:
                self.__dict__[name] = val
        elif name in self.basis.__dict__:
            if self.basis.__dict__[name] == val:
                update_fit=False
            else:
                self.basis.__dict__[name] = val
                self.Phi = self.basis.Phi(self.X)            
        else:
            raise ValueError("Unknown parameter '{}' being set.".format(name))
        if update_fit:
            self.fit()

    def update_QR(self):
        """
        Perform QR decomposition on the basis matrix.
        
        The QR decomposition is used for numerically stable computation
        of the least squares solution. The basis matrix Φ is decomposed as:
        Φ = QR, where Q is orthogonal and R is upper triangular.
        
        This decomposition allows efficient computation of the weight vector:
        w* = R^(-1) Q^T y
        """
        self.Q, self.R = np.linalg.qr(self.Phi)

    def fit(self):
        """
        Fit the linear model using maximum likelihood estimation.
        
        This method computes the optimal weight vector w* using the
        least squares solution and estimates the noise variance σ².
        
        Mathematical formulation:
        - w* = R^(-1) Q^T y (using QR decomposition)
        - σ² = Σ(y_i - ŷ_i)² / n (maximum likelihood estimate)
        
        Examples:
            >>> model = LM(X, y, basis)
            >>> model.fit()
            >>> print(f"Weights: {model.w_star}")
            >>> print(f"Noise variance: {model.sigma2}")
        """
        self.update_QR()
        self.w_star = la.solve_triangular(self.R, self.Q.T@self.y)
        self.update_sum_squares()
        self.sigma2=self.sum_squares/self.num_data

    def predict(self, X):
        """
        Make predictions for new input data.
        
        This method applies the learned weight vector to the basis function
        transformation of the input data.
        
        Mathematical formulation:
        ŷ = Φ(X) w*
        where Φ(X) is the basis function matrix and w* are the fitted weights.
        
        :param X: Input features, shape (n_samples, n_features)
        :type X: numpy.ndarray
        :returns: (predictions, None) where predictions are the predicted values
            and None indicates no uncertainty estimate (this is a deterministic model)
        :rtype: tuple
        
        Examples:
            >>> y_pred, _ = model.predict(X_test)
            >>> print(f"Predictions shape: {y_pred.shape}")
        """
        return self.basis.Phi(X)@self.w_star, None
        
    def update_f(self):
        """Update values at the prediction points."""
        self.f = self.Phi@self.w_star
        
    def update_sum_squares(self):
        """Compute the sum of squares error."""
        self.update_f()
        self.sum_squares = ((self.y-self.f)**2).sum()
        
    def objective(self):
        """Compute the objective function."""
        self.update_sum_squares()
        return self.sum_squares

    def log_likelihood(self):
        """Compute the log likelihood."""
        self.update_sum_squares()
        return -self.num_data/2.*np.log(np.pi*2.)-self.num_data/2.*np.log(self.sigma2)-self.sum_squares/(2.*self.sigma2)
    

class Basis():
    """
    Basis function wrapper class.
    
    This class provides a wrapper around basis functions to standardize
    their interface and store their parameters.
    
    :param function: The basis function to wrap
    :type function: function
    :param number: Number of basis functions
    :type number: int
    :param **kwargs: Additional arguments passed to the basis function
    :type **kwargs: dict
    """
    def __init__(self, function, number, **kwargs):
        """
        Initialize the basis function wrapper.
        
        :param function: The basis function to wrap
        :type function: function
        :param number: Number of basis functions
        :type number: int
        :param **kwargs: Additional arguments passed to the basis function
        :type **kwargs: dict
        """
        self.arguments=kwargs
        self.number=number
        self.function=function

    def Phi(self, X):
        """
        Compute the basis function matrix.
        
        :param X: Input features, shape (n_samples, n_features)
        :type X: numpy.ndarray
        :returns: Basis function matrix Φ(X), shape (n_samples, num_basis)
        :rtype: numpy.ndarray
        """
        return self.function(X, num_basis=self.number, **self.arguments)

def linear(x, **kwargs):
    """
    Define the linear basis function.
    
    Creates a basis matrix with a constant term (bias) and linear terms.
    
    :param x: Input features, shape (n_samples, n_features)
    :type x: numpy.ndarray
    :param **kwargs: Additional arguments (ignored for linear basis)
    :type **kwargs: dict
    :returns: Basis matrix with constant and linear terms, shape (n_samples, n_features + 1)
    :rtype: numpy.ndarray
    
    Examples:
        >>> X = np.array([[1, 2], [3, 4]])
        >>> Phi = linear(X)
        >>> print(Phi.shape)  # (2, 3) - includes bias term
    """
    return np.hstack([np.ones((x.shape[0], 1)), np.asarray(x, dtype=float)])



def polynomial(x, num_basis=4, data_limits=[-1., 1.]):
    """
    Define the polynomial basis function.
    
    Creates a basis matrix with polynomial terms up to degree (num_basis - 1).
    The input is normalized to the range [-1, 1] for numerical stability.
    
    :param x: Input features, shape (n_samples, n_features)
    :type x: numpy.ndarray
    :param num_basis: Number of basis functions (polynomial degree + 1)
    :type num_basis: int, optional
    :param data_limits: Range for normalizing the input data [min, max]
    :type data_limits: list, optional
    :returns: Polynomial basis matrix, shape (n_samples, num_basis)
    :rtype: numpy.ndarray
    
    Examples:
        >>> X = np.array([[0.5], [1.0]])
        >>> Phi = polynomial(X, num_basis=3)
        >>> print(Phi.shape)  # (2, 3) - constant, linear, quadratic terms
    """
    centre = data_limits[0]/2. + data_limits[1]/2.
    span = data_limits[1] - data_limits[0]
    z = np.asarray(x, dtype=float) - centre
    z = 2*z/span
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i+1] = z**i
    return Phi

def radial(x, num_basis=4, data_limits=[-1., 1.], width=None):
    """
    Define the radial basis function (RBF).
    
    Creates a basis matrix using Gaussian radial basis functions centered
    at evenly spaced points across the data range.
    
    :param x: Input features, shape (n_samples, n_features)
    :type x: numpy.ndarray
    :param num_basis: Number of radial basis functions
    :type num_basis: int, optional
    :param data_limits: Range for centering the basis functions [min, max]
    :type data_limits: list, optional
    :param width: Width parameter for the Gaussian functions. If None, auto-computed.
    :type width: float, optional
    :returns: Radial basis matrix, shape (n_samples, num_basis)
    :rtype: numpy.ndarray
    
    Examples:
        >>> X = np.array([[0.5], [1.0]])
        >>> Phi = radial(X, num_basis=3)
        >>> print(Phi.shape)  # (2, 3)
    """
    if num_basis>1:
        centres=np.linspace(data_limits[0], data_limits[1], num_basis)
        if width is None:
            width = (centres[1]-centres[0])/2.
    else:
        centres = np.asarray([data_limits[0]/2. + data_limits[1]/2.])
        if width is None:
            width = (data_limits[1]-data_limits[0])/2.
    
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i+1] = np.exp(-0.5*((np.asarray(x, dtype=float)-centres[i])/width)**2)
    return Phi


def fourier(x, num_basis=4, data_limits=[-1., 1.], frequency_range=None):
    """
    Define the Fourier basis function.
    
    Creates a basis matrix using sine and cosine functions with different
    frequencies. The first basis function is a constant (1), followed by
    alternating sine and cosine terms.
    
    :param x: Input features, shape (n_samples, n_features)
    :type x: numpy.ndarray
    :param num_basis: Number of basis functions (including constant term)
    :type num_basis: int, optional
    :param data_limits: Range for scaling the frequencies [min, max]
    :type data_limits: list, optional
    :param frequency_range: Specific frequencies to use. If None, auto-computed.
    :type frequency_range: list, optional
    :returns: Fourier basis matrix, shape (n_samples, num_basis)
    :rtype: numpy.ndarray
    
    Examples:
        >>> X = np.array([[0.5], [1.0]])
        >>> Phi = fourier(X, num_basis=4)
        >>> print(Phi.shape)  # (2, 4) - constant, sin, cos, sin
    """
    tau = 2*np.pi
    span = float(data_limits[1]-data_limits[0])
    Phi = np.ones((x.shape[0], num_basis))
    for i in range(1, num_basis):
        count = float((i+1)//2)
        if frequency_range is None:
            frequency = count/span
        else:
            frequency = frequency_range[i]
        if i % 2:
            Phi[:, i:i+1] = np.sin(tau*frequency*np.asarray(x, dtype=float))
        else:
            Phi[:, i:i+1] = np.cos(tau*frequency*np.asarray(x, dtype=float))
    return Phi

def relu(x, num_basis=4, data_limits=[-1., 1.], gain=None):
    """
    Define the rectified linear units (ReLU) basis function.
    
    Creates a basis matrix using ReLU activation functions with different
    thresholds. The first basis function is a constant (1), followed by
    ReLU functions with varying thresholds and gains.
    
    :param x: Input features, shape (n_samples, n_features)
    :type x: numpy.ndarray
    :param num_basis: Number of basis functions (including constant term)
    :type num_basis: int, optional
    :param data_limits: Range for positioning the ReLU thresholds [min, max]
    :type data_limits: list, optional
    :param gain: Gain parameters for each ReLU function. If None, auto-computed.
    :type gain: numpy.ndarray, optional
    :returns: ReLU basis matrix, shape (n_samples, num_basis)
    :rtype: numpy.ndarray
    
    Examples:
        >>> X = np.array([[0.5], [1.0]])
        >>> Phi = relu(X, num_basis=3)
        >>> print(Phi.shape)  # (2, 3) - constant + 2 ReLU functions
    """
    if num_basis>2:
        centres=np.linspace(data_limits[0], data_limits[1], num_basis)[:-1]
    elif num_basis==2:
        centres = np.asarray([data_limits[0]/2. + data_limits[1]/2.])
    else:
        centres = []
    if num_basis < 3:
        basis_gap = (data_limits[1]-data_limits[0])
    else:
        basis_gap = (data_limits[1]-data_limits[0])/(num_basis-2)
    if gain is None:
        gain = np.ones(num_basis-1)/basis_gap
    Phi = np.zeros((x.shape[0], num_basis))
    # Create the bias
    Phi[:, 0] = 1.0
    for i in range(1, num_basis):
        Phi[:, i:i+1] = gain[i-1]*(np.asarray(x, dtype=float)>centres[i-1])*(np.asarray(x, dtype=float)-centres[i-1])
    return Phi

def hyperbolic_tangent(x, num_basis=4, data_limits=[-1., 1.], gain=None):
    """
    Define the hyperbolic tangent basis function.
    
    Creates a basis matrix using tanh activation functions with different
    thresholds and gains. The first basis function is a constant (1),
    followed by tanh functions with varying parameters.
    
    :param x: Input features, shape (n_samples, n_features)
    :type x: numpy.ndarray
    :param num_basis: Number of basis functions (including constant term)
    :type num_basis: int, optional
    :param data_limits: Range for positioning the tanh thresholds [min, max]
    :type data_limits: list, optional
    :param gain: Gain parameters for each tanh function. If None, auto-computed.
    :type gain: numpy.ndarray, optional
    :returns: Tanh basis matrix, shape (n_samples, num_basis)
    :rtype: numpy.ndarray
    
    Examples:
        >>> X = np.array([[0.5], [1.0]])
        >>> Phi = hyperbolic_tangent(X, num_basis=3)
        >>> print(Phi.shape)  # (2, 3) - constant + 2 tanh functions
    """
    if num_basis>2:
        centres=np.linspace(data_limits[0], data_limits[1], num_basis-1)
        width = (centres[1]-centres[0])/2.
    elif num_basis==2:
        centres = np.asarray([data_limits[0]/2. + data_limits[1]/2.])
        width = (data_limits[1]-data_limits[0])/2.
    else:
        centres = []
        width = None
    if gain is None and width is not None:
        gain = np.ones(num_basis-1)/width
    Phi = np.zeros((x.shape[0], num_basis))
    # Create the bias
    Phi[:, 0] = 1.0
    for i in range(1, num_basis):
        Phi[:, i:i+1] = np.tanh(gain[i-1]*(np.asarray(x, dtype=float)-centres[i-1]))
    return Phi

class Noise(ProbModel):
    """
    Abstract base class for noise models.
    
    This class extends ProbModel to provide a framework for modeling
    noise distributions in probabilistic models.
    """
    def __init__(self):
        ProbModel.__init__(self)

    def _repr_html_(self):
        raise NotImplementedError

    
class Gaussian(Noise):
    """
    Gaussian noise model.
    
    This class implements a Gaussian noise model with configurable
    offset and scale parameters.
    
    :param offset: Offset parameter for the noise model
    :type offset: float, optional
    :param scale: Scale parameter for the noise model
    :type scale: float, optional
    """
    def __init__(self, offset=0., scale=1.):
        Noise.__init__(self)
        self.scale = scale
        self.offset = offset
        self.variance = scale*scale

    def log_likelihood(self, mu, varsigma, y):
        """
        Compute the log likelihood of the data under a Gaussian noise model.
        
        :param mu: Input mean locations for the log likelihood
        :type mu: numpy.ndarray
        :param varsigma: Input variance locations for the log likelihood
        :type varsigma: numpy.ndarray
        :param y: Target locations for the log likelihood
        :type y: numpy.ndarray
        :returns: Log likelihood value
        :rtype: float
        """

        n = y.shape[0]
        d = y.shape[1]
        varsigma = varsigma + self.scale*self.scale
        for i in range(d):
            mu[:, i] += self.offset[i]
        arg = (y - mu);
        arg = arg*arg/varsigma

        return - 0.5*(np.log(varsigma).sum() + arg.sum() + n*d*np.log(2*np.pi))


    def grad_vals(self, mu, varsigma, y):
        """
        Compute the gradient of noise log Z with respect to input mean and variance.
        
        :param mu: Mean input locations with respect to which gradients are being computed
        :type mu: numpy.ndarray
        :param varsigma: Variance input locations with respect to which gradients are being computed
        :type varsigma: numpy.ndarray
        :param y: Noise model output observed values associated with the given points
        :type y: numpy.ndarray
        :returns: Tuple containing the gradient of log Z with respect to the input mean and the gradient of log Z with respect to the input variance
        :rtype: tuple
        """

        d = y.shape[1]
        nu = 1/(self.scale*self.scale+varsigma)
        dlnZ_dmu = np.zeros(nu.shape)
        for i in range(d):
            dlnZ_dmu[:, i] = y[:, i] - mu[:, i] - self.offset[i]
        dlnZ_dmu = dlnZ_dmu*nu
        dlnZ_dvs = 0.5*(dlnZ_dmu*dlnZ_dmu - nu)
        return dlnZ_dmu, dlnZ_dvs

class SimpleNeuralNetwork(Model):
    """
    A simple one-layer neural network.
    
    This class implements a basic neural network with one hidden layer
    using ReLU activation functions.
    
    :param nodes: Number of hidden nodes
    :type nodes: int
    """
    def __init__(self, nodes):
        if nodes <= 0:
            raise ValueError("Number of nodes must be positive, got {}".format(nodes))
        self.nodes = nodes
        self.w2 = np.random.normal(size=self.nodes)/self.nodes
        self.b2 = np.random.normal(size=1)
        self.w1 = np.random.normal(size=self.nodes)
        self.b1 = np.random.normal(size=self.nodes)
        

    def predict(self, x):
        """
        Compute output given current basis functions.
        
        :param x: Input value
        :type x: float
        :returns: Network output
        :rtype: float
        """
        vxmb = self.w1*x + self.b1
        phi = vxmb*(vxmb>0)
        return np.sum(self.w2*phi) + self.b2

class SimpleDropoutNeuralNetwork(SimpleNeuralNetwork):
    """
    Simple neural network with dropout.
    
    This class extends SimpleNeuralNetwork to include dropout regularization
    during training.
    
    :param nodes: Number of hidden nodes
    :type nodes: int
    :param drop_p: Dropout probability
    :type drop_p: float, optional
    """
    def __init__(self, nodes, drop_p=0.5):
        if drop_p <= 0 or drop_p >= 1:
            raise ValueError("Dropout probability must be between 0 and 1, got {}".format(drop_p))
        self.drop_p = drop_p
        super().__init__(nodes=nodes)
        # renormalize the network weights
        self.w2 /= self.drop_p 
        
    def do_samp(self):
        """
        Sample the set of basis functions to use.
        
        This method randomly selects which basis functions to use
        based on the dropout probability.
        """ 
        gen = np.random.rand(self.nodes)
        self.use = gen > self.drop_p
        
    def predict(self, x):
        """
        Compute output given current basis functions used.
        
        :param x: Input value
        :type x: float
        :returns: Network output using only the sampled basis functions
        :rtype: float
        """
        vxmb = self.w1[self.use]*x + self.b1[self.use]
        phi = vxmb*(vxmb>0)
        return np.sum(self.w2[self.use]*phi) + self.b2

class NonparametricDropoutNeuralNetwork(SimpleDropoutNeuralNetwork):
    """
    A non-parametric dropout neural network.
    
    This class implements a neural network with non-parametric dropout
    using the Indian Buffet Process (IBP) to control the dropout mechanism.
    
    :param alpha: Alpha parameter of the IBP controlling dropout
    :type alpha: float, optional
    :param beta: Beta parameter of the two-parameter IBP controlling dropout
    :type beta: float, optional
    :param n: Number of data points for computing expected features
    :type n: int, optional
    """
    def __init__(self, alpha=10, beta=1, n=1000):
        if alpha <= 0:
            raise ValueError("Alpha parameter must be positive, got {}".format(alpha))
        if beta <= 0:
            raise ValueError("Beta parameter must be positive, got {}".format(beta))
        if n <= 0:
            raise ValueError("Number of data points must be positive, got {}".format(n))
        self.update_num = 0
        self.alpha = alpha
        self.beta = beta
        self.gamma = 0.5772156649
        tot = np.log(n) + self.gamma + 0.5/n * (1./12.)/(n*n)
        self.exp_features = alpha*beta*tot
        self.maxk = np.max((10000,int(self.exp_features + np.ceil(4*np.sqrt(self.exp_features)))))
        super().__init__(nodes=self.maxk, drop_p=self.alpha/self.maxk)
        self.maxval = 0
        self.w2 *= self.maxk/self.alpha
        self.count = np.zeros(self.maxk)
    
    
        
    def do_samp(self):
        """
        Sample the next set of basis functions to be used.
        
        This method implements the Indian Buffet Process (IBP) sampling
        to determine which basis functions to use in the current iteration.
        """
        
        new=np.random.poisson(self.alpha*self.beta/(self.beta + self.update_num))
        use_prob = self.count[:self.maxval]/(self.update_num+self.beta)
        gen = np.random.rand(1, self.maxval)
        self.use = np.zeros(self.maxk, dtype=bool)
        self.use[:self.maxval] = gen < use_prob
        self.use[self.maxval:self.maxval+new] = True
        self.maxval+=new
        self.update_num+=1
        self.count[:self.maxval] += self.use[:self.maxval]

def relu_activation(x):
    """
    ReLU activation function.
    
    This function applies the ReLU (Rectified Linear Unit) activation
    to the input array.
    
    :param x: Input array
    :type x: numpy.ndarray
    :returns: Activated array with ReLU applied
    :rtype: numpy.ndarray
    
    Examples:
        >>> x = np.array([-1, 0, 1])
        >>> y = relu_activation(x)
        >>> print(y)  # Output: [0, 0, 1]
    """
    return x*(x>0)


def sigmoid_activation(x):
    """
    Sigmoid activation function.
    
    This function applies the sigmoid activation to the input array,
    mapping values to the range (0, 1).
    
    :param x: Input array
    :type x: numpy.ndarray
    :returns: Activated array with sigmoid applied
    :rtype: numpy.ndarray
    
    Examples:
        >>> x = np.array([-1, 0, 1])
        >>> y = sigmoid_activation(x)
        >>> print(y)  # Output: [0.26894142, 0.5, 0.73105858]
    """
    return 1./(1.+np.exp(-x))

def linear_activation(x):
    """
    Linear activation function (identity/null operation).
    
    This function applies a linear transformation to the input array,
    effectively returning the input unchanged. This is useful for
    output layers or when no activation is desired.
    
    :param x: Input array
    :type x: numpy.ndarray
    :returns: Input array unchanged
    :rtype: numpy.ndarray
    
    Examples:
        >>> x = np.array([-1, 0, 1])
        >>> y = linear_activation(x)
        >>> print(y)  # Output: [-1, 0, 1]
    """
    return x

def soft_relu_activation(x):
    """
    Soft ReLU activation function (log(1 + exp(x))).
    
    This function applies the soft ReLU (also known as softplus) activation
    to the input array. It is a smooth approximation to the ReLU function
    that is differentiable everywhere.
    
    Mathematical formulation:
    soft_relu(x) = log(1 + exp(x))
    
    :param x: Input array
    :type x: numpy.ndarray
    :returns: Activated array with soft ReLU applied
    :rtype: numpy.ndarray
    
    Examples:
        >>> x = np.array([-1, 0, 1])
        >>> y = soft_relu_activation(x)
        >>> print(y)  # Output: [0.31326169, 0.69314718, 1.31326169]
    """
    return np.log(1. + np.exp(x))

class Activation:
    """
    Base class for activation functions with gradients.
    
    This class provides a framework for activation functions that include
    both the forward pass and gradient computation, useful for neural networks
    that require backpropagation.
    """
    
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Forward pass of the activation function.
        
        :param x: Input array
        :type x: numpy.ndarray
        :returns: Activated array
        :rtype: numpy.ndarray
        """
        raise NotImplementedError
    
    def gradient(self, x):
        """
        Compute the gradient of the activation function.
        
        :param x: Input array
        :type x: numpy.ndarray
        :returns: Gradient array
        :rtype: numpy.ndarray
        """
        raise NotImplementedError

class LinearActivation(Activation):
    """
    Linear activation function (identity) with gradient.
    
    This activation function returns the input unchanged and has a gradient
    of 1 everywhere. Useful for output layers or when no activation is desired.
    """
    
    def forward(self, x):
        """
        Forward pass: returns input unchanged.
        
        :param x: Input array
        :type x: numpy.ndarray
        :returns: Input array unchanged
        :rtype: numpy.ndarray
        """
        return x
    
    def gradient(self, x):
        """
        Gradient of linear activation (always 1).
        
        :param x: Input array
        :type x: numpy.ndarray
        :returns: Array of ones with same shape as input
        :rtype: numpy.ndarray
        """
        return np.ones_like(x)

class ReLUActivation(Activation):
    """
    ReLU activation function with gradient.
    
    ReLU (Rectified Linear Unit) activation function that returns max(0, x)
    with appropriate gradient computation.
    """
    
    def forward(self, x):
        """
        Forward pass: ReLU activation.
        
        :param x: Input array
        :type x: numpy.ndarray
        :returns: ReLU activated array
        :rtype: numpy.ndarray
        """
        return x * (x > 0)
    
    def gradient(self, x):
        """
        Gradient of ReLU activation.
        
        :param x: Input array
        :type x: numpy.ndarray
        :returns: Gradient array (1 where x > 0, 0 elsewhere)
        :rtype: numpy.ndarray
        """
        return (x > 0).astype(float)

class SigmoidActivation(Activation):
    """
    Sigmoid activation function with gradient.
    
    Sigmoid activation function that maps inputs to (0, 1) range
    with appropriate gradient computation.
    """
    
    def forward(self, x):
        """
        Forward pass: sigmoid activation.
        
        :param x: Input array
        :type x: numpy.ndarray
        :returns: Sigmoid activated array
        :rtype: numpy.ndarray
        """
        return 1. / (1. + np.exp(-x))
    
    def gradient(self, x):
        """
        Gradient of sigmoid activation.
        
        :param x: Input array
        :type x: numpy.ndarray
        :returns: Gradient array
        :rtype: numpy.ndarray
        """
        s = self.forward(x)
        return s * (1 - s)

class SoftReLUActivation(Activation):
    """
    Soft ReLU (softplus) activation function with gradient.
    
    Soft ReLU is a smooth approximation to ReLU that is differentiable
    everywhere. It computes log(1 + exp(x)) with appropriate gradient.
    """
    
    def forward(self, x):
        """
        Forward pass: soft ReLU activation.
        
        :param x: Input array
        :type x: numpy.ndarray
        :returns: Soft ReLU activated array
        :rtype: numpy.ndarray
        """
        return np.log(1. + np.exp(x))
    
    def gradient(self, x):
        """
        Gradient of soft ReLU activation.
        
        :param x: Input array
        :type x: numpy.ndarray
        :returns: Gradient array (sigmoid of input)
        :rtype: numpy.ndarray
        """
        return 1. / (1. + np.exp(-x))

class NeuralNetwork(Model):
    """
    Neural network model.

    This class implements a neural network model with different basis functions and variable numbers of weight layers.

    :param dimensions: dimensions of the model.
    :type dimensions: list[int]
    :param activations: Activation functions for layers
    :type activations: list[Activation]
    """

    def __init__(self, dimensions, activations):
        """
        Initialise the neural network.
        """
        if len(dimensions) < 2:
            raise ValueError("At least input and output layers must be specified.")
        if len(activations) != len(dimensions) - 1:
            raise ValueError("Number of activation functions must be one less than number of layers.")
        self.dimensions = dimensions
        self.activations = activations
        self.weights = []
        self.biases = []
        for i in range(len(dimensions) - 1):
            weight_matrix = np.random.normal(size=(dimensions[i], dimensions[i + 1])) * np.sqrt(2. / (dimensions[i]+1))
            self.weights.append(weight_matrix)
            bias_vector = np.random.normal(size=dimensions[i+1]) * np.sqrt(2./(dimensions[i]+1))
            self.biases.append(bias_vector)
            
    def predict(self, x):
        """
        Compute output given current basis functions.
        :param x: Input value
        :type x: numpy.ndarray
        :returns: Network output
        :rtype: numpy.ndarray
        """

        nbatch = np.asarray(x).shape[0]
        self.a = []
        self.z = []
        self.a.append(np.asarray(x, dtype=float))
        self.z.append(np.asarray(x, dtype=float))

        for i in range(len(self.weights)):
            self.z.append(self.a[-1] @ self.weights[i] + self.biases[i])
            self.a.append(self.activations[i].forward(self.z[-1]))

        return self.a[-1]
    
    def backward(self, output_gradient):
        """
        Compute gradients using backpropagation through the network.
        
        This method implements the chain rule for computing gradients of the
        loss function with respect to all weight matrices and biases in the network.
        
        Mathematical formulation:
        For layer ℓ and weight matrix W_{ℓ-k}, the gradient is:
        d f_ℓ / d w_{ℓ-k} = [∏_{i=0}^{k-1} W_{ℓ-i} Φ'_{ℓ-i-1}] φ_{ℓ-k-1}^T ⊗ I_{d_{ℓ-k}}
        
        where Φ' is the derivative matrix of the activation function.
        
        :param output_gradient: Gradient of the loss with respect to the output
        :type output_gradient: numpy.ndarray
        :returns: Dictionary containing gradients for weights and biases
        :rtype: dict
        """
        # Initialize gradient storage
        weight_gradients = []
        bias_gradients = []
        
        # Start with the output gradient
        delta = output_gradient
        
        # Backpropagate through each layer (from output to input)
        for i in reversed(range(len(self.weights))):
            # Get the activation derivative for this layer
            activation_derivative = self.activations[i].gradient(self.z[i+1])
            
            # Compute gradient with respect to weights
            # dL/dW_i = (dL/da_i) * (da_i/dz_i) * (dz_i/dW_i)
            # where dz_i/dW_i = a_{i-1}^T
            if i == 0:
                # For first layer, use input
                input_to_layer = self.a[0]
            else:
                input_to_layer = self.a[i]
            
            # Weight gradient: delta * activation_derivative * input^T
            # delta is (batch_size, output_size), activation_derivative is (batch_size, output_size)
            # input_to_layer is (batch_size, input_size)
            # We need: (batch_size, output_size) * (batch_size, output_size) * (batch_size, input_size)^T
            # = (output_size, input_size) for each sample, then sum over batch
            # Weight matrix W_i has shape (input_size, output_size)
            # So gradient should have shape (input_size, output_size)
            weight_grad = np.zeros((self.weights[i].shape[0], self.weights[i].shape[1]))
            for b in range(delta.shape[0]):
                # delta[b] * activation_derivative[b] is (output_size,)
                # input_to_layer[b] is (input_size,)
                # outer product gives (output_size, input_size)
                # But we need (input_size, output_size) to match weight matrix
                weight_grad += np.outer(input_to_layer[b], delta[b] * activation_derivative[b])
            weight_gradients.insert(0, weight_grad)
            
            # Bias gradient: delta * activation_derivative
            # Sum over batch dimension
            bias_grad = np.sum(delta * activation_derivative, axis=0)
            bias_gradients.insert(0, bias_grad)
            
            # Propagate gradient backward through weights
            # delta_{i-1} = W_i^T * (delta_i * activation_derivative_i)
            if i > 0:
                # delta is (batch_size, output_size), activation_derivative is (batch_size, output_size)
                # W_i is (input_size, output_size), so W_i^T is (output_size, input_size)
                # We need: (batch_size, output_size) @ (output_size, input_size) = (batch_size, input_size)
                delta = (delta * activation_derivative) @ self.weights[i].T
        
        return {
            'weight_gradients': weight_gradients,
            'bias_gradients': bias_gradients
        }
    
    def compute_gradient_for_layer(self, layer_idx, output_gradient):
        """
        Compute gradient for a specific layer using the chain rule formula.
        
        This implements the mathematical formula you derived:
        d f_ℓ / d w_{ℓ-k} = [∏_{i=0}^{k-1} W_{ℓ-i} Φ'_{ℓ-i-1}] φ_{ℓ-k-1}^T ⊗ I_{d_{ℓ-k}}
        
        :param layer_idx: Index of the layer to compute gradient for
        :type layer_idx: int
        :param output_gradient: Gradient of the loss with respect to the output
        :type output_gradient: numpy.ndarray
        :returns: Gradient matrix for the specified layer
        :rtype: numpy.ndarray
        """
        if layer_idx >= len(self.weights):
            raise ValueError(f"Layer index {layer_idx} out of range. Network has {len(self.weights)} layers.")
        
        # Start with output gradient
        gradient = output_gradient
        
        # Apply chain rule: multiply by weight matrices and activation derivatives
        # from output layer down to the target layer
        for i in reversed(range(layer_idx + 1, len(self.weights))):
            # Get activation derivative for layer i
            activation_derivative = self.activations[i].gradient(self.z[i+1])
            
            # Apply chain rule: gradient = W^T * (gradient * activation_derivative)
            gradient = (gradient * activation_derivative) @ self.weights[i].T
        
        # Get activation derivative for target layer
        activation_derivative = self.activations[layer_idx].gradient(self.z[layer_idx+1])
        
        # Get input to target layer
        if layer_idx == 0:
            input_to_layer = self.a[0]
        else:
            input_to_layer = self.a[layer_idx]
        
        # Compute final gradient using outer product
        # This implements: φ_{ℓ-k-1}^T ⊗ I_{d_{ℓ-k}} part of the formula
        # Sum over batch dimension
        final_gradient = np.zeros((self.weights[layer_idx].shape[0], self.weights[layer_idx].shape[1]))
        for b in range(gradient.shape[0]):
            # gradient[b] * activation_derivative[b] is (output_size,)
            # input_to_layer[b] is (input_size,)
            # outer product gives (output_size, input_size)
            # But we need (input_size, output_size) to match weight matrix
            final_gradient += np.outer(input_to_layer[b], gradient[b] * activation_derivative[b])
        
        return final_gradient


def finite_difference_gradient(func, x, h=1e-5):
    """
    Compute gradient using finite differences.
    
    This is a numerical method to approximate gradients by computing
    the difference quotient: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    
    This is useful for:
    1. Verifying analytical gradient implementations
    2. Educational purposes to understand gradient computation
    3. Debugging gradient-related issues
    
    :param func: Function to compute gradient for
    :type func: callable
    :param x: Point at which to compute gradient
    :type x: numpy.ndarray
    :param h: Step size for finite differences
    :type h: float
    :returns: Numerical gradient approximation
    :rtype: numpy.ndarray
    
    Examples:
        >>> def f(x): return x**2
        >>> x = np.array([2.0])
        >>> grad = finite_difference_gradient(f, x)
        >>> print(grad)  # Should be close to [4.0]
    """
    x = np.asarray(x, dtype=float)
    gradient = np.zeros_like(x)
    
    # Compute gradient for each dimension
    for i in range(x.size):
        # Create perturbation vectors
        x_plus = x.copy()
        x_minus = x.copy()
        
        # Perturb the i-th element
        x_plus.flat[i] += h
        x_minus.flat[i] -= h
        
        # Compute finite difference
        # Handle both scalar and array outputs
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        
        # If function returns array, take the sum (for activation functions)
        if np.asarray(f_plus).ndim > 0:
            f_plus = np.sum(f_plus)
            f_minus = np.sum(f_minus)
        
        gradient.flat[i] = (f_plus - f_minus) / (2 * h)
    
    return gradient


def finite_difference_jacobian(func, x, h=1e-5):
    """
    Compute Jacobian matrix using finite differences.
    
    This computes the Jacobian matrix of a vector-valued function
    using finite differences. Useful for testing neural network
    gradient computations.
    
    :param func: Vector-valued function to compute Jacobian for
    :type func: callable
    :param x: Point at which to compute Jacobian
    :type x: numpy.ndarray
    :param h: Step size for finite differences
    :type h: float
    :returns: Jacobian matrix (output_size × input_size)
    :rtype: numpy.ndarray
    
    Examples:
        >>> def f(x): return np.array([x[0]**2, x[1]**3])
        >>> x = np.array([2.0, 3.0])
        >>> jacobian = finite_difference_jacobian(f, x)
        >>> print(jacobian)  # Should be close to [[4, 0], [0, 27]]
    """
    x = np.asarray(x, dtype=float)
    output = func(x)
    output = np.asarray(output, dtype=float)
    
    # Initialize Jacobian matrix
    jacobian = np.zeros((output.size, x.size))
    
    # Compute gradient for each input dimension
    for i in range(x.size):
        # Create perturbation vectors
        x_plus = x.copy()
        x_minus = x.copy()
        
        # Perturb the i-th element
        x_plus.flat[i] += h
        x_minus.flat[i] -= h
        
        # Compute finite difference
        output_plus = func(x_plus)
        output_minus = func(x_minus)
        
        jacobian[:, i] = (output_plus - output_minus).flatten() / (2 * h)
    
    return jacobian


def verify_gradient_implementation(analytical_grad, numerical_grad, rtol=1e-5, atol=1e-8):
    """
    Verify that analytical gradient matches numerical gradient.
    
    This function compares analytical and numerical gradients to ensure
    the analytical implementation is correct. This is crucial for
    debugging gradient computations in neural networks.
    
    :param analytical_grad: Analytically computed gradient
    :type analytical_grad: numpy.ndarray
    :param numerical_grad: Numerically computed gradient
    :type numerical_grad: numpy.ndarray
    :param rtol: Relative tolerance for comparison
    :type rtol: float
    :param atol: Absolute tolerance for comparison
    :type atol: float
    :returns: True if gradients match within tolerance
    :rtype: bool
    :raises ValueError: If gradient dimensions don't match
    
    Examples:
        >>> analytical = np.array([4.0, 6.0])
        >>> numerical = np.array([4.0001, 6.0001])
        >>> verify_gradient_implementation(analytical, numerical)
        True
    """
    # Convert to numpy arrays if needed
    analytical_grad = np.asarray(analytical_grad)
    numerical_grad = np.asarray(numerical_grad)
    
    # Check dimension compatibility
    if analytical_grad.shape != numerical_grad.shape:
        raise ValueError(
            f"Gradient dimension mismatch: analytical gradient shape {analytical_grad.shape} "
            f"does not match numerical gradient shape {numerical_grad.shape}"
        )
    
    try:
        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False








class LossFunction:
    """
    Abstract base class for loss functions.
    
    This class defines the interface for all loss functions used in
    neural network training. Loss functions measure the difference
    between predicted and actual values.
    
    Methods
    -------
    forward(predictions, targets) : float
        Compute the loss value
    gradient(predictions, targets) : numpy.ndarray
        Compute the gradient of the loss with respect to predictions
    
    Examples:
        >>> loss = MeanSquaredError()
        >>> loss_value = loss.forward(y_pred, y_true)
        >>> gradient = loss.gradient(y_pred, y_true)
    """
    def __init__(self):
        pass
    
    def forward(self, predictions, targets):
        """
        Compute the loss value.
        
        :param predictions: Model predictions
        :type predictions: numpy.ndarray
        :param targets: True target values
        :type targets: numpy.ndarray
        :returns: Loss value
        :rtype: float
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError
    
    def gradient(self, predictions, targets):
        """
        Compute the gradient of the loss with respect to predictions.
        
        :param predictions: Model predictions
        :type predictions: numpy.ndarray
        :param targets: True target values
        :type targets: numpy.ndarray
        :returns: Gradient of loss with respect to predictions
        :rtype: numpy.ndarray
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error loss function.
    
    This loss function is commonly used for regression problems.
    It computes the average of the squared differences between
    predictions and targets.
    
    Mathematical formulation:
    MSE = (1/n) * Σ(y_pred - y_true)²
    
    The gradient is: dMSE/dy_pred = (2/n) * (y_pred - y_true)
    
    Examples:
        >>> loss = MeanSquaredError()
        >>> y_pred = np.array([[1.0], [2.0], [3.0]])
        >>> y_true = np.array([[1.1], [1.9], [3.1]])
        >>> loss_value = loss.forward(y_pred, y_true)
        >>> gradient = loss.gradient(y_pred, y_true)
    """
    def __init__(self):
        super().__init__()
        self.name = "Mean Squared Error"
    
    def forward(self, predictions, targets):
        """
        Compute the mean squared error.
        
        :param predictions: Model predictions, shape (n_samples, n_outputs)
        :type predictions: numpy.ndarray
        :param targets: True target values, shape (n_samples, n_outputs)
        :type targets: numpy.ndarray
        :returns: Mean squared error
        :rtype: float
        """
        return np.mean((predictions - targets) ** 2)
    
    def gradient(self, predictions, targets):
        """
        Compute the gradient of MSE with respect to predictions.
        
        :param predictions: Model predictions, shape (n_samples, n_outputs)
        :type predictions: numpy.ndarray
        :param targets: True target values, shape (n_samples, n_outputs)
        :type targets: numpy.ndarray
        :returns: Gradient of MSE with respect to predictions
        :rtype: numpy.ndarray
        """
        return (2.0 / predictions.size) * (predictions - targets)


class CrossEntropyLoss(LossFunction):
    """
    Cross-entropy loss function for classification.
    
    This loss function is commonly used for multi-class classification
    problems. It measures the difference between predicted class
    probabilities and true class labels.
    
    Mathematical formulation:
    CE = -Σ y_true * log(y_pred)
    
    The gradient is: dCE/dy_pred = -y_true / y_pred
    
    Examples:
        >>> loss = CrossEntropyLoss()
        >>> y_pred = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> y_true = np.array([[0, 1], [1, 0]])
        >>> loss_value = loss.forward(y_pred, y_true)
        >>> gradient = loss.gradient(y_pred, y_true)
    """
    def __init__(self, epsilon=1e-15):
        """
        Initialize cross-entropy loss.
        
        :param epsilon: Small value to prevent log(0)
        :type epsilon: float
        """
        super().__init__()
        self.epsilon = epsilon
        self.name = "Cross Entropy"
    
    def forward(self, predictions, targets):
        """
        Compute the cross-entropy loss.
        
        :param predictions: Model predictions (probabilities), shape (n_samples, n_classes)
        :type predictions: numpy.ndarray
        :param targets: True target values (one-hot encoded), shape (n_samples, n_classes)
        :type targets: numpy.ndarray
        :returns: Cross-entropy loss
        :rtype: float
        """
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))
    
    def gradient(self, predictions, targets):
        """
        Compute the gradient of cross-entropy with respect to predictions.
        
        :param predictions: Model predictions (probabilities), shape (n_samples, n_classes)
        :type predictions: numpy.ndarray
        :param targets: True target values (one-hot encoded), shape (n_samples, n_classes)
        :type targets: numpy.ndarray
        :returns: Gradient of cross-entropy with respect to predictions
        :rtype: numpy.ndarray
        """
        # Clip predictions to prevent division by zero
        predictions = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        return -targets / (predictions * predictions.shape[0])


class BinaryCrossEntropyLoss(LossFunction):
    """
    Binary cross-entropy loss function for binary classification.
    
    This loss function is used for binary classification problems
    where the output is a single probability value.
    
    Mathematical formulation:
    BCE = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    
    The gradient is: dBCE/dy_pred = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
    
    Examples:
        >>> loss = BinaryCrossEntropyLoss()
        >>> y_pred = np.array([[0.8], [0.3], [0.9]])
        >>> y_true = np.array([[1.0], [0.0], [1.0]])
        >>> loss_value = loss.forward(y_pred, y_true)
        >>> gradient = loss.gradient(y_pred, y_true)
    """
    def __init__(self, epsilon=1e-15):
        """
        Initialize binary cross-entropy loss.
        
        :param epsilon: Small value to prevent log(0)
        :type epsilon: float
        """
        super().__init__()
        self.epsilon = epsilon
        self.name = "Binary Cross Entropy"
    
    def forward(self, predictions, targets):
        """
        Compute the binary cross-entropy loss.
        
        :param predictions: Model predictions (probabilities), shape (n_samples, 1)
        :type predictions: numpy.ndarray
        :param targets: True target values (0 or 1), shape (n_samples, 1)
        :type targets: numpy.ndarray
        :returns: Binary cross-entropy loss
        :rtype: float
        """
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    def gradient(self, predictions, targets):
        """
        Compute the gradient of binary cross-entropy with respect to predictions.
        
        :param predictions: Model predictions (probabilities), shape (n_samples, 1)
        :type predictions: numpy.ndarray
        :param targets: True target values (0 or 1), shape (n_samples, 1)
        :type targets: numpy.ndarray
        :returns: Gradient of binary cross-entropy with respect to predictions
        :rtype: numpy.ndarray
        """
        # Clip predictions to prevent division by zero
        predictions = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        return -(targets / predictions - (1 - targets) / (1 - predictions)) / predictions.shape[0]


class MeanAbsoluteError(LossFunction):
    """
    Mean Absolute Error loss function.
    
    This loss function is used for regression problems and is more
    robust to outliers than mean squared error.
    
    Mathematical formulation:
    MAE = (1/n) * Σ|y_pred - y_true|
    
    The gradient is: dMAE/dy_pred = (1/n) * sign(y_pred - y_true)
    
    Examples:
        >>> loss = MeanAbsoluteError()
        >>> y_pred = np.array([[1.0], [2.0], [3.0]])
        >>> y_true = np.array([[1.1], [1.9], [3.1]])
        >>> loss_value = loss.forward(y_pred, y_true)
        >>> gradient = loss.gradient(y_pred, y_true)
    """
    def __init__(self):
        super().__init__()
        self.name = "Mean Absolute Error"
    
    def forward(self, predictions, targets):
        """
        Compute the mean absolute error.
        
        :param predictions: Model predictions, shape (n_samples, n_outputs)
        :type predictions: numpy.ndarray
        :param targets: True target values, shape (n_samples, n_outputs)
        :type targets: numpy.ndarray
        :returns: Mean absolute error
        :rtype: float
        """
        return np.mean(np.abs(predictions - targets))
    
    def gradient(self, predictions, targets):
        """
        Compute the gradient of MAE with respect to predictions.
        
        :param predictions: Model predictions, shape (n_samples, n_outputs)
        :type predictions: numpy.ndarray
        :param targets: True target values, shape (n_samples, n_outputs)
        :type targets: numpy.ndarray
        :returns: Gradient of MAE with respect to predictions
        :rtype: numpy.ndarray
        """
        return np.sign(predictions - targets) / predictions.size


class HuberLoss(LossFunction):
    """
    Huber loss function (smooth L1 loss).
    
    This loss function combines the benefits of mean squared error
    and mean absolute error. It's quadratic for small errors and
    linear for large errors, making it robust to outliers.
    
    Mathematical formulation:
    Huber = (1/n) * Σ L_δ(y_pred - y_true)
    where L_δ(a) = 0.5 * a² if |a| ≤ δ, else δ * (|a| - 0.5 * δ)
    
    Examples:
        >>> loss = HuberLoss(delta=1.0)
        >>> y_pred = np.array([[1.0], [2.0], [3.0]])
        >>> y_true = np.array([[1.1], [1.9], [3.1]])
        >>> loss_value = loss.forward(y_pred, y_true)
        >>> gradient = loss.gradient(y_pred, y_true)
    """
    def __init__(self, delta=1.0):
        """
        Initialize Huber loss.
        
        :param delta: Threshold parameter
        :type delta: float
        """
        super().__init__()
        self.delta = delta
        self.name = f"Huber Loss (δ={delta})"
    
    def forward(self, predictions, targets):
        """
        Compute the Huber loss.
        
        :param predictions: Model predictions, shape (n_samples, n_outputs)
        :type predictions: numpy.ndarray
        :param targets: True target values, shape (n_samples, n_outputs)
        :type targets: numpy.ndarray
        :returns: Huber loss
        :rtype: float
        """
        error = predictions - targets
        abs_error = np.abs(error)
        
        # Quadratic part for small errors
        quadratic = 0.5 * error ** 2
        # Linear part for large errors
        linear = self.delta * (abs_error - 0.5 * self.delta)
        
        return np.mean(np.where(abs_error <= self.delta, quadratic, linear))
    
    def gradient(self, predictions, targets):
        """
        Compute the gradient of Huber loss with respect to predictions.
        
        :param predictions: Model predictions, shape (n_samples, n_outputs)
        :type predictions: numpy.ndarray
        :param targets: True target values, shape (n_samples, n_outputs)
        :type targets: numpy.ndarray
        :returns: Gradient of Huber loss with respect to predictions
        :rtype: numpy.ndarray
        """
        error = predictions - targets
        abs_error = np.abs(error)
        
        # Gradient is error for small errors, sign(error) * delta for large errors
        gradient = np.where(abs_error <= self.delta, error, self.delta * np.sign(error))
        
        return gradient / predictions.size








class BLM(LM):
    """
    Bayesian Linear Model.
    
    This class implements a Bayesian linear model with a Gaussian prior
    on the weights and Gaussian noise. The model provides uncertainty
    estimates for predictions.
    
    :param X: Input values
    :type X: numpy.ndarray
    :param y: Target values, shape (n_samples, 1)
    :type y: numpy.ndarray
    :param basis: Basis function
    :type basis: Basis
    :param alpha: Scale of prior on parameters (default: 1.0)
    :type alpha: float, optional
    :param sigma2: Noise variance (default: 1.0)
    :type sigma2: float, optional
    """

    def __init__(self, X, y, basis, alpha=1.0, sigma2=1.0):
        """
        Initialize the Bayesian Linear Model.
        
        :param X: Input values
        :type X: numpy.ndarray
        :param y: Target values, shape (n_samples, 1)
        :type y: numpy.ndarray
        :param basis: Basis function
        :type basis: Basis
        :param alpha: Scale of prior on parameters (default: 1.0)
        :type alpha: float, optional
        :param sigma2: Noise variance (default: 1.0)
        :type sigma2: float, optional
        """
        # Call LM constructor to handle 2D target arrays
        super().__init__(X, y, basis)
        self.sigma2 = sigma2
        self.alpha = alpha
        self.basis = basis
        self.Phi = self.basis.Phi(X)
        self.name = 'BLM_'+self.basis.function.__name__
        self.objective_name = 'Negative Marginal Likelihood'

    def set_param(self, name, val, update_fit=True):
        """
        Set a parameter to a given value.
        
        :param name: Name of the parameter to set
        :type name: str
        :param val: New value for the parameter
        :type val: float
        :param update_fit: Whether to refit the model after setting the parameter
        :type update_fit: bool, optional
        """
        if name in self.__dict__:
            if self.__dict__[name] == val:
                update_fit=False
            else:
                self.__dict__[name] = val
        elif name in self.basis.__dict__:
            if self.basis.__dict__[name] == val:
                update_fit=False
            else:
                self.basis.__dict__[name] = val
                self.Phi = self.basis.Phi(self.X)            
        else:
            raise ValueError("Unknown parameter being set.")
        if update_fit:
            self.fit()
        
    def update_QR(self):
        """
        Perform the QR decomposition on the basis matrix.
        
        This method performs QR decomposition on the augmented basis matrix
        that includes the prior regularization term.
        """
        self.Q, self.R = np.linalg.qr(np.vstack([self.Phi, np.sqrt(self.sigma2/self.alpha)*np.eye(self.basis.number)]))

    def fit(self):
        """
        Minimize the objective function with respect to the parameters.
        
        This method computes the posterior mean and covariance of the weights
        using the QR decomposition approach.
        """
        self.update_QR()
        self.QTy = self.Q[:self.y.shape[0], :].T@self.y
        self.mu_w = la.solve_triangular(self.R, self.QTy)
        self.RTinv = la.solve_triangular(self.R, np.eye(self.R.shape[0]), trans='T')
        self.C_w = self.RTinv@self.RTinv.T
        self.update_sum_squares()

    def predict(self, X, full_cov=False):
        """
        Return the result of the prediction function.
        
        :param X: Input features for prediction
        :type X: numpy.ndarray
        :param full_cov: Whether to return full covariance matrix
        :type full_cov: bool, optional
        :returns: Tuple of (predictions, uncertainties)
        :rtype: tuple
        """
        Phi = self.basis.Phi(X)
        # A= R^-T Phi.T
        A = la.solve_triangular(self.R, Phi.T, trans='T')
        mu = A.T@self.QTy
        if full_cov:
            return mu, self.sigma2*A.T@A
        else:
            return mu, self.sigma2*(A*A).sum(0)[:, None]
        
    def update_f(self):
        """
        Update values at the prediction points.
        
        This method computes the posterior mean and variance of the
        function values at the training points.
        """
        self.f_bar = self.Phi@self.mu_w
        self.f_cov = (self.Q[:self.y.shape[0], :]*self.Q[:self.y.shape[0], :]).sum(1)

    def update_sum_squares(self):
        """
        Compute the sum of squares error.
        
        This method computes the sum of squared differences between
        the observed targets and the posterior mean predictions.
        """
        self.update_f()
        self.sum_squares = ((self.y-self.f_bar)**2).sum()
    
    def objective(self):
        """
        Compute the objective function.
        
        For the Bayesian linear model, this is the negative log-likelihood.
        """
        return - self.log_likelihood()

    def update_nll(self):
        """
        Precompute terms needed for the log likelihood.
        
        This method computes the log determinant and quadratic terms
        that are used in the negative log-likelihood calculation.
        """
        self.log_det = self.num_data*np.log(self.sigma2*np.pi*2.)-2*np.log(np.abs(np.linalg.det(self.Q[self.y.shape[0]:, :])))
        self.quadratic = (self.y*self.y).sum()/self.sigma2 - (self.QTy*self.QTy).sum()/self.sigma2
        
    def nll_split(self):
        """
        Compute the determinant and quadratic term of the negative log likelihood.
        
        :returns: Tuple of (log_det, quadratic) terms
        :rtype: tuple
        """
        self.update_nll()
        return self.log_det, self.quadratic
    
    def log_likelihood(self):
        """
        Compute the log likelihood.
        
        :returns: Log likelihood value
        :rtype: float
        """
        self.update_nll()
        return -self.log_det - self.quadratic

##########          Week 8            ##########

    

# Code for loading pgm from http://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
def load_pgm(filename, directory=None, byteorder='>'):
    """
    Return image data from a raw PGM file as a numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    :param filename: Name of the PGM file to load
    :type filename: str
    :param directory: Directory containing the file (optional)
    :type directory: str, optional
    :param byteorder: Byte order for 16-bit images ('>' for big-endian, '<' for little-endian)
    :type byteorder: str, optional
    :returns: Image data as a numpy array
    :rtype: numpy.ndarray
    :raises ValueError: If the file is not a valid raw PGM file
    """
    import re
    from .utils import filename_join
    savename = filename_join(filename, directory)
    with open(savename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\\s(?:\s*#.*[\r\n])*"
            b"(\\d+)\\s(?:\s*#.*[\r\n])*"
            b"(\\d+)\\s(?:\s*#.*[\r\n])*"
            b"(\\d+)\\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(
        buffer,
        dtype='u1' if int(maxval) < 256 else byteorder+'u2',
        count=int(width)*int(height),
        offset=len(header)
    ).reshape((int(height), int(width)))

##########          Week 10          ##########

class LR(ProbMapModel):
    """
    Logistic regression model.

    :param X: Input values
    :type X: numpy.ndarray
    :param y: Target values
    :type y: numpy.ndarray
    :param basis: Basis function
    :type basis: function
    """
    def __init__(self, X, y, basis):
        # Ensure y is 2D (n_samples, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2 or y.shape[1] != 1:
            raise ValueError("y must be 2D with shape (n_samples, 1)")
        
        ProbMapModel.__init__(self, X, y)
        self.basis = basis
        self.Phi = self.basis.Phi(X)
        # Ensure w_star is (n_basis, 1)
        self.w_star = np.zeros((self.basis.number, 1))

    def predict(self, X):
        """
        Generate the prediction function and the basis matrix.

        :param X: Input features for prediction
        :type X: numpy.ndarray
        :returns: Tuple of (predicted probabilities, basis matrix)
        :rtype: tuple
        """
        Phi = self.basis.Phi(X)
        f = Phi @ self.w_star  # (n_samples, 1)
        proba = 1. / (1 + np.exp(-f))
        return proba, Phi

    def gradient(self):
        """
        Generate the gradient of the parameter vector.

        :returns: Gradient vector
        :rtype: numpy.ndarray
        """
        self.update_g()
        y_bool = self.y.flatten().astype(bool)  # Ensure 1D
        grad = np.zeros((self.Phi.shape[1], 1))
        grad += -(self.Phi[y_bool, :].T @ (1 - self.g[y_bool, :]))
        grad += (self.Phi[~y_bool, :].T @ self.g[~y_bool, :])
        return grad.flatten()  # Return 1D array 
    
    def fit(self, learning_rate=0.1, max_iterations=1000, tolerance=1e-6):
        """
        Fit the logistic regression model using gradient descent.
        
        :param learning_rate: Learning rate for gradient descent
        :type learning_rate: float, optional
        :param max_iterations: Maximum number of iterations
        :type max_iterations: int, optional
        :param tolerance: Convergence tolerance
        :type tolerance: float, optional
        """
        for iteration in range(max_iterations):
            old_objective = self.objective()
            gradient = self.gradient()
            # Flatten w_star for optimization, then reshape back
            w_flat = self.w_star.flatten()
            w_flat -= learning_rate * gradient
            self.w_star = w_flat.reshape(self.w_star.shape)
            new_objective = self.objective()
            
            if abs(new_objective - old_objective) < tolerance:
                break

    def compute_g(self, f):
        """
        Compute the transformation and its logarithms.

        :param f: Linear combination of features and weights
        :type f: numpy.ndarray
        :returns: Tuple of (g, log_g, log_gminus)
        :rtype: tuple
        """
        eps = 1e-16
        g = 1./(1+np.exp(f))
        log_g = np.zeros((f.shape))
        log_gminus = np.zeros((f.shape))
        # compute log_g for values out of bound
        bound = np.log(eps)
        ind = f<-bound
        log_g[ind] = -f[ind]
        log_gminus[ind] = eps
        ind = f>bound
        log_g[ind] = eps
        log_gminus[ind] = f[ind]
        ind = np.logical_and(f>=-bound, f<=bound)
        log_g[ind] = np.log(g[ind])
        log_gminus[ind] = np.log(1-g[ind])
        return g, log_g, log_gminus
        
    def update_g(self):
        """
        Compute the prediction function on training data.
        """
        self.f = self.Phi @ self.w_star  # (n_samples, 1)
        self.g, self.log_g, self.log_gminus = self.compute_g(self.f)
        
    def objective(self):
        """
        Compute the objective function (log-likelihood).

        :returns: Log-likelihood value
        :rtype: float
        """
        self.update_g()
        y_bool = self.y.flatten().astype(bool)  # Ensure 1D
        return self.log_g[y_bool, :].sum() + self.log_gminus[~y_bool, :].sum()

class GP(ProbMapModel):
    """
    Gaussian Process model.

    :param X: Input values
    :type X: numpy.ndarray
    :param y: Target values
    :type y: numpy.ndarray
    :param sigma2: Noise variance
    :type sigma2: float
    :param kernel: Covariance function
    :type kernel: function
    """
    def __init__(self, X, y, sigma2, kernel):
        self.sigma2 = sigma2
        self.kernel = kernel
        self.K = kernel.K(X, X)
        self.X = X
        self.y = y
        self.update_inverse()
        self.name = 'GP_'+kernel.function.__name__
        self.objective_name = 'Negative Marginal Likelihood'

    def update_inverse(self):
        """
        Pre-compute the inverse covariance and related quantities.
        """
        self.Kinv = np.linalg.inv(self.K+self.sigma2*np.eye(self.K.shape[0]))
        self.logdetK = np.linalg.det(self.K+self.sigma2*np.eye(self.K.shape[0]))
        self.Kinvy = self.Kinv@self.y
        self.yKinvy = (self.y*self.Kinvy).sum()
    
    def update_kernel_matrix(self):
        """
        Update the kernel matrix when kernel parameters change.
        This should be called after changing kernel parameters.
        """
        self.K = self.kernel.K(self.X, self.X)
        self.update_inverse()

    def fit(self):
        """
        Fit the Gaussian Process model (no-op placeholder).
        """
        pass

    def update_nll(self):
        """
        Precompute the log determinant and quadratic term from the negative log likelihod
        """
        self.log_det = 0.5*(self.K.shape[0]*np.log(2*np.pi) + self.logdetK)
        self.quadratic =  0.5*self.yKinvy
                            
    def nll_split(self):
        """
        Return the two components of the negative log likelihood
        """
        return self.log_det, self.quadratic
    
    def log_likelihood(self):
        """
        Compute the log likelihood.
        
        :returns: Log likelihood value
        :rtype: float
        """
        self.update_nll()
        return -self.log_det - self.quadratic
    
    def objective(self):
        """
        Compute the objective function.
        
        :returns: Objective function value
        :rtype: float
        """
        return -self.log_likelihood()

    def predict(self, X_test, full_cov=False):
        """
        Give a mean and a variance of the prediction.
        
        :param X_test: Input features for prediction
        :type X_test: numpy.ndarray
        :param full_cov: Whether to return full covariance matrix
        :type full_cov: bool, optional
        :returns: Tuple of (mean, variance)
        :rtype: tuple
        """
        K_star = self.kernel.K(self.X, X_test)
        A = self.Kinv@K_star
        mu_f = A.T@self.y
        k_starstar = self.kernel.diag(X_test)
        c_f = k_starstar - (A*K_star).sum(0)[:, np.newaxis]
        return mu_f, c_f
        
def posterior_f(self, X_test):
    """
    Compute the posterior distribution for f in the GP
    """
    K_star = self.kernel.K(self.X, X_test)
    A = self.Kinv@K_star
    mu_f = A.T@self.y
    K_starstar = self.kernel.K(X_test, X_test)
    C_f = K_starstar - A.T@K_star
    return mu_f, C_f

def update_inverse(self):
    """
    Update the inverse covariance in a numerically more stable manner
    """
    # Perform Cholesky decomposition on matrix (scipy returns upper triangular)
    self.R = la.cholesky(self.K + self.sigma2*np.eye(self.K.shape[0]))
    # compute the log determinant from Cholesky decomposition
    self.logdetK = 2*np.log(np.diag(self.R)).sum()
    # compute y^\top K^{-1}y from Cholesky factor
    # For K = R^T R, we have K^{-1} = R^{-1} R^{-T}
    # So y^T K^{-1} y = y^T R^{-1} R^{-T} y = ||R^{-T} y||^2
    self.Rinvy = la.solve_triangular(self.R, self.y, trans='T')
    self.yKinvy = (self.Rinvy**2).sum()
    
    # compute the inverse of the upper triangular Cholesky factor
    self.Rinv = la.solve_triangular(self.R, np.eye(self.K.shape[0]))
    self.Kinv = self.Rinv@self.Rinv.T
    
    # Add missing Kinvy attribute for compatibility with predict method
    self.Kinvy = self.Kinv@self.y


class Kernel():
    """
    Covariance function
    
    :param function: covariance function
    :type function: function
    :param name: name of covariance function
    :type name: string
    :param shortname: abbreviated name of covariance function
    :type shortname: string
    :param formula: latex formula of covariance function
    :type formula: string
    :param function: covariance function
    :type function: function
    :param \\**kwargs:
        See below

    :Keyword Arguments:
        * """

    def __init__(self, function, name=None, shortname=None, formula=None, **kwargs):        
        self.function=function
        self.formula = formula
        self.name = name
        self.shortname = shortname
        self.parameters=kwargs
        
    def K(self, X, X2=None):
        """
        Compute the full covariance function given a kernel function for two data points.
        
        :param X: First set of data points
        :type X: numpy.ndarray
        :param X2: Second set of data points (optional, defaults to X)
        :type X2: numpy.ndarray, optional
        :returns: Covariance matrix
        :rtype: numpy.ndarray
        """
        if X2 is None:
            X2 = X
        K = np.zeros((X.shape[0], X2.shape[0]))
        for i in np.arange(X.shape[0]):
            for j in np.arange(X2.shape[0]):
                K[i, j] = self.function(X[i, :], X2[j, :], **self.parameters)

        return K

    def diag(self, X):
        """
        Compute the diagonal of the covariance function
        
        :param X: Data points to compute the diagonal for
        :type X: numpy.ndarray
        :returns: Diagonal of the covariance matrix
        :rtype: numpy.ndarray
        """
        diagK = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):            
            diagK[i] = self.function(X[i, :], X[i, :], **self.parameters)
        return diagK

    def _repr_html_(self):
        raise NotImplementedError

    
def eq_cov(x, x_prime, variance=1., lengthscale=1.):
    """
    Exponentiated quadratic covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param lengthscale: Lengthscale parameter
    :type lengthscale: float, optional
    :returns: Covariance value
    :rtype: float
    """
    diffx = x - x_prime
    return variance*np.exp(-0.5*np.dot(diffx, diffx)/lengthscale**2)

def ou_cov(x, x_prime, variance=1., lengthscale=1.):
    """
    Exponential covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param lengthscale: Lengthscale parameter
    :type lengthscale: float, optional
    :returns: Covariance value
    :rtype: float
    """
    diffx = x - x_prime
    return variance*np.exp(-np.sqrt(np.dot(diffx, diffx))/lengthscale)

def matern32_cov(x, x_prime, variance=1., lengthscale=1.):
    """
    Matern 3/2 covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param lengthscale: Lengthscale parameter
    :type lengthscale: float, optional
    :returns: Covariance value
    :rtype: float
    """
    diffx = x - x_prime
    r_norm = np.sqrt(np.dot(diffx, diffx))/lengthscale
    np.sqrt3r_norm = r_norm*np.sqrt(3)
    return variance*(1+np.sqrt3r_norm)*np.exp(-np.sqrt3r_norm)

def matern52_cov(x, x_prime, variance=1., lengthscale=1.):
    """
    Matern 5/2 covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param lengthscale: Lengthscale parameter
    :type lengthscale: float, optional
    :returns: Covariance value
    :rtype: float
    """
    diffx = x - x_prime
    r_norm = np.sqrt(np.dot(diffx, diffx))/lengthscale
    sqrt5r_norm = r_norm*np.sqrt(5)
    return variance*(1+sqrt5r_norm+sqrt5r_norm*sqrt5r_norm/3)*np.exp(-sqrt5r_norm)

def mlp_cov(x, x_prime, variance=1., w=1., b=5., alpha=1.):
    """
    Covariance function for a MLP based neural network.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param w: Overall scale of the weights on the input.
    :type w: float, optional
    :param b: Overall scale of the bias on the input
    :type b: float, optional
    :param alpha: Smoothness of the relu activation
    :type alpha: float, optional
    :returns: Covariance value
    :rtype: float
    """
    inner = np.dot(x, x_prime)*w + b
    norm = np.sqrt(np.dot(x, x)*w + b + alpha)*np.sqrt(np.dot(x_prime, x_prime)*w + b+alpha)
    arg = np.clip(inner/norm, -1, 1) # clip as numerically can be > 1
    theta = np.arccos(arg)
    return variance*0.5*(1. - theta/np.pi)      

def icm_cov(x, x_prime, B, subkernel, **kwargs):
    """
    Intrinsic coregionalization model. First index is outputs considered for covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param B: Coregionalization matrix
    :type B: numpy.ndarray
    :param subkernel: Sub-kernel function
    :type subkernel: function
    :param **kwargs: Additional arguments for the sub-kernel
    :type **kwargs: dict
    :returns: Covariance value
    :rtype: float
    """
    # Validate and cast first column to integer for indexing
    i_float = x[0]
    i_prime_float = x_prime[0]
    
    # Check if values are integer-valued (even if stored as float)
    if not (i_float == int(i_float) and i_prime_float == int(i_prime_float)):
        raise ValueError(f"First column of x must be integer-valued for indexing. Got x[0]={i_float}, x_prime[0]={i_prime_float}")
    
    # Cast to integer for indexing
    i = int(i_float)
    i_prime = int(i_prime_float)
    
    return B[i, i_prime]*subkernel(x[1:], x_prime[1:], **kwargs)

def lmc_cov(x, x_prime, B_list, subkernel_list, **kwargs):
    """
    Linear Model of Coregionalisation. Combines multiple ICM components.
    
    The LMC is defined as: k(x, x') = Σ_q B_q[i, i'] * k_q(x[1:], x'[1:])
    where B_q are coregionalization matrices and k_q are subkernels.
    
    This is implemented as a sum of ICM components using add_cov.
    Each ICM component handles its own integer validation.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param B_list: List of coregionalization matrices
    :type B_list: list of numpy.ndarray
    :param subkernel_list: List of sub-kernel functions
    :type subkernel_list: list of functions
    :param **kwargs: Additional arguments for the sub-kernels
    :type **kwargs: dict
    :returns: Covariance value
    :rtype: float
    """
    if len(B_list) != len(subkernel_list):
        raise ValueError(f"Number of coregionalization matrices ({len(B_list)}) must match number of subkernels ({len(subkernel_list)})")
    
    # Create ICM components - each will handle its own validation
    icm_components = []
    icm_kwargs = []
    
    for B, subkernel in zip(B_list, subkernel_list):
        # Create an ICM component using the actual icm_cov function
        def icm_component(x_inner, x_prime_inner, **inner_kwargs):
            return icm_cov(x_inner, x_prime_inner, B, subkernel, **inner_kwargs)
        
        icm_components.append(icm_component)
        icm_kwargs.append(kwargs)
    
    # Use add_cov to sum the ICM components
    return add_cov(x, x_prime, icm_components, icm_kwargs)

def slfm_cov(x, x_prime, W, subkernel, **kwargs):
    """
    Semi-parametric latent factor model covariance function. First index is the output of the covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param W: Latent factor matrix
    :type W: numpy.ndarray
    :param subkernel: Sub-kernel function
    :type subkernel: function
    :param **kwargs: Additional arguments for the sub-kernel
    :type **kwargs: dict
    :returns: Covariance value
    :rtype: float
    """
    W = np.asarray(W)
    B = W@W.T
    return icm_cov(x, x_prime, B, subkernel, **kwargs)

    

def relu_cov(x, x_prime, variance=1., scale=1., w=1., b=5., alpha=1e-6):
    """
    Covariance function for a ReLU based neural network (arc-cosine kernel).
    
    This implements the arc-cosine kernel which is the covariance function
    for ReLU neural networks in the infinite width limit.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall scale of the covariance
    :type variance: float, optional
    :param scale: Overall scale of the weights on the input
    :type scale: float, optional
    :param w: Weight scale parameter
    :type w: float, optional
    :param b: Bias parameter
    :type b: float, optional
    :param alpha: Small positive constant for numerical stability
    :type alpha: float, optional
    :returns: Covariance value
    :rtype: float
    """
    # Compute dot products with scaling
    x_scaled = x * scale
    x_prime_scaled = x_prime * scale
    
    # Compute the inner products
    inner = np.dot(x_scaled, x_prime_scaled) + b
    inner_1 = np.dot(x_scaled, x_scaled) + b
    inner_2 = np.dot(x_prime_scaled, x_prime_scaled) + b
    
    # Add small constant for numerical stability
    inner_1 = inner_1 + alpha
    inner_2 = inner_2 + alpha
    
    # Compute norms
    norm_1 = np.sqrt(inner_1)
    norm_2 = np.sqrt(inner_2)
    
    # Compute cosine of angle between vectors
    cos_theta = inner / (norm_1 * norm_2)
    
    # Clip to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Compute the arc-cosine kernel
    theta = np.arccos(cos_theta)
    
    # The arc-cosine kernel formula
    k_relu = (norm_1 * norm_2) / (2 * np.pi) * (np.pi - theta)
    
    return variance * k_relu 


def polynomial_cov(x, x_prime, variance=1., degree=2., w=1., b=1.):
    """
    Polynomial covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param degree: Degree of the polynomial
    :type degree: int, optional
    :param w: Overall scale of the weights on the input.
    :type w: float, optional
    :param b: Overall scale of the bias on the input
    :type b: float, optional
    :returns: Covariance value
    :rtype: float
    """
    return variance*(np.dot(x, x_prime)*w + b)**degree

def linear_cov(x, x_prime, variance=1.):
    """
    Linear covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :returns: Covariance value
    :rtype: float
    """
    return variance*np.dot(x, x_prime)

def bias_cov(x, x_prime, variance=1.):
    """
    Bias covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :returns: Covariance value
    :rtype: float
    """
    return variance

def mlp_cov(x, x_prime, variance=1., w=1., b=1.):
    """
    MLP covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param w: Overall scale of the weights on the input.
    :type w: float, optional
    :param b: Overall scale of the bias on the input
    :type b: float, optional
    :returns: Covariance value
    :rtype: float
    """
    return variance*np.arcsin((w*np.dot(x, x_prime) + b)/np.sqrt((np.dot(x, x)*w +b + 1)*(np.dot(x_prime, x_prime)*w + b + 1)))

def sinc_cov(x, x_prime, variance=1., w=1.):
    """
    Sinc covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param w: Overall scale of the weights on the input.
    :type w: float, optional
    :returns: Covariance value
    :rtype: float
    """
    r = np.linalg.norm(x-x_prime, 2)
    return variance*np.sinc(np.pi*w*r)

def ou_cov(x, x_prime, variance=1., lengthscale=1.):
    """
    Ornstein Uhlenbeck covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param lengthscale: Lengthscale parameter
    :type lengthscale: float, optional
    :returns: Covariance value
    :rtype: float
    """
    r = np.linalg.norm(x-x_prime, 2)
    return variance*np.exp(-r/lengthscale)        

def brownian_cov(t, t_prime, variance=1.):
    """
    Brownian motion covariance function.
    
    :param t: First time point
    :type t: float
    :param t_prime: Second time point
    :type t_prime: float
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :returns: Covariance value
    :rtype: float
    """
    if t>=0 and t_prime>=0:
        return variance*np.min([t, t_prime])
    else:
        raise ValueError("For Brownian motion covariance only positive times are valid.")

def periodic_cov(x, x_prime, variance=1., lengthscale=1., w=1.):
    """
    Periodic covariance function
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param lengthscale: Lengthscale parameter
    :type lengthscale: float, optional
    :param w: Overall scale of the weights on the input.
    :type w: float, optional
    :returns: Covariance value
    :rtype: float
    """
    r = np.linalg.norm(x-x_prime, 2)
    return variance*np.exp(-2./(lengthscale*lengthscale)*np.sin(np.pi*r*w)**2)

def ratquad_cov(x, x_prime, variance=1., lengthscale=1., alpha=1.):
    """
    Rational quadratic covariance function
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall variance of the covariance
    :type variance: float, optional
    :param lengthscale: Lengthscale parameter
    :type lengthscale: float, optional
    :param alpha: Smoothness parameter
    :type alpha: float, optional
    :returns: Covariance value
    :rtype: float
    """
    r = np.linalg.norm(x-x_prime, 2)
    return variance*(1. + r*r/(2*alpha*lengthscale*lengthscale))**-alpha

def prod_cov(x, x_prime, kerns, kern_args):
    """
    Product covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param kerns: List of kernel functions
    :type kerns: list of functions
    :param kern_args: List of dictionaries of kernel arguments
    :type kern_args: list of dicts
    :returns: Product of covariance values
    :rtype: float
    """
    k = 1.
    for kern, kern_arg in zip(kerns, kern_args):
        k*=kern(x, x_prime, **kern_arg)
    return k
        
def add_cov(x, x_prime, kerns, kern_args):
    """
    Additive covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param kerns: List of kernel functions
    :type kerns: list of functions
    :param kern_args: List of dictionaries of kernel arguments
    :type kern_args: list of dicts
    :returns: Summed covariance value
    :rtype: float
    """
    k = 0.
    for kern, kern_arg in zip(kerns, kern_args):
        k+=kern(x, x_prime, **kern_arg)
    return k

def basis_cov(x, x_prime, basis):
    """
    Basis function covariance.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param basis: Basis function object
    :type basis: Basis
    :returns: Covariance value
    :rtype: float
    """
    return (basis.Phi(x)*basis.Phi(x_prime)).sum()

def contour_data(model, data, length_scales, log_SNRs):
    """
    Evaluate the GP objective function for a given data set for a range of
    signal to noise ratios and a range of lengthscales.

    :param model: The GP model to evaluate
    :type model: GP
    :param data: The data dictionary containing 'Y'
    :type data: dict
    :param length_scales: a list of length scales to explore for the contour plot.
    :type length_scales: list of floats
    :param log_SNRs: a list of base 10 logarithm signal to noise ratios to explore for the contour plot.
    :type log_SNRs: list of floats
    :returns: Array of log-likelihood values for different length scales and SNRs
    :rtype: numpy.ndarray
    """

    
    lls = []
    total_var = np.var(data['Y'])
    for log_SNR in log_SNRs:
        SNR = 10.**log_SNR
        noise_var = total_var / (1. + SNR)
        signal_var = total_var - noise_var
        model.kern['.*variance'] = signal_var
        model.likelihood.variance = noise_var
        length_scale_lls = []

        for length_scale in length_scales:
            model['.*lengthscale'] = length_scale
            length_scale_lls.append(model.log_likelihood())

        lls.append(length_scale_lls)

    return np.asarray(lls)

def radial_multivariate(x, num_basis=4, width=None, random_state=0):
    """
    Multivariate radial basis function (RBF) for multi-dimensional input.

    :param x: Input features, shape (n_samples, n_features)
    :type x: numpy.ndarray
    :param num_basis: Number of radial basis functions
    :type num_basis: int, optional
    :param width: Width parameter for the Gaussian functions. If None, auto-computed.
    :type width: float, optional
    :param random_state: Seed for reproducible center placement
    :type random_state: int, optional
    :returns: Radial basis matrix, shape (n_samples, num_basis)
    :rtype: numpy.ndarray
    """
    x = np.asarray(x, dtype=float)
    n_samples, n_features = x.shape
    rng = np.random.RandomState(random_state)
    # Place centers randomly within the min/max of the data
    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    centres = rng.uniform(mins, maxs, size=(num_basis, n_features))
    if width is None:
        from scipy.spatial.distance import cdist
        # Calculate distances between all pairs of centers
        center_dists = cdist(centres, centres)
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(center_dists, np.inf)
        # Use average distance to nearest neighbor as width
        nearest_dists = np.min(center_dists, axis=1)
        width = np.mean(nearest_dists) if len(nearest_dists) > 0 else 1.0
        # Optionally scale down the width for better separation
        width = width * 0.5
    Phi = np.zeros((n_samples, num_basis))
    for i in range(num_basis):
        diff = x - centres[i]
        Phi[:, i] = np.exp(-0.5 * np.sum(diff**2, axis=1) / width**2)
    return Phi


def generate_cluster_data(n_points_per_cluster=30):
    """Generate synthetic data with clear cluster structure for educational purposes"""
    # Define cluster centres in 2D space
    cluster_centres = np.array([[2.5, 2.5], [-2.5, -2.5], [2.5, -2.5]])
    
    # Generate data points around each center
    data_points = []
    for center in cluster_centres:
        # Generate points with some spread around each center
        cluster_points = np.random.normal(loc=center, scale=0.8, size=(n_points_per_cluster, 2))
        data_points.append(cluster_points)
    
    return np.vstack(data_points)


class ClusterModel():
    pass

def dist2(X1, X2):
    """
    Return the squared distance matrix between two 2-D arrays.
    
    Key insight: ||x - y||² = ||x||² + ||y||² - 2⟨x,y⟩
    
    Why? Expand (x-y)·(x-y) = x·x - 2x·y + y·y
    """

    return (np.sum(X1*X1, axis=1, keepdims=True)
            + np.sum(X2*X2, axis=1) 
            - 2*X1@X2.T)

        
def kmeans_assignments(Y, centres):
    """Assign each point to nearest centre"""
    sq_distances = ((Y[:, np.newaxis] - centres[np.newaxis, :])**2).sum(axis=2)
    return np.argmin(sq_distances, axis=1)
    
def kmeans_update(Y, centres):
    """Perform an update of centre locations for k-means algorithm"""
    assignments = kmeans_assignments(Y, centres)
    
    # Update centres to be mean of assigned points
    new_centres = np.array([Y[assignments == k].mean(axis=0) 
                           for k in range(len(centres))])
    
    return new_centres, assignments

def kmeans_objective(Y, centres, assignments=None):
    """Calculate the k-means objective function (sum of squared distances)"""

    if assignments is None:
        assignments = kmeans_assignments(Y, centres)
        
    total_error = 0
    for k in range(len(centres)):
        cluster_points = Y[assignments == k]
        if len(cluster_points) > 0:
            sq_distances = dist2(cluster_points, centres[k:k+1, :])
            total_error += np.sum(sq_distances)
    return total_error


class WardsMethod(ClusterModel):
    def __init__(self, X):
        """
        Simple implementation of Ward's hierarchical clustering
        
        Parameters:
        X: numpy array of shape (n_samples, n_features)
        """
        self.numdata = len(X)
        
        # Initialize each point as its own cluster
        self.clusters = {i: [i] for i in range(self.numdata)}
        self.centroids = {i: X[i].copy() for i in range(self.numdata)}
        self.cluster_sizes = {i: 1 for i in range(self.numdata)}
        
        # Track merges for dendrogram
        self.merges = []
        self.distances = []
        
    def ward_distance(self, cluster_a, cluster_b):
        """
        Calculate Ward distance between two clusters
        """
        n_a = self.cluster_sizes[cluster_a]
        n_b = self.cluster_sizes[cluster_b]
        
        centroid_a = self.centroids[cluster_a]
        centroid_b = self.centroids[cluster_b]
        
        # Ward distance formula
        weight = (n_a * n_b) / (n_a + n_b)
        distance = np.sum((centroid_a - centroid_b) ** 2)
        
        return np.sqrt(weight * distance)
    
    def find_closest_clusters(self):
        """
        Find the pair of clusters with minimum Ward distance
        """
        min_distance = float('inf')
        closest_pair = None
        
        cluster_ids = list(self.clusters.keys())
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cluster_a = cluster_ids[i]
                cluster_b = cluster_ids[j]
                
                distance = self.ward_distance(cluster_a, cluster_b)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (cluster_a, cluster_b)
        
        return closest_pair, min_distance
    
    def merge_clusters(self, cluster_a, cluster_b):
        """
        Merge two clusters and update centroids
        """
        n_a = self.cluster_sizes[cluster_a]
        n_b = self.cluster_sizes[cluster_b]
        
        # Calculate new centroid as weighted mean
        centroid_a = self.centroids[cluster_a]
        centroid_b = self.centroids[cluster_b]
        new_centroid = (n_a * centroid_a + n_b * centroid_b) / (n_a + n_b)
        
        # Create new cluster ID
        new_cluster_id = max(self.clusters.keys()) + 1
        
        # Merge point lists
        new_cluster_points = self.clusters[cluster_a] + self.clusters[cluster_b]
        
        # Update data structures
        self.clusters[new_cluster_id] = new_cluster_points
        self.centroids[new_cluster_id] = new_centroid
        self.cluster_sizes[new_cluster_id] = n_a + n_b
        
        # Remove old clusters
        del self.clusters[cluster_a]
        del self.clusters[cluster_b]
        del self.centroids[cluster_a]
        del self.centroids[cluster_b]
        del self.cluster_sizes[cluster_a]
        del self.cluster_sizes[cluster_b]
        
        return new_cluster_id
    
    def fit(self):
        """
        Perform Ward's hierarchical clustering
        """
        step = 0
        
        while len(self.clusters) > 1:
            # Find closest pair
            (cluster_a, cluster_b), distance = self.find_closest_clusters()
            
            print(f"Step {step}: Merging clusters {cluster_a} and {cluster_b}, "
                  f"distance = {distance:.3f}")
            
            # Record merge for dendrogram
            self.merges.append([cluster_a, cluster_b])
            self.distances.append(distance)
            
            # Merge clusters
            new_cluster_id = self.merge_clusters(cluster_a, cluster_b)
            
            step += 1
        
        return self
    
    def get_linkage_matrix(self):
        """
        Convert to scipy linkage matrix format for dendrogram plotting
        """
        linkage_matrix = []
        
        # Track cluster sizes and scipy IDs
        cluster_sizes = {i: 1 for i in range(self.numdata)}
        cluster_to_scipy = {i: i for i in range(self.numdata)}
        next_scipy_id = self.numdata
        
        for i, (merge, distance) in enumerate(zip(self.merges, self.distances)):
            cluster_a, cluster_b = merge
            
            # Get scipy IDs for the clusters being merged
            scipy_a = cluster_to_scipy.get(cluster_a, cluster_a)
            scipy_b = cluster_to_scipy.get(cluster_b, cluster_b)
            
            # Get cluster sizes
            size_a = cluster_sizes.get(cluster_a, 1)
            size_b = cluster_sizes.get(cluster_b, 1)
            
            # Create the linkage matrix row
            linkage_matrix.append([scipy_a, scipy_b, distance, size_a + size_b])
            
            # Update tracking for the new merged cluster
            new_cluster_id = self.numdata + i
            cluster_sizes[new_cluster_id] = size_a + size_b
            cluster_to_scipy[new_cluster_id] = new_cluster_id
        
        return np.array(linkage_matrix)

def ppca_eig(Y, q):
    """Perform probabilistic principle component analysis"""
    
    Y_cent = Y - Y.mean(0)

    # Comute covariance
    S = np.dot(Y_cent.T, Y_cent)/Y.shape[0]
    lambd, U = np.linalg.eig(S)

    # Choose number of eigenvectors
    sigma2 = np.sum(lambd[q:])/(Y.shape[1]-q)
    l = np.sqrt(lambd[:q]-sigma2)
    W = U[:, :q]*l[None, :]
    return W, sigma2

def ppca_svd(Y, q, center=True):
    """Probabilistic PCA through singular value decomposition"""
    # remove mean
    if center:
        Y_cent = Y - Y.mean(0)
    else:
        Y_cent = Y
    import scipy as sp
    # Comute singluar values, discard 'R' as we will assume orthogonal
    U, sqlambd, _ = sp.linalg.svd(Y_cent.T,full_matrices=False)
    lambd = (sqlambd**2)/Y.shape[0]
    # Compute residual and extract eigenvectors
    sigma2 = np.sum(lambd[q:])/(Y.shape[1]-q)
    ell = np.sqrt(lambd[:q]-sigma2)
    return U[:, :q], ell, sigma2

def ppca_posterior(Y, U, ell, sigma2, center=True):
    """Posterior computation for the latent variables given the eigendecomposition."""
    if center:
        Y_cent = Y - Y.mean(0)
    else:
        Y_cent = Y
    C_x = np.diag(sigma2/(sigma2+ell**2))
    d = ell/(sigma2+ell**2)
    mu_x = np.dot(Y_cent, U)*d[None, :]
    return mu_x, C_x

    
def kruskal_stress(D_original, D_reduced):
    """Compute Kruskal's stress between original and reduced distances"""
    numerator = np.sum((D_original - D_reduced)**2)
    denominator = np.sum(D_original**2)
    return np.sqrt(numerator/denominator)


def generate_swiss_roll(n_points=1000, noise=0.05):
    """Generate Swiss roll dataset"""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_points))
    y = 21 * np.random.rand(n_points)
    x = t * np.cos(t)
    z = t * np.sin(t)
    X = np.stack([x, y, z])
    X += noise * np.random.randn(*X.shape)
    return X.T, t
