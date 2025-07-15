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


def filename_join(filename, directory=None):
    """
    Join a filename to a directory and create directory if it doesn't exist.
    
    This utility function ensures that the target directory exists before
    attempting to create files, which is useful for saving figures, animations,
    and other outputs during tutorials.
    
    :param filename: The name of the file to create
    :type filename: str
    :param directory: The directory path. If None, returns just the filename.
        If the directory doesn't exist, it will be created.
    :type directory: str, optional
    :returns: The full path to the file
    :rtype: str
    
    Examples:
        >>> filename_join("plot.png", "figures")
        'figures/plot.png'
        >>> filename_join("data.csv")
        'data.csv'
    """
    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)
    return filename

def write_animation(anim, filename, directory=None, **kwargs):
    """
    Write a matplotlib animation to a file.
    
    This function saves animations (e.g., from matplotlib.animation) to files
    in the specified directory, creating the directory if necessary.
    
    :param anim: The animation object to save
    :type anim: matplotlib.animation.Animation
    :param filename: The name of the output file (e.g., 'animation.gif', 'animation.mp4')
    :type filename: str
    :param directory: The directory to save the animation in. If None, saves in current directory.
    :type directory: str, optional
    :param **kwargs: Additional arguments passed to anim.save()
    :type **kwargs: dict
    
    Examples:
        >>> import matplotlib.animation as animation
        >>> # Create animation...
        >>> write_animation(anim, "learning_process.gif", "animations")
    """
    savename = filename_join(filename, directory)
    anim.save(savename, **kwargs)

def write_animation_html(anim, filename, directory=None):
    """
    Save a matplotlib animation as an HTML file with embedded JavaScript.
    
    This function creates an HTML file containing the animation that can be
    viewed in a web browser. The animation is embedded as JavaScript code.
    
    :param anim: The animation object to save
    :type anim: matplotlib.animation.Animation
    :param filename: The name of the output HTML file
    :type filename: str
    :param directory: The directory to save the HTML file in. If None, saves in current directory.
    :type directory: str, optional
    
    Examples:
        >>> import matplotlib.animation as animation
        >>> # Create animation...
        >>> write_animation_html(anim, "learning_process.html", "animations")
    """
    savename = filename_join(filename, directory)
    f = open(savename, 'w')
    f.write(anim.to_jshtml())
    f.close()

def write_figure(filename, figure=None, directory=None, frameon=None, **kwargs):
    """
    Save a matplotlib figure to a file with proper formatting.
    
    This function saves figures with transparent background by default,
    which is useful for presentations and publications. The function
    automatically creates the target directory if it doesn't exist.
    
    :param filename: The name of the output file (e.g., 'plot.png', 'figure.pdf')
    :type filename: str
    :param figure: The figure to save. If None, saves the current figure.
    :type figure: matplotlib.figure.Figure, optional
    :param directory: The directory to save the figure in. If None, saves in current directory.
    :type directory: str, optional
    :param frameon: Whether to draw a frame around the figure. If None, uses matplotlib default.
    :type frameon: bool, optional
    :param **kwargs: Additional arguments passed to plt.savefig() or figure.savefig()
    :type **kwargs: dict
    
    Examples:
        >>> plt.plot([1, 2, 3], [1, 4, 2])
        >>> write_figure("linear_plot.png", directory="figures")
        >>> write_figure("presentation_plot.png", transparent=False, dpi=300)
    """
    savename = filename_join(filename, directory)
    if 'transparent' not in kwargs:
        kwargs['transparent'] = True
    if figure is None:
        plt.savefig(savename, **kwargs)
    else:
        figure.savefig(savename, **kwargs)
    
##########          Week 2          ##########
def init_perceptron(x_plus, x_minus, seed=1000001):
    """
    Initialize the perceptron algorithm with random weights and bias.
    
    The perceptron is a simple binary classifier that learns a linear decision boundary.
    This function initializes the weight vector w and bias b by randomly selecting
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
    choose_plus = np.random.rand(1)>0.5
    if choose_plus:
        # generate a random point from the positives
        index = np.random.randint(0, x_plus.shape[0])
        x_select = x_plus[index, :]
        w = x_plus[index, :] # set the normal vector to that point.
        b = 1
    else:
        # generate a random point from the negatives
        index = np.random.randint(0, x_minus.shape[0])
        x_select = x_minus[index, :]
        w = -x_minus[index, :] # set the normal vector to minus that point.
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
    choose_plus = np.random.rand(1)>0.5
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
    :param y: Target values, shape (n_samples,)
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
        :param y: Target values, shape (n_samples,)
        :type y: numpy.ndarray
        :param basis: Basis function object that provides the feature transformation
        :type basis: Basis
        """
        ProbModel.__init__(self)
        self.y = y
        self.num_data = y.shape[0]
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
        

    
class BLM(LM):
    """
    Bayesian Linear Model.
    
    This class implements a Bayesian linear model with a Gaussian prior
    on the weights and Gaussian noise. The model provides uncertainty
    estimates for predictions.
    
    :param X: Input values
    :type X: numpy.ndarray
    :param y: Target values
    :type y: numpy.ndarray
    :param alpha: Scale of prior on parameters
    :type alpha: float
    :param sigma2: Noise variance
    :type sigma2: float
    :param basis: Basis function
    :type basis: function
    """

    def __init__(self, X, y, alpha, sigma2, basis):
        """
        Initialize the Bayesian Linear Model.
        
        :param X: Input values
        :type X: numpy.ndarray
        :param y: Target values
        :type y: numpy.ndarray
        :param alpha: Scale of prior on parameters
        :type alpha: float
        :param sigma2: Noise variance
        :type sigma2: float
        :param basis: Basis function
        :type basis: function
        """
        ProbMapModel.__init__(self, X, y)
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
        self.Q, self.R = np.linalg.qr(vstack([self.Phi, np.sqrt(self.sigma2/self.alpha)*np.eye(self.basis.number)]))

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
        ProbMapModel.__init__(self, X, y)
        self.basis = basis
        self.Phi = self.basis.Phi(X)
        self.w_star = np.zeros(self.basis.number)
        
    def predict(self, X):
        """
        Generate the prediction function and the basis matrix.

        :param X: Input features for prediction
        :type X: numpy.ndarray
        :returns: Tuple of (predicted probabilities, basis matrix)
        :rtype: tuple
        """
        Phi = self.basis.Phi(X)
        f = Phi@self.w_star
        return 1./(1+np.exp(-f)), Phi

    def gradient(self):
        """
        Generate the gradient of the parameter vector.

        :returns: Gradient vector
        :rtype: numpy.ndarray
        """
        self.update_g()
        dw = -(self.Phi[self.y.values, :]*(1-self.g[self.y.values, :])).sum(0)
        dw += (self.Phi[~self.y.values, :]*self.g[~self.y.values, :]).sum(0)
        return dw[:, None]

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
        ind = (f>=-bound & f<=bound)
        log_g[ind] = np.log(self.g[ind])
        log_gminus[ind] = np.log(1-self.g[ind])
        return g, log_g, log_gminus
        
    def update_g(self):
        """
        Compute the prediction function on training data.
        """
        self.f = self.Phi@self.w_star
        self.g, self.log_g, self.log_gminus = self.compute_g(self.f)
        
    def objective(self):
        """
        Compute the objective function (log-likelihood).

        :returns: Log-likelihood value
        :rtype: float
        """
        self.update_g()
        return self.log_g[self.y.values, :].sum() + self.log_gminus[~self.y.values, :].sum()
    
##########          Week 12          ##########
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
        c_f = k_starstar - (A*K_star).sum(0)[:, None]
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
    # Perform Cholesky decomposition on matrix
    self.R = la.cholesky(self.K + self.sigma2*self.K.shape[0])
    # compute the log determinant from Cholesky decomposition
    self.logdetK = 2*np.log(np.diag(self.R)).sum()
    # compute y^\top K^{-1}y from Cholesky factor
    self.Rinvy = la.solve_triangular(self.R, self.y)
    self.yKinvy = (self.Rinvy**2).sum()
    
    # compute the inverse of the upper triangular Cholesky factor
    self.Rinv = la.solve_triangular(self.R, np.eye(self.K.shape[0]))
    self.Kinv = self.Rinv@self.Rinv.T


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

    
def exponentiated_quadratic(x, x_prime, variance=1., lengthscale=1.):
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
    r = np.linalg.norm(x-x_prime, 2)
    return variance*np.exp((-0.5*r*r)/lengthscale**2)        

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
    i = x[0]
    i_prime = x_prime[0]
    return B[i, i_prime]*subkernel(x[1:], x_prime[1:], **kwargs)

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

def add_cov(x, x_prime, kernargs):
    """
    Additive covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param kernargs: List of tuples (kernel, kwargs)
    :type kernargs: list of tuples
    :returns: Summed covariance value
    :rtype: float
    """
    k = 0.
    for kernel, kwargs in kernargs:
        k+=kernel(x, x_prime, **kwargs)
    return k

def prod_kern(x, x_prime, kernargs):
    """
    Product covariance function.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param kernargs: List of tuples (kernel, kwargs)
    :type kernargs: list of tuples
    :returns: Product of covariance values
    :rtype: float
    """
    k = 1.
    for kernel, kwargs in kernargs:
        k*=kernel(x, x_prime, **kwargs)
    return k

def relu_cov(x, x_prime, variance=1., scale=1., w=1., b=5., alpha=0.):
    """
    Covariance function for a ReLU based neural network.
    
    :param x: First data point
    :type x: numpy.ndarray
    :param x_prime: Second data point
    :type x_prime: numpy.ndarray
    :param variance: Overall scale of the covariance
    :type variance: float, optional
    :param scale: Overall scale of the weights on the input.
    :type scale: float, optional
    :param w: Overall scale of the bias on the input
    :type w: float, optional
    :param b: Smoothness of the relu activation
    :type alpha: float, optional
    :returns: Covariance value
    :rtype: float
    """
    def h(costheta, inner, s, a):
        """
        Helper function
        """
        cos2th = costheta*costheta
        return (1-(2*s*s-1)*cos2th)/np.sqrt(a/inner + 1 - s*s*cos2th)*s

    inner = np.dot(x, x_prime)*w + b
    inner_1 = np.dot(x, x)*w + b
    inner_2 = np.dot(x_prime, x_prime)*w + b
    norm_1 = np.sqrt(inner_1 + alpha)
    norm_2 = np.sqrt(inner_2 + alpha)
    norm = norm_1*norm_2
    s = np.sqrt(inner_1)/norm_1
    s_prime = np.sqrt(inner_2)/norm_2
    arg = np.clip(inner/norm, -1, 1) # clip as numerically can be > 1
    arg2 = np.clip(inner/np.sqrt(inner_1*inner_2), -1, 1) # clip as numerically can be > 1
    theta = np.arccos(arg)
    return variance*0.5*((1. - theta/np.pi)*inner + h(arg2, inner_2, s, alpha)/np.pi + h(arg2, inner_1, s_prime, alpha)/np.pi) 


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



