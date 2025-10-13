"""
Linear Models Module

This module contains linear model implementations including:
- Linear Regression (LM)
- Bayesian Linear Model (BLM) 
- Logistic Regression (LR)
- Basis functions (linear, polynomial, radial, fourier, relu, hyperbolic_tangent)
- Basis class for managing basis functions

TODO: Extract from mlai.py during refactoring
"""

# Placeholder for future linear models functionality
# This will be populated during the refactoring process

__all__ = [
    # Linear Models
    'LM',
    'BLM', 
    'LR',
    
    # Basis Functions
    'linear',
    'polynomial', 
    'radial',
    'fourier',
    'relu',
    'hyperbolic_tangent',
    'Basis',
]

import numpy as np
import scipy.linalg as la

# Import base classes from models module
from .models import Model, ProbModel, MapModel, ProbMapModel

        
    
##########           Weeks 4 and 5           ##########
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
    
    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector: [weights, bias].
        
        This property returns the model parameters in the standard format
        [weights, bias] to be consistent with neural networks and GLMs.
        The internal w_star format [bias, weights] is reordered for external use.
        
        :returns: 1D array of parameters in order [weights, bias]
        :rtype: numpy.ndarray
        
        Examples:
            >>> model = LM(X, y, basis)
            >>> model.fit()
            >>> params = model.parameters
            >>> print(f"Parameters shape: {params.shape}")
            >>> print(f"First {len(params)-1} are weights, last is bias")
        """
        if not hasattr(self, 'w_star'):
            raise ValueError("Model must be fitted or initialised before accessing parameters")
        
        return self.w_star.flatten()
        
    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector: [weights, bias].
        
        This property setter allows updating all model parameters from a
        single 1D array in the standard format [weights, bias]. The parameters
        are reordered to the internal w_star format [bias, weights].
        
        :param value: 1D array of parameters in order [weights, bias]
        :type value: numpy.ndarray
        
        Examples:
            >>> model = LM(X, y, basis)
            >>> model.fit()
            >>> new_params = model.parameters + 0.1  # Add small perturbation
            >>> model.parameters = new_params
        """
        
        value = np.asarray(value)
        if value.ndim != 1:
            raise ValueError("Parameters must be a 1D array")
        
        expected_length = self.w_star.size
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")
        
        n_weights = len(value)
        self.w_star = value.reshape(-1, 1)
        self.update_f()  # Update cached values after setting new parameters
    
    @property
    def gradients(self):
        """
        Get the gradients of the objective function with respect to parameters.
        
        This property computes and returns the gradients of the sum of squares
        objective function with respect to the model parameters.
        
        :returns: 1D array of gradients
        :rtype: numpy.ndarray
        
        Examples:
            >>> model = LM(X, y, basis)
            >>> model.fit()
            >>> grads = model.gradients
            >>> print(f"Gradients shape: {grads.shape}")
        """
        # Compute gradient of sum of squares: 2 * Phi^T * (Phi * w_star - y)
        self.update_f()
        residual = self.f - self.y  # (n_samples, 1)
        grad = 2 * (self.Phi.T @ residual)  # (n_params, 1)
        return grad.flatten()
    

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
        Generate the gradient of the objective function (the negative log-likelihood).

        :returns: Gradient vector of negative log-likelihood
        :rtype: numpy.ndarray
        """
        self.update_g()
        y_bool = self.y.flatten().astype(bool)  # Ensure 1D
        grad = np.zeros((self.Phi.shape[1], 1))
        grad += -(self.Phi[y_bool, :].T @ (1 - self.g[y_bool, :]))
        grad += (self.Phi[~y_bool, :].T @ self.g[~y_bool, :])
        return -grad.flatten()  # Return negative gradient for minimization 
    
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
            self.parameters -= learning_rate * self.gradients  # Minimize negative log-likelihood (go in direction of negative gradient)
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
        
    def log_likelihood(self):
        """
        Compute the log-likelihood.

        :returns: Log-likelihood value
        :rtype: float
        """
        self.update_g()
        y_bool = self.y.flatten().astype(bool)  # Ensure 1D
        return self.log_g[y_bool, :].sum() + self.log_gminus[~y_bool, :].sum()
            
    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.
        
        This property returns the model parameters as a flattened version
        of w_star. The bias is included as the first element due to the
        design of the Phi matrix (with a column of ones).
        
        :returns: 1D array of parameters
        :rtype: numpy.ndarray
        
        Examples:
            >>> model = LR(X, y, basis)
            >>> model.fit()
            >>> params = model.parameters
            >>> print(f"Parameters shape: {params.shape}")
        """
        return self.w_star.flatten()
    
    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.
        
        This property setter allows updating all model parameters from a
        single 1D array. The parameters are reshaped to match w_star.
        
        :param value: 1D array of parameters
        :type value: numpy.ndarray
        
        Examples:
            >>> model = LR(X, y, basis)
            >>> model.fit()
            >>> new_params = model.parameters + 0.1  # Add small perturbation
            >>> model.parameters = new_params
        """
        value = np.asarray(value)
        if value.ndim != 1:
            raise ValueError("Parameters must be a 1D array")
        
        expected_length = self.w_star.size
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")
        
        self.w_star = value.reshape(-1, 1)
        self.update_g()  # Update cached values after setting new parameters
    
    @property
    def gradients(self):
        """
        Get the gradients of the objective function with respect to parameters.
        
        This property computes and returns the gradients of the log-likelihood
        objective function with respect to the model parameters.
        
        :returns: 1D array of gradients
        :rtype: numpy.ndarray
        
        Examples:
            >>> model = LR(X, y, basis)
            >>> model.fit()
            >>> grads = model.gradients
            >>> print(f"Gradients shape: {grads.shape}")
        """
        return self.gradient()
