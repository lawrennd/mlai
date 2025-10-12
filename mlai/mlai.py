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

# Import linear models from the linear_models module
from .linear_models import LM
from .models import Model, ProbModel, MapModel, ProbMapModel










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
