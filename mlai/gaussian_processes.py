"""
Gaussian Processes Module

This module contains Gaussian Process implementations including:
- Gaussian Process (GP) class
- Kernel functions (eq_cov, ou_cov, matern32_cov, matern52_cov, periodic_cov, 
  linear_cov, polynomial_cov, relu_cov, bias_cov, add_cov, prod_kern)
- Kernel class for managing kernel functions

TODO: Extract from mlai.py during refactoring
"""

import numpy as np
import scipy.linalg as la
from .models import ProbMapModel


__all__ = [
    # Gaussian Process Classes
    'GP',
    'Kernel',
    
    # Kernel Functions
    'eq_cov',
    'ou_cov', 
    'matern32_cov',
    'matern52_cov',
    'periodic_cov',
    'linear_cov',
    'polynomial_cov',
    'relu_cov',
    'bias_cov',
    'add_cov',
    'prod_cov',
    'mlp_cov',
    'sinc_cov',
    'brownian_cov',
    'ratquad_cov',
    'basis_cov',

    # Multi output
    'icm_cov',
    'lmc_cov',
    'slfm_cov',
    
    # Utility functions
    'posterior_f',
    'update_inverse',
    'contour_data',
]


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

