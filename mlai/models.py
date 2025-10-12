"""
Base Models Module

This module contains base model classes including:
- ProbModel (abstract base class)
- Model (base class for all models)
- Loss (base class for loss functions)

TODO: Extract from mlai.py during refactoring
"""

# Placeholder for future base models functionality
# This will be populated during the refactoring process

__all__ = [
    # Base Model Classes
    'ProbModel',
    'Model',
    'MapModel',
    'ProbMapModel',
]

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

