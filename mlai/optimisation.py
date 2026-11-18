"""
Optimisation Module

This module contains optimisation utilities including:
- Gradient-based optimisation algorithms (SGD, Adam, SCG, L-BFGS)
- Training utilities

This provides a unified interface for training models with different optimisers.
"""

import numpy as np

__all__ = [
    'Optimiser',
    'SGD',
    'Adam',
    'SCG',
    'LBFGS',
    'train_model',
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


def _objective_at(model, x):
    """Evaluate ``model.objective()`` at unconstrained parameters ``x``."""
    model.parameters = x
    return float(model.objective())


def _gradient_at(model, x):
    """Evaluate ``model.gradients`` at unconstrained parameters ``x``."""
    model.parameters = x
    return np.asarray(model.gradients, dtype=float).ravel().copy()


class SCG(Optimiser):
    """
    Scaled Conjugate Gradients (Møller / Netlab).

    Full-batch optimiser for smooth objectives such as GP marginal likelihood
    or sparse variational bounds. Prefer this (or :class:`LBFGS`) over SGD/Adam
    for exact and sparse GP teaching demos.

    The algorithm follows Netlab ``scg`` / paramz ``SCG``: directional
    finite-difference curvature, adaptive scale ``beta``, Polak–Ribière
    updates, and a restart every ``n_params`` successful steps.

    Examples
    --------
    >>> opt = SCG()
    >>> flog = opt.optimize(gp, max_iters=50)
    """

    def __init__(self, xtol=1e-6, ftol=1e-6, gtol=1e-5, sigma0=1e-7):
        """
        :param xtol: Absolute step-size convergence tolerance
        :param ftol: Relative objective reduction tolerance
        :param gtol: Gradient-norm convergence tolerance
        :param sigma0: Scale for directional finite-difference curvature probe
        """
        super().__init__()
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.sigma0 = sigma0
        self.status = None
        self.function_evals = 0
        self._state = None

    def _reset_state(self, model):
        x = np.asarray(model.parameters, dtype=float).ravel().copy()
        fold = _objective_at(model, x)
        gradnew = _gradient_at(model, x)
        self.function_evals = 2
        self._state = {
            'x': x,
            'fold': fold,
            'fnow': fold,
            'gradnew': gradnew,
            'gradold': gradnew.copy(),
            'd': -gradnew.copy(),
            'success': True,
            'nsuccess': 0,
            'beta': 1.0,
            'betamin': 1e-15,
            'betamax': 1e15,
            'flog': [fold],
        }

    def step(self, model):
        """
        Perform one SCG iteration.

        :param model: Model with ``parameters``, ``gradients``, ``objective``
        :returns: Current objective value
        :rtype: float
        """
        if self._state is None:
            self._reset_state(model)
        st = self._state
        x = st['x']
        d = st['d']
        gradnew = st['gradnew']
        beta = st['beta']
        success = st['success']

        if success:
            mu = float(np.dot(d, gradnew))
            if mu >= 0.0:
                d = -gradnew
                mu = float(np.dot(d, gradnew))
                st['d'] = d
            kappa = float(np.dot(d, d))
            if kappa < np.finfo(float).eps:
                self.status = 'converged - zero search direction'
                model.parameters = x
                return st['fnow']
            sigma = self.sigma0 / np.sqrt(kappa)
            gplus = _gradient_at(model, x + sigma * d)
            self.function_evals += 1
            theta = float(np.dot(d, (gplus - gradnew))) / sigma
            st['mu'] = mu
            st['kappa'] = kappa
            st['theta'] = theta
        else:
            mu = st['mu']
            kappa = st['kappa']
            theta = st['theta']

        delta = theta + beta * kappa
        if delta <= 0.0:
            delta = beta * kappa
            beta = beta - theta / kappa
        alpha = -mu / delta

        xnew = x + alpha * d
        fnew = _objective_at(model, xnew)
        self.function_evals += 1
        Delta = 2.0 * (fnew - st['fold']) / (alpha * mu) if alpha * mu != 0 else 0.0

        if Delta >= 0.0:
            success = True
            st['nsuccess'] += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = st['fold']
            model.parameters = x

        st['flog'].append(fnow)
        st['success'] = success
        st['x'] = x
        st['fnow'] = fnow
        st['beta'] = beta

        if success:
            if abs(fnew - st['fold']) < self.ftol:
                self.status = 'converged - relative reduction in objective'
                model.parameters = x
                return fnow
            if np.max(np.abs(alpha * d)) < self.xtol:
                self.status = 'converged - relative stepsize'
                model.parameters = x
                return fnow
            st['gradold'] = gradnew
            gradnew = _gradient_at(model, x)
            self.function_evals += 1
            st['gradnew'] = gradnew
            st['fold'] = fnew
            if float(np.dot(gradnew, gradnew)) <= self.gtol:
                self.status = 'converged - relative reduction in gradient'
                return fnow

        if Delta < 0.25:
            beta = min(4.0 * beta, st['betamax'])
        if Delta > 0.75:
            beta = max(0.25 * beta, st['betamin'])
        st['beta'] = beta

        if st['nsuccess'] == x.size:
            st['d'] = -st['gradnew']
            st['beta'] = 1.0
            st['nsuccess'] = 0
        elif success:
            gamma = float(np.dot(st['gradold'] - st['gradnew'], st['gradnew'])) / mu
            st['d'] = gamma * d - st['gradnew']

        model.parameters = st['x']
        return st['fnow']

    def optimize(self, model, max_iters=100, verbose=False):
        """
        Run SCG until convergence or ``max_iters``.

        :param model: Model with ``parameters``, ``gradients``, ``objective``
        :param max_iters: Maximum SCG iterations
        :param verbose: Print cycle / objective / scale
        :returns: List of objective values (including the start)
        :rtype: list
        """
        self.status = None
        self._reset_state(model)
        for iteration in range(max_iters):
            fnow = self.step(model)
            if verbose:
                print(
                    f"Cycle {iteration:4d}  Error {fnow:11.6f}  "
                    f"Scale {self._state['beta']:.3e}"
                )
            if self.status is not None and self.status.startswith('converged'):
                break
        else:
            self.status = 'maxiter exceeded'
        model.parameters = self._state['x']
        return list(self._state['flog'])


class LBFGS(Optimiser):
    """
    L-BFGS-B via SciPy on the shared ``parameters`` / ``gradients`` surface.

    Full-batch quasi-Newton method for smooth GP-style objectives. SciPy owns
    the limited-memory BFGS updates; this class is the teaching-facing wrapper.

    Examples
    --------
    >>> opt = LBFGS()
    >>> flog = opt.optimize(gp, max_iters=100)
    """

    def __init__(self, ftol=1e-9, gtol=1e-5):
        """
        :param ftol: SciPy relative objective tolerance
        :param gtol: SciPy gradient tolerance
        """
        super().__init__()
        self.ftol = ftol
        self.gtol = gtol
        self.status = None
        self.result_ = None
        self._flog = None

    def step(self, model):
        """
        Run a short L-BFGS-B pulse (``maxiter=1`` budget) for manual loops.

        Prefer :meth:`optimize` for normal teaching use.
        """
        flog = self.optimize(model, max_iters=1, verbose=False)
        return flog[-1]

    def optimize(self, model, max_iters=100, verbose=False):
        """
        Minimise ``model.objective()`` with L-BFGS-B.

        :param model: Model with ``parameters``, ``gradients``, ``objective``
        :param max_iters: Maximum iterations passed to SciPy
        :param verbose: Print SciPy iteration messages
        :returns: Objective trace (start plus accepted values when available)
        :rtype: list
        """
        from scipy.optimize import minimize

        x0 = np.asarray(model.parameters, dtype=float).ravel().copy()
        flog = [float(model.objective())]

        def fun(x):
            val = _objective_at(model, x)
            flog.append(val)
            return val

        def jac(x):
            return _gradient_at(model, x)

        self.result_ = minimize(
            fun,
            x0,
            method='L-BFGS-B',
            jac=jac,
            options={
                'maxiter': int(max_iters),
                'ftol': self.ftol,
                'gtol': self.gtol,
            },
        )
        model.parameters = self.result_.x
        self.status = self.result_.message
        self._flog = flog
        if verbose:
            print(f"L-BFGS-B: {self.status}  f={self.result_.fun:.6g}")
        return list(flog)


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

        # Handle models that return (predictions, uncertainty) tuples
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        # Compute loss
        loss = loss_fn.forward(y_pred, y)
        losses.append(loss)

        # Compute loss gradient
        loss_gradient = loss_fn.gradient(y_pred, y)

        # Set output gradient for the model (if it has this method)
        if hasattr(model, 'set_output_gradient'):
            model.set_output_gradient(loss_gradient)

        # optimisation step
        optimiser.step(model)

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch}: Loss = {loss:.6f}")

    return losses
