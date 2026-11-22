"""

Loss functions module

"""

import numpy as np

__all__ = [
    # Loss Functions
    'LossFunction',
    'MeanSquaredError',
    'MeanAbsoluteError',
    'HuberLoss',
    'BinaryCrossEntropyLoss',
    'CrossEntropyLoss',
]



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
    CE = -(1/N) Σ_n Σ_k y_true_{nk} log(y_pred_{nk})

    The gradient is: dCE/dy_pred = -y_true / (N * y_pred)
    where N is the number of samples (predictions.shape[0]).

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
    BCE = -mean[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    where the mean is over all elements of the prediction array.

    The gradient is:
    dBCE/dy_pred = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / predictions.size

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

        :param predictions: Model predictions (probabilities), shape (n_samples, n_outputs)
        :type predictions: numpy.ndarray
        :param targets: True target values (0 or 1), shape (n_samples, n_outputs)
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

        :param predictions: Model predictions (probabilities), shape (n_samples, n_outputs)
        :type predictions: numpy.ndarray
        :param targets: True target values (0 or 1), shape (n_samples, n_outputs)
        :type targets: numpy.ndarray
        :returns: Gradient of binary cross-entropy with respect to predictions
        :rtype: numpy.ndarray
        """
        # Clip predictions to prevent division by zero
        predictions = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        # Divide by predictions.size to match np.mean in forward (all elements)
        return -(targets / predictions - (1 - targets) / (1 - predictions)) / predictions.size


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


class SoftmaxCrossEntropyLoss(LossFunction):
    """
    Softmax cross-entropy loss computed from unnormalised scores (logits).

    This combines a softmax over the last axis with the cross-entropy loss.
    Working with logits is numerically stable and gives the simple gradient

        dL/dz = (softmax(z) - y) / N

    where N is the number of predictions (all leading dimensions, e.g.
    batch_size * seq_len for sequence models).

    Mathematical formulation:
    L = -(1/N) * Σ_n Σ_k y_nk log softmax(z_n)_k

    Examples:
        >>> loss = SoftmaxCrossEntropyLoss()
        >>> logits = np.array([[2.0, 0.5], [0.1, 1.5]])
        >>> targets = np.array([[1, 0], [0, 1]])
        >>> loss.forward(logits, targets)
    """
    def __init__(self):
        super().__init__()
        self.name = "Softmax Cross Entropy"

    @staticmethod
    def _log_softmax(logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

    def forward(self, predictions, targets):
        """
        Compute the softmax cross-entropy loss.

        :param predictions: Logits, shape (..., n_classes)
        :type predictions: numpy.ndarray
        :param targets: One-hot (or probability) targets, same shape as predictions
        :type targets: numpy.ndarray
        :returns: Mean cross-entropy over all predictions
        :rtype: float
        """
        n = predictions.size // predictions.shape[-1]
        return -np.sum(targets * self._log_softmax(predictions)) / n

    def gradient(self, predictions, targets):
        """
        Compute the gradient of the loss with respect to the logits.

        :param predictions: Logits, shape (..., n_classes)
        :type predictions: numpy.ndarray
        :param targets: One-hot (or probability) targets, same shape as predictions
        :type targets: numpy.ndarray
        :returns: Gradient with respect to the logits
        :rtype: numpy.ndarray
        """
        n = predictions.size // predictions.shape[-1]
        return (np.exp(self._log_softmax(predictions)) - targets) / n
