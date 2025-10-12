"""
Neural Networks Module

This module contains neural network implementations including:
- SimpleNeuralNetwork
- SimpleDropoutNeuralNetwork  
- NonparametricDropoutNeuralNetwork
- NeuralNetwork
- Activation functions (ReLUActivation, SigmoidActivation, LinearActivation, SoftReLUActivation)
- Perceptron functionality

TODO: Extract from mlai.py during refactoring
"""

import numpy as np
from .models import Model, ProbModel, MapModel, ProbMapModel

__all__ = [
    # Neural Network Classes
    'SimpleNeuralNetwork',
    'SimpleDropoutNeuralNetwork',
    'NonparametricDropoutNeuralNetwork', 
    'NeuralNetwork',
    'LayeredNeuralNetwork',
    
    # Layer Classes
    'Layer',
    'LinearLayer',
    'FullyConnectedLayer',
    'MultiInputLayer',
    'AttentionLayer',
    
    # Activation Functions
    'ReLUActivation',
    'SigmoidActivation',
    'LinearActivation',
    'SoftReLUActivation',
    
    # Loss Functions
    'LossFunction',
    'MeanSquaredError',
    'MeanAbsoluteError',
    'HuberLoss',
    'BinaryCrossEntropyLoss',
    'CrossEntropyLoss',
    
    # Perceptron
    'update_perceptron',
]


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

    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.
        
        Returns all network parameters (w1, b1, w2, b2) as a flattened array
        in the order: [w1, b1, w2, b2].
        
        :returns: 1D array of all trainable parameters
        :rtype: numpy.ndarray
        """
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(), 
            self.w2.flatten(),
            self.b2.flatten()
        ])
    
    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.
        
        Updates all network parameters from a flattened array in the order:
        [w1, b1, w2, b2].
        
        :param value: 1D array of parameters to set
        :type value: numpy.ndarray
        
        :raises ValueError: If the parameter vector has incorrect length
        """
        expected_length = (self.w1.size + self.b1.size + 
                          self.w2.size + self.b2.size)
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")
        
        # Unpack parameters in the same order as the getter
        w1_size = self.w1.size
        b1_size = self.b1.size
        w2_size = self.w2.size
        b2_size = self.b2.size
        
        start = 0
        self.w1 = value[start:start+w1_size].reshape(self.w1.shape)
        start += w1_size
        self.b1 = value[start:start+b1_size].reshape(self.b1.shape)
        start += b1_size
        self.w2 = value[start:start+w2_size].reshape(self.w2.shape)
        start += w2_size
        self.b2 = value[start:start+b2_size].reshape(self.b2.shape)


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
    
    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.
        
        Returns all network parameters (weights and biases for all layers) as a flattened array
        in the order: [weights[0], biases[0], weights[1], biases[1], ...].
        
        :returns: 1D array of all trainable parameters
        :rtype: numpy.ndarray
        """
        params = []
        for i in range(len(self.weights)):
            params.append(self.weights[i].flatten())
            params.append(self.biases[i].flatten())
        return np.concatenate(params)
    
    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.
        
        Updates all network parameters from a flattened array in the order:
        [weights[0], biases[0], weights[1], biases[1], ...].
        
        :param value: 1D array of parameters to set
        :type value: numpy.ndarray
        
        :raises ValueError: If the parameter vector has incorrect length
        """
        # Calculate expected length
        expected_length = 0
        for i in range(len(self.weights)):
            expected_length += self.weights[i].size + self.biases[i].size
        
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")
        
        # Unpack parameters in the same order as the getter
        start = 0
        for i in range(len(self.weights)):
            # Set weights
            w_size = self.weights[i].size
            self.weights[i] = value[start:start+w_size].reshape(self.weights[i].shape)
            start += w_size
            
            # Set biases
            b_size = self.biases[i].size
            self.biases[i] = value[start:start+b_size].reshape(self.biases[i].shape)
            start += b_size



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

class SoftmaxActivation:
    """
    Softmax activation function for attention mechanisms.
    
    This implements the standard softmax function used in attention mechanisms,
    with proper gradient computation for backpropagation.
    """
    
    def forward(self, x, axis=-1):
        """
        Forward pass: softmax activation.
        
        :param x: Input array
        :type x: numpy.ndarray
        :param axis: Axis along which to apply softmax
        :type axis: int
        :returns: Softmax activated array
        :rtype: numpy.ndarray
        """
        # Numerical stability: subtract max before softmax
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def gradient(self, x, grad_output, axis=-1):
        """
        Gradient of softmax activation.
        
        For softmax: ∂s_i/∂x_j = s_i * (δ_ij - s_j)
        where δ_ij is the Kronecker delta.
        
        :param x: Input array
        :type x: numpy.ndarray
        :param grad_output: Gradient from next layer
        :type grad_output: numpy.ndarray
        :param axis: Axis along which softmax was applied
        :type axis: int
        :returns: Gradient array
        :rtype: numpy.ndarray
        """
        # Compute softmax output
        softmax_output = self.forward(x, axis=axis)
        
        # Compute gradient: softmax * (grad - sum(grad * softmax))
        grad_sum = np.sum(grad_output * softmax_output, axis=axis, keepdims=True)
        return softmax_output * (grad_output - grad_sum)

class Layer:
    """
    Abstract base class for neural network layers.
    
    A layer is a building block that contains parameters and
    implements forward and backward passes. Layers can be composed to
    create complex neural network architectures.
    
    Methods
    -------
    forward(input) : numpy.ndarray
        Forward pass through the layer
    backward(grad_output) : numpy.ndarray
        Backward pass through the layer
    parameters : numpy.ndarray
        Property to get/set all trainable parameters as a 1D vector
    
    Examples:
        >>> class LinearLayer(Layer):
        ...     def __init__(self, input_size, output_size):
        ...         self.W = np.random.normal(0, 0.1, (input_size, output_size))
        ...         self.b = np.random.normal(0, 0.1, output_size)
        ...     
        ...     def forward(self, x):
        ...         return x @ self.W + self.b
        ...     
        ...     def backward(self, grad_output):
        ...         return grad_output @ self.W.T
        ...     
        ...     @property
        ...     def parameters(self):
        ...         return np.concatenate([self.W.flatten(), self.b.flatten()])
        ...     
        ...     @parameters.setter
        ...     def parameters(self, value):
        ...         w_size = self.W.size
        ...         self.W = value[:w_size].reshape(self.W.shape)
        ...         self.b = value[w_size:].reshape(self.b.shape)
    """
    
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Forward pass through the layer.
        
        :param x: Input to the layer
        :type x: numpy.ndarray
        :returns: Layer output
        :rtype: numpy.ndarray
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement the forward method")
    
    def backward(self, grad_output):
        """
        Backward pass through the layer.
        
        :param grad_output: Gradient from the next layer
        :type grad_output: numpy.ndarray
        :returns: Gradient with respect to layer input
        :rtype: numpy.ndarray
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement the backward method")
    
    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.
        
        :returns: 1D array of all trainable parameters
        :rtype: numpy.ndarray
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement the parameters property")
    
    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.
        
        :param value: 1D array of parameters to set
        :type value: numpy.ndarray
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement the parameters setter")


class LinearLayer(Layer):
    """
    Linear (fully connected) layer.
    
    This layer implements a linear transformation: y = xW + b
    where W is the weight matrix and b is the bias vector.
    
    :param input_size: Number of input features
    :type input_size: int
    :param output_size: Number of output features
    :type output_size: int
    """
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights using Xavier initialization
        self.W = np.random.normal(0, np.sqrt(2.0 / input_size), (input_size, output_size))
        self.b = np.random.normal(0, np.sqrt(2.0 / input_size), output_size)
    
    def forward(self, x):
        """
        Forward pass: linear transformation.
        
        :param x: Input tensor of shape (batch_size, input_size)
        :type x: numpy.ndarray
        :returns: Output tensor of shape (batch_size, output_size)
        :rtype: numpy.ndarray
        """
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        """
        Backward pass: compute gradients.
        
        :param grad_output: Gradient from next layer, shape (batch_size, output_size)
        :type grad_output: numpy.ndarray
        :returns: Gradient with respect to input, shape (batch_size, input_size)
        :rtype: numpy.ndarray
        """
        return grad_output @ self.W.T
    
    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.
        
        Returns parameters in the order: [W, b]
        
        :returns: 1D array of all trainable parameters
        :rtype: numpy.ndarray
        """
        return np.concatenate([self.W.flatten(), self.b.flatten()])
    
    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.
        
        Updates parameters in the order: [W, b]
        
        :param value: 1D array of parameters to set
        :type value: numpy.ndarray
        
        :raises ValueError: If the parameter vector has incorrect length
        """
        expected_length = self.W.size + self.b.size
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")
        
        w_size = self.W.size
        self.W = value[:w_size].reshape(self.W.shape)
        self.b = value[w_size:].reshape(self.b.shape)


class FullyConnectedLayer(Layer):
    """
    Fully connected layer combining linear transformation and activation.
    
    This layer combines a linear transformation (xW + b) with an activation function,
    providing a complete fully connected layer implementation.
    
    :param input_size: Number of input features
    :type input_size: int
    :param output_size: Number of output features
    :type output_size: int
    :param activation: Activation function to apply
    :type activation: Activation
    """
    
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights using Xavier initialization
        self.W = np.random.normal(0, np.sqrt(2.0 / input_size), (input_size, output_size))
        self.b = np.random.normal(0, np.sqrt(2.0 / input_size), output_size)
        
        # Store input for backward pass
        self.last_input = None
    
    def forward(self, x):
        """
        Forward pass: linear transformation + activation.
        
        :param x: Input array of shape (batch_size, input_size)
        :type x: numpy.ndarray
        :returns: Output array of shape (batch_size, output_size)
        :rtype: numpy.ndarray
        """
        # Store input for backward pass
        self.last_input = x
        
        # Linear transformation
        z = x @ self.W + self.b
        
        # Apply activation
        return self.activation.forward(z)
    
    def backward(self, grad_output):
        """
        Backward pass: compute gradients through activation and linear transformation.
        
        :param grad_output: Gradient from next layer, shape (batch_size, output_size)
        :type grad_output: numpy.ndarray
        :returns: Gradient with respect to input, shape (batch_size, input_size)
        :rtype: numpy.ndarray
        """
        if self.last_input is None:
            raise ValueError("Must call forward before backward")
        
        # Compute linear transformation for gradient computation
        z = self.last_input @ self.W + self.b
        
        # Gradient through activation
        if hasattr(self.activation, 'gradient'):
            activation_grad = self.activation.gradient(z)
            grad_z = grad_output * activation_grad
        else:
            # For activations without gradient method, assume identity
            grad_z = grad_output
        
        # Gradient with respect to input
        grad_input = grad_z @ self.W.T
        
        return grad_input
    
    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.
        
        Returns parameters in the order: [W, b]
        
        :returns: 1D array of all trainable parameters
        :rtype: numpy.ndarray
        """
        return np.concatenate([self.W.flatten(), self.b.flatten()])
    
    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.
        
        Updates parameters in the order: [W, b]
        
        :param value: 1D array of parameters to set
        :type value: numpy.ndarray
        
        :raises ValueError: If the parameter vector has incorrect length
        """
        expected_length = self.W.size + self.b.size
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")
        
        w_size = self.W.size
        self.W = value[:w_size].reshape(self.W.shape)
        self.b = value[w_size:].reshape(self.b.shape)


class LayeredNeuralNetwork(Model):
    """
    Neural network composed of Layer objects.
    
    This class allows for flexible neural network architectures by composing
    different types of layers. Each layer handles its own parameters and
    forward/backward passes.
    
    :param layers: List of Layer objects to compose
    :type layers: list[Layer]
    """
    
    def __init__(self, layers):
        super().__init__()
        if not layers:
            raise ValueError("At least one layer must be provided")
        self.layers = layers
        
        # Store activations for backward pass
        self.activations = []
    
    def forward(self, x):
        """
        Forward pass through all layers.
        
        :param x: Input tensor
        :type x: numpy.ndarray
        :returns: Output tensor
        :rtype: numpy.ndarray
        """
        self.activations = [x]  # Store input
        current = x
        
        for layer in self.layers:
            current = layer.forward(current)
            self.activations.append(current)
        
        return current
    
    def backward(self, grad_output):
        """
        Backward pass through all layers.
        
        :param grad_output: Gradient from the next layer
        :type grad_output: numpy.ndarray
        :returns: Dictionary containing gradients for all layers
        :rtype: dict
        """
        gradients = {}
        current_grad = grad_output
        
        # Backpropagate through layers in reverse order
        for i, layer in enumerate(reversed(self.layers)):
            layer_idx = len(self.layers) - 1 - i
            current_grad = layer.backward(current_grad)
            gradients[f'layer_{layer_idx}'] = current_grad
        
        return gradients
    
    def predict(self, x):
        """
        Predict method for compatibility with Model interface.
        
        :param x: Input tensor
        :type x: numpy.ndarray
        :returns: Network output
        :rtype: numpy.ndarray
        """
        return self.forward(x)
    
    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.
        
        Returns parameters from all layers concatenated in order.
        
        :returns: 1D array of all trainable parameters
        :rtype: numpy.ndarray
        """
        params = []
        for layer in self.layers:
            params.append(layer.parameters)
        return np.concatenate(params)
    
    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.
        
        Updates parameters for all layers in order.
        
        :param value: 1D array of parameters to set
        :type value: numpy.ndarray
        
        :raises ValueError: If the parameter vector has incorrect length
        """
        expected_length = sum(len(layer.parameters) for layer in self.layers)
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")
        
        start = 0
        for layer in self.layers:
            layer_size = len(layer.parameters)
            layer.parameters = value[start:start + layer_size]
            start += layer_size


class MultiInputLayer(Layer):
    """
    Base class for layers that can process multiple inputs.
    
    This class provides a framework for layers that need to handle
    multiple input streams, such as attention mechanisms, cross-correlation
    layers, or other interaction-based computations.
    
    Methods
    -------
    forward(*inputs) : numpy.ndarray
        Forward pass through the layer with multiple inputs
    backward(grad_output) : tuple
        Backward pass returning gradients for each input
    parameters : numpy.ndarray
        Property to get/set all trainable parameters as a 1D vector
    
    Examples:
        >>> class AttentionLayer(MultiInputLayer):
        ...     def forward(self, x, query_input=None, key_value_input=None):
        ...         # Handle self-attention or cross-attention
        ...         pass
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, *inputs):
        """
        Forward pass with multiple inputs.
        
        :param inputs: Variable number of input tensors
        :type inputs: tuple of numpy.ndarray
        :returns: Layer output
        :rtype: numpy.ndarray
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement the forward method")
    
    def backward(self, grad_output):
        """
        Backward pass returning gradients for each input.
        
        :param grad_output: Gradient from the next layer
        :type grad_output: numpy.ndarray
        :returns: Tuple of gradients for each input
        :rtype: tuple of numpy.ndarray
        
        :raises NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement the backward method")


class AttentionLayer(MultiInputLayer):
    """
    General attention layer supporting both self and cross attention.
    
    This layer implements the scaled dot-product attention mechanism
    and can handle both self-attention (single input) and cross-attention
    (multiple inputs) scenarios.
    
    :param d_model: Model dimension
    :type d_model: int
    :param n_heads: Number of attention heads
    :type n_heads: int
    :param dropout: Dropout rate for regularization
    :type dropout: float, optional
    :param activation: Activation function for attention weights
    :type activation: Activation, optional
    """
    
    def __init__(self, d_model, n_heads, dropout=0.0, activation=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Set up activation function (default to softmax for attention)
        if activation is None:
            self.activation = SoftmaxActivation()
        else:
            self.activation = activation
        
        # Initialize weight matrices for Q, K, V transformations
        # Using Xavier initialization for stability
        self.W_q = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        self.W_k = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        self.W_v = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        self.W_o = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        
        # Scale factor for attention
        self.scale = 1.0 / np.sqrt(d_model)
        
        # Store for backward pass
        self.last_inputs = None
        self.last_output = None
        self.last_attention_weights = None
    
    def forward(self, x, query_input=None, key_value_input=None):
        """
        Forward pass of the attention mechanism.
        
        Supports three modes:
        1. Self-attention: forward(x) - x becomes Q, K, V
        2. Cross-attention: forward(x, query_input=q, key_value_input=kv) - Q=q, K=V=kv
        3. Mixed: forward(x, key_value_input=kv) - Q=x, K=V=kv
        
        :param x: Primary input tensor
        :type x: numpy.ndarray
        :param query_input: Optional separate query input (for cross-attention)
        :type query_input: numpy.ndarray, optional
        :param key_value_input: Optional separate key/value input (for cross-attention)
        :type key_value_input: numpy.ndarray, optional
        :returns: Attention output
        :rtype: numpy.ndarray
        """
        # Store inputs for backward pass
        self.last_inputs = (x, query_input, key_value_input)
        
        # Determine Q, K, V based on inputs
        if query_input is not None:
            # Cross-attention: Q from query_input, K,V from key_value_input
            if key_value_input is None:
                raise ValueError("key_value_input must be provided when query_input is given")
            Q = query_input @ self.W_q
            K = key_value_input @ self.W_k
            V = key_value_input @ self.W_v
        else:
            # Self-attention or mixed attention
            if key_value_input is not None:
                # Mixed attention: Q from x, K,V from key_value_input
                Q = x @ self.W_q
                K = key_value_input @ self.W_k
                V = key_value_input @ self.W_v
            else:
                # Self-attention: Q, K, V from x
                Q = x @ self.W_q
                K = x @ self.W_k
                V = x @ self.W_v
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        attention_scores = (Q @ K.transpose(0, 2, 1)) * self.scale
        
        # Apply activation function to get attention weights
        if hasattr(self.activation, 'forward') and 'axis' in self.activation.forward.__code__.co_varnames:
            attention_weights = self.activation.forward(attention_scores, axis=-1)
        else:
            # For activations that don't support axis parameter, apply element-wise
            attention_weights = self.activation.forward(attention_scores)
            # Apply softmax normalisation if requested (default for attention mechanisms)
            attention_weights = self._softmax(attention_weights, axis=-1)
        
        # Apply dropout during training
        if self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout)
        
        # Apply attention to values
        output = attention_weights @ V
        
        # Output projection
        output = output @ self.W_o
        
        # Store for backward pass
        self.last_output = output
        self.last_attention_weights = attention_weights
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass implementing the attention gradient computation.
        
        :param grad_output: Gradient from the next layer
        :type grad_output: numpy.ndarray
        :returns: Tuple of gradients for each input
        :rtype: tuple of numpy.ndarray
        """
        if self.last_inputs is None:
            raise ValueError("Must call forward before backward")
        
        x, query_input, key_value_input = self.last_inputs
        
        # Recompute Q, K, V for gradient computation
        if query_input is not None:
            # Cross-attention case
            if key_value_input is None:
                raise ValueError("key_value_input must be provided when query_input is given")
            Q = query_input @ self.W_q
            K = key_value_input @ self.W_k
            V = key_value_input @ self.W_v
        else:
            # Self-attention or mixed attention
            if key_value_input is not None:
                # Mixed attention: Q from x, K,V from key_value_input
                Q = x @ self.W_q
                K = key_value_input @ self.W_k
                V = key_value_input @ self.W_v
            else:
                # Self-attention: Q, K, V from x
                Q = x @ self.W_q
                K = x @ self.W_k
                V = x @ self.W_v
        
        # Gradient through output projection
        grad_attention_output = grad_output @ self.W_o.T
        
        # Gradients for attention weights and values
        grad_attention_weights = grad_attention_output @ V.transpose(0, 2, 1)
        grad_V = self.last_attention_weights.transpose(0, 2, 1) @ grad_attention_output
        
        # Gradient through activation function
        attention_scores = (Q @ K.transpose(0, 2, 1)) * self.scale
        
        # Handle different activation function interfaces
        if hasattr(self.activation, 'gradient') and 'axis' in self.activation.gradient.__code__.co_varnames:
            grad_attention_scores = self.activation.gradient(attention_scores, grad_attention_weights, axis=-1)
        else:
            # Use standard softmax gradient formula
            grad_sum = np.sum(grad_attention_weights * self.last_attention_weights, axis=-1, keepdims=True)
            grad_attention_scores = self.last_attention_weights * (grad_attention_weights - grad_sum)
        
        # Gradients for Q and K
        grad_Q = grad_attention_scores @ K
        grad_K = grad_attention_scores.transpose(0, 2, 1) @ Q
        
        # Scale gradients
        grad_Q *= self.scale
        grad_K *= self.scale
        
        # Compute input gradients
        if query_input is not None:
            # Cross-attention: separate gradients for query and key_value inputs
            grad_query = grad_Q @ self.W_q.T
            grad_key_value = (grad_K @ self.W_k.T + grad_V @ self.W_v.T)
            return grad_query, grad_key_value
        else:
            if key_value_input is not None:
                # Mixed attention: separate gradients for x and key_value_input
                grad_x = grad_Q @ self.W_q.T  # Only Q comes from x
                grad_key_value = (grad_K @ self.W_k.T + grad_V @ self.W_v.T)  # K,V from key_value_input
                return grad_x, grad_key_value
            else:
                # Self-attention: single gradient for input (three-path chain rule)
                grad_input = (grad_Q @ self.W_q.T + grad_K @ self.W_k.T + grad_V @ self.W_v.T)
                return (grad_input,)
    
    def _softmax(self, x, axis=-1):
        """Numerically stable softmax implementation."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.
        
        Returns parameters in the order: [W_q, W_k, W_v, W_o]
        
        :returns: 1D array of all trainable parameters
        :rtype: numpy.ndarray
        """
        return np.concatenate([
            self.W_q.flatten(),
            self.W_k.flatten(),
            self.W_v.flatten(),
            self.W_o.flatten()
        ])
    
    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.
        
        Updates parameters in the order: [W_q, W_k, W_v, W_o]
        
        :param value: 1D array of parameters to set
        :type value: numpy.ndarray
        
        :raises ValueError: If the parameter vector has incorrect length
        """
        expected_length = self.W_q.size + self.W_k.size + self.W_v.size + self.W_o.size
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")
        
        # Unpack parameters in the same order as the getter
        start = 0
        
        # W_q
        w_q_size = self.W_q.size
        self.W_q = value[start:start+w_q_size].reshape(self.W_q.shape)
        start += w_q_size
        
        # W_k
        w_k_size = self.W_k.size
        self.W_k = value[start:start+w_k_size].reshape(self.W_k.shape)
        start += w_k_size
        
        # W_v
        w_v_size = self.W_v.size
        self.W_v = value[start:start+w_v_size].reshape(self.W_v.shape)
        start += w_v_size
        
        # W_o
        w_o_size = self.W_o.size
        self.W_o = value[start:start+w_o_size].reshape(self.W_o.shape)





