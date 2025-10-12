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
    
    # Activation Functions
    'ReLUActivation',
    'SigmoidActivation',
    'LinearActivation',
    'SoftReLUActivation',
    
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
    
