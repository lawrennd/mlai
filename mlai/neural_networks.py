"""
Neural Networks Module

This module contains neural network implementations including:
- SimpleNeuralNetwork
- NeuralNetwork
- Activation functions (ReLUActivation, SigmoidActivation, LinearActivation, SoftReLUActivation)
- Perceptron functionality

"""

import numpy as np
from .models import Model

__all__ = [
    # Neural Network Classes
    'SimpleNeuralNetwork',
    'NeuralNetwork',
    'LayeredNeuralNetwork',

    # Layer Classes
    'Layer',
    'LinearLayer',
    'FullyConnectedLayer',
    'MultiInputLayer',
    'AttentionLayer',
    'MultiHeadAttentionLayer',
    'PositionalEncodingLayer',

    # Convolutional Layers
    'ConvolutionalLayer',
    'MaxPoolingLayer',
    'FlattenLayer',

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
        output = np.sum(self.w2*phi) + self.b2

        # Cache values needed for gradient computation
        self._last_input = x
        self._last_vxmb = vxmb
        self._last_phi = phi
        self._last_output = output

        return output

    def set_output_gradient(self, output_gradient):
        """
        Set the gradient of the loss with respect to the network output.

        This method should be called after computing the loss to enable
        gradient computation via the gradients property.

        :param output_gradient: Gradient of loss with respect to network output
        :type output_gradient: float
        """
        self._last_output_gradient = output_gradient

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

    @property
    def gradients(self):
        """
        Get gradients of the loss with respect to all parameters.

        This property computes the gradients of the loss function with respect
        to all trainable parameters (w1, b1, w2, b2) using backpropagation.

        Note: This requires that the network has been used in a forward pass
        and that the loss has been computed. The gradients are computed
        using the chain rule through the ReLU activation function.

        :returns: 1D array of gradients in the same order as parameters
        :rtype: numpy.ndarray

        :raises ValueError: If the network hasn't been used in a forward pass
        """
        if not hasattr(self, '_last_input') or not hasattr(self, '_last_output_gradient'):
            raise ValueError("Network must be used in a forward pass with loss computation before accessing gradients")

        x = self._last_input
        output_grad = self._last_output_gradient

        # Use cached values from last forward pass
        vxmb = self._last_vxmb
        phi = self._last_phi

        # Gradients for output layer (w2, b2)
        # dL/dw2 = dL/dy * dy/dw2 = output_grad * phi
        # dL/db2 = dL/dy * dy/db2 = output_grad * 1
        grad_w2 = output_grad * phi
        grad_b2 = output_grad

        # Gradients for hidden layer (w1, b1) using chain rule
        # dL/dw1 = dL/dy * dy/dphi * dphi/dvxmb * dvxmb/dw1
        # dL/db1 = dL/dy * dy/dphi * dphi/dvxmb * dvxmb/db1

        # dy/dphi = w2 (from output layer)
        # dphi/dvxmb = (vxmb > 0) (ReLU derivative)
        # dvxmb/dw1 = x, dvxmb/db1 = 1

        # Chain: output_grad * w2 * (vxmb > 0) * x
        relu_derivative = (vxmb > 0).astype(float)
        hidden_grad = output_grad * self.w2 * relu_derivative

        grad_w1 = hidden_grad * x
        grad_b1 = hidden_grad

        # Return gradients in the same order as parameters: [w1, b1, w2, b2]
        return np.concatenate([
            grad_w1.flatten(),
            grad_b1.flatten(),
            grad_w2.flatten(),
            np.array([grad_b2])  # Convert scalar to array
        ])


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

        Numerically stable for large positive or negative inputs.

        :param x: Input array
        :type x: numpy.ndarray
        :returns: Sigmoid activated array
        :rtype: numpy.ndarray
        """
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1. / (1. + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1. + exp_x)
        return out

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

        Uses logaddexp for numerical stability at large positive inputs.

        :param x: Input array
        :type x: numpy.ndarray
        :returns: Soft ReLU activated array
        :rtype: numpy.ndarray
        """
        return np.logaddexp(0., x)

    def gradient(self, x):
        """
        Gradient of soft ReLU activation.

        :param x: Input array
        :type x: numpy.ndarray
        :returns: Gradient array (sigmoid of input)
        :rtype: numpy.ndarray
        """
        return SigmoidActivation().forward(x)

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

        The lecture notes use column-vector layout (z = W φ) and Kronecker
        products. This implementation uses the transposed row-vector layout
        (z = φ @ W + b). Under that convention the Kronecker term
        φ^T ⊗ I collapses to an outer product of the layer input and the
        backpropagated delta, which is what we compute below.

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

            # Layer input: a[0] is the network input; a[i] is the previous
            # layer activation for i > 0.
            input_to_layer = self.a[i]

            # Weight gradient via outer product (row-vector layout):
            # for each batch item, outer(input, delta * φ'(z)).
            # Weight matrix W_i has shape (input_size, output_size).
            weight_grad = np.zeros((self.weights[i].shape[0], self.weights[i].shape[1]))
            for b in range(delta.shape[0]):
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
        Compute gradient for a specific layer using the chain rule.

        Equivalent to the lecture Kronecker formula under the row-vector
        layout used by this class (z = φ @ W + b): the Kronecker term
        collapses to an outer product of the layer input and the
        backpropagated delta.

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

        # Layer input (a[0] is network input; a[layer_idx] is previous activation)
        input_to_layer = self.a[layer_idx]

        # Outer product of layer input and delta (row-vector layout)
        final_gradient = np.zeros((self.weights[layer_idx].shape[0], self.weights[layer_idx].shape[1]))
        for b in range(gradient.shape[0]):
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

    def set_output_gradient(self, output_gradient):
        """
        Set the gradient of the loss with respect to the network output.

        This method should be called after computing the loss to enable
        gradient computation via the gradients property.

        :param output_gradient: Gradient of loss with respect to network output
        :type output_gradient: numpy.ndarray
        """
        self._last_output_gradient = output_gradient

    @property
    def gradients(self):
        """
        Get gradients of the loss with respect to all parameters.

        This property computes the gradients of the loss function with respect
        to all trainable parameters using backpropagation through the network.

        Note: This requires that the network has been used in a forward pass
        and that the loss gradient has been set via set_output_gradient().

        :returns: 1D array of gradients in the same order as parameters
        :rtype: numpy.ndarray

        :raises ValueError: If the network hasn't been used in a forward pass
        """
        if not hasattr(self, '_last_output_gradient'):
            raise ValueError("Network must be used in a forward pass with loss computation before accessing gradients")

        # Use the existing backward method to compute gradients
        gradient_dict = self.backward(self._last_output_gradient)

        # Flatten gradients in the same order as parameters: [weights[0], biases[0], weights[1], biases[1], ...]
        gradients = []
        for i in range(len(self.weights)):
            gradients.append(gradient_dict['weight_gradients'][i].flatten())
            gradients.append(gradient_dict['bias_gradients'][i].flatten())

        return np.concatenate(gradients)



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

    @property
    def gradients(self):
        """
        Gradient of the loss with respect to the layer's parameters.

        Layers with trainable parameters compute these during ``backward()``
        and return them here as a 1D vector in the same order as
        ``parameters``. Layers without parameters return an empty array.

        :returns: 1D array of parameter gradients from the last backward pass
        :rtype: numpy.ndarray
        :raises ValueError: If the layer has parameters but ``backward()`` has not been called
        """
        grads = getattr(self, '_parameter_gradients', None)
        if grads is not None:
            return grads
        if np.asarray(self.parameters).size == 0:
            return np.array([])
        raise ValueError("Call backward() before accessing gradients")


def _unwrap_gradient(grad_output):
    """Return the array inside a single-element gradient tuple."""
    if isinstance(grad_output, tuple):
        return grad_output[0]
    return grad_output


def _linear_parameter_gradients(layer_input, grad_z):
    """
    Weight and bias gradients for z = xW + b.

    Works for inputs of shape (batch_size, input_size) and for sequence
    inputs of shape (batch_size, seq_len, input_size): every leading
    dimension is treated as a batch dimension and summed over.
    """
    x2 = layer_input.reshape(-1, layer_input.shape[-1])
    g2 = grad_z.reshape(-1, grad_z.shape[-1])
    return np.concatenate([(x2.T @ g2).flatten(), g2.sum(axis=0)])


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

        # He initialisation: variance 2 / fan_in
        self.W = np.random.normal(0, np.sqrt(2.0 / input_size), (input_size, output_size))
        self.b = np.random.normal(0, np.sqrt(2.0 / input_size), output_size)

    def forward(self, x):
        """
        Forward pass: linear transformation.

        :param x: Input tensor of shape (batch_size, input_size) or
            (batch_size, seq_len, input_size)
        :type x: numpy.ndarray
        :returns: Output tensor of shape (..., output_size)
        :rtype: numpy.ndarray
        """
        self.last_input = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        """
        Backward pass: compute gradients.

        Stores dL/dW = x^T (dL/dy) and dL/db = sum(dL/dy) so they are
        available from ``gradients``, and returns dL/dx = (dL/dy) W^T.

        :param grad_output: Gradient from next layer, shape (..., output_size)
        :type grad_output: numpy.ndarray
        :returns: Gradient with respect to input, shape (..., input_size)
        :rtype: numpy.ndarray
        """
        grad_output = _unwrap_gradient(grad_output)
        if getattr(self, 'last_input', None) is not None:
            self._parameter_gradients = _linear_parameter_gradients(self.last_input, grad_output)
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

        # He initialisation: variance 2 / fan_in
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
        grad_output = _unwrap_gradient(grad_output)

        # Compute linear transformation for gradient computation
        z = self.last_input @ self.W + self.b

        # Gradient through activation
        if hasattr(self.activation, 'gradient'):
            activation_grad = self.activation.gradient(z)
            grad_z = grad_output * activation_grad
        else:
            # For activations without gradient method, assume identity
            grad_z = grad_output

        # Parameter gradients: dL/dW = x^T dL/dz, dL/db = sum(dL/dz)
        self._parameter_gradients = _linear_parameter_gradients(self.last_input, grad_z)

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


class ResidualLayer(Layer):
    """
    Residual (skip) connection around a sequence of layers.

    Computes y = x + f(x), where f is the composition of the wrapped layers.
    Because x reaches the output through two paths, the input gradient is
    the sum of the two:

        dL/dx = dL/dy + (dL/dy backpropagated through f)

    Residual connections are used around the attention and feed-forward
    sub-layers of a transformer block.

    :param layers: A layer, or a list of layers applied in order, forming f
    :type layers: Layer or list[Layer]
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = list(layers) if isinstance(layers, (list, tuple)) else [layers]
        if not self.layers:
            raise ValueError("At least one layer must be provided")

    def forward(self, x):
        """
        Forward pass: y = x + f(x).

        :param x: Input tensor
        :type x: numpy.ndarray
        :returns: Output tensor with the same shape as x
        :rtype: numpy.ndarray
        """
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        if h.shape != x.shape:
            raise ValueError(f"Residual branch output shape {h.shape} does not match input shape {x.shape}")
        return x + h

    def backward(self, grad_output):
        """
        Backward pass: sum of the skip path and the branch path.

        :param grad_output: Gradient from the next layer
        :type grad_output: numpy.ndarray
        :returns: Gradient with respect to input
        :rtype: tuple
        """
        grad_output = _unwrap_gradient(grad_output)
        grad = grad_output
        for layer in reversed(self.layers):
            grad = _unwrap_gradient(layer.backward(grad))
        return (grad_output + grad,)

    @property
    def parameters(self):
        """
        All trainable parameters of the wrapped layers as a 1D vector.

        :returns: 1D array of parameters
        :rtype: numpy.ndarray
        """
        return np.concatenate([np.asarray(layer.parameters).flatten() for layer in self.layers])

    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters of the wrapped layers from a 1D vector.

        :param value: 1D array of parameters
        :type value: numpy.ndarray
        :raises ValueError: If the parameter vector has incorrect length
        """
        expected_length = sum(np.asarray(layer.parameters).size for layer in self.layers)
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")
        start = 0
        for layer in self.layers:
            size = np.asarray(layer.parameters).size
            if size:
                layer.parameters = value[start:start + size]
            start += size

    @property
    def gradients(self):
        """
        Parameter gradients of the wrapped layers from the last backward pass.

        :returns: 1D array of gradients in the same order as ``parameters``
        :rtype: numpy.ndarray
        """
        return np.concatenate([np.asarray(layer.gradients).flatten() for layer in self.layers])


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
            current_grad = _unwrap_gradient(layer.backward(current_grad))
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

    def set_output_gradient(self, output_gradient):
        """
        Set the gradient of the loss with respect to the network output.

        This method should be called after computing the loss to enable
        gradient computation via the gradients property.

        :param output_gradient: Gradient of loss with respect to network output
        :type output_gradient: numpy.ndarray
        """
        self._last_output_gradient = output_gradient

    @property
    def gradients(self):
        """
        Get gradients of the loss with respect to all parameters.

        This property runs backpropagation through all layers. Each layer
        computes the gradient of the loss with respect to its own parameters
        during ``backward()``; these are collected in the same order as
        ``parameters``.

        Note: This requires that the network has been used in a forward pass
        and that the loss gradient has been set via set_output_gradient().

        :returns: 1D array of gradients in the same order as parameters
        :rtype: numpy.ndarray

        :raises ValueError: If the network hasn't been used in a forward pass
        """
        if not hasattr(self, '_last_output_gradient'):
            raise ValueError("Network must be used in a forward pass with loss computation before accessing gradients")

        self.backward(self._last_output_gradient)
        return np.concatenate([np.asarray(layer.gradients).flatten() for layer in self.layers])


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
    Single-head attention layer supporting both self and cross attention.

    This layer implements the scaled dot-product attention mechanism

        A = softmax(Q K^T / sqrt(d_k)),   output = A V W_o

    with Q = X_q W_q, K = X_kv W_k and V = X_kv W_v. It can handle
    self-attention (single input) and cross-attention (separate query and
    key/value inputs).

    For multi-head attention, use MultiHeadAttentionLayer, which composes
    several AttentionLayer heads with ``output_projection=False`` and a
    shared output projection.

    :param d_model: Model (input and output) dimension
    :type d_model: int
    :param dropout: Dropout rate applied to the attention weights
    :type dropout: float, optional
    :param activation: Activation function for attention weights (default softmax).
        A non-softmax activation f is applied element-wise and followed by a
        softmax, i.e. A = softmax(f(scores)).
    :type activation: Activation, optional
    :param d_k: Dimension of queries, keys and values (default ``d_model``)
    :type d_k: int, optional
    :param output_projection: Whether to apply the output projection W_o
    :type output_projection: bool, optional
    :param causal: If True, position t can only attend to positions <= t
        (needed for next-token prediction)
    :type causal: bool, optional
    """

    def __init__(self, d_model, dropout=0.0, activation=None, d_k=None,
                 output_projection=True, causal=False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model if d_k is None else d_k
        self.dropout = dropout
        self.causal = causal

        # Set up activation function (default to softmax for attention)
        if activation is None:
            self.activation = SoftmaxActivation()
        else:
            self.activation = activation

        # Weight matrices for Q, K, V projections, d_model -> d_k
        # He initialisation: variance 2 / fan_in
        std = np.sqrt(2.0 / d_model)
        self.W_q = np.random.normal(0, std, (d_model, self.d_k))
        self.W_k = np.random.normal(0, std, (d_model, self.d_k))
        self.W_v = np.random.normal(0, std, (d_model, self.d_k))
        if output_projection:
            self.W_o = np.random.normal(0, np.sqrt(2.0 / self.d_k), (self.d_k, d_model))
        else:
            self.W_o = None

        # Scale factor for attention
        self.scale = 1.0 / np.sqrt(self.d_k)

        # Store for backward pass
        self.last_inputs = None
        self.last_output = None
        self.last_attention_weights = None
        self._cache = None

    def _uses_axis_activation(self, method):
        return hasattr(self.activation, method) and 'axis' in getattr(self.activation, method).__code__.co_varnames

    def _causal_mask(self, n_queries, n_keys):
        """Boolean mask that is True where attention is *not* allowed."""
        return np.triu(np.ones((n_queries, n_keys), dtype=bool), k=1)

    def forward(self, x, query_input=None, key_value_input=None):
        """
        Forward pass of the attention mechanism.

        Supports three modes:
        1. Self-attention: forward(x) - x becomes Q, K, V
        2. Cross-attention: forward(x, query_input=q, key_value_input=kv) - Q=q, K=V=kv
        3. Mixed: forward(x, key_value_input=kv) - Q=x, K=V=kv

        :param x: Primary input tensor of shape (batch_size, seq_len, d_model)
        :type x: numpy.ndarray
        :param query_input: Optional separate query input (for cross-attention)
        :type query_input: numpy.ndarray, optional
        :param key_value_input: Optional separate key/value input (for cross-attention)
        :type key_value_input: numpy.ndarray, optional
        :returns: Attention output of shape (batch_size, seq_len, d_model),
            or (batch_size, seq_len, d_k) if ``output_projection=False``
        :rtype: numpy.ndarray
        """
        if query_input is not None and key_value_input is None:
            raise ValueError("key_value_input must be provided when query_input is given")

        # Store inputs for backward pass
        self.last_inputs = (x, query_input, key_value_input)

        q_source = query_input if query_input is not None else x
        kv_source = key_value_input if key_value_input is not None else x

        Q = q_source @ self.W_q
        K = kv_source @ self.W_k
        V = kv_source @ self.W_v

        # Attention scores: Q K^T / sqrt(d_k)
        scores = (Q @ K.transpose(0, 2, 1)) * self.scale
        mask = self._causal_mask(scores.shape[-2], scores.shape[-1]) if self.causal else None

        # Attention weights
        if self._uses_axis_activation('forward'):
            masked = np.where(mask, -np.inf, scores) if mask is not None else scores
            weights = self.activation.forward(masked, axis=-1)
        else:
            # Element-wise activation followed by softmax normalisation
            activated = self.activation.forward(scores)
            if mask is not None:
                activated = np.where(mask, -np.inf, activated)
            weights = self._softmax(activated, axis=-1)

        # Dropout on the attention weights (inverted dropout)
        if self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, weights.shape) / (1 - self.dropout)
            dropped = weights * dropout_mask
        else:
            dropout_mask = None
            dropped = weights

        attended = dropped @ V
        output = attended @ self.W_o if self.W_o is not None else attended

        self._cache = dict(q_source=q_source, kv_source=kv_source, Q=Q, K=K, V=V,
                           scores=scores, weights=weights, dropout_mask=dropout_mask,
                           dropped=dropped, attended=attended)
        self.last_output = output
        self.last_attention_weights = weights
        return output

    def backward(self, grad_output):
        """
        Backward pass implementing the attention gradient computation.

        Computes the gradients with respect to the inputs (returned) and with
        respect to W_q, W_k, W_v and W_o (available from ``gradients``).

        :param grad_output: Gradient from the next layer
        :type grad_output: numpy.ndarray
        :returns: Tuple of input gradients: ``(grad_x,)`` for self-attention,
            ``(grad_query, grad_key_value)`` for cross-attention and
            ``(grad_x, grad_key_value)`` for mixed attention
        :rtype: tuple
        """
        if self._cache is None:
            raise ValueError("Must call forward before backward")
        grad_output = _unwrap_gradient(grad_output)
        c = self._cache
        x, query_input, key_value_input = self.last_inputs

        # Output projection: output = attended W_o
        if self.W_o is not None:
            grad_W_o = np.einsum('bti,btj->ij', c['attended'], grad_output)
            grad_attended = grad_output @ self.W_o.T
        else:
            grad_attended = grad_output

        # attended = dropped V
        grad_dropped = grad_attended @ c['V'].transpose(0, 2, 1)
        grad_V = c['dropped'].transpose(0, 2, 1) @ grad_attended

        # Dropout: dropped = weights * mask
        grad_weights = grad_dropped * c['dropout_mask'] if c['dropout_mask'] is not None else grad_dropped

        # Softmax (and optional element-wise activation)
        weights = c['weights']
        grad_sum = np.sum(grad_weights * weights, axis=-1, keepdims=True)
        grad_pre_softmax = weights * (grad_weights - grad_sum)
        if self._uses_axis_activation('forward'):
            grad_scores = grad_pre_softmax
        else:
            grad_scores = grad_pre_softmax * self.activation.gradient(c['scores'])

        # scores = Q K^T * scale
        grad_Q = (grad_scores @ c['K']) * self.scale
        grad_K = (grad_scores.transpose(0, 2, 1) @ c['Q']) * self.scale

        # Weight gradients: Q = X_q W_q, K = X_kv W_k, V = X_kv W_v
        grads = [np.einsum('bti,btj->ij', c['q_source'], grad_Q),
                 np.einsum('bti,btj->ij', c['kv_source'], grad_K),
                 np.einsum('bti,btj->ij', c['kv_source'], grad_V)]
        if self.W_o is not None:
            grads.append(grad_W_o)
        self._parameter_gradients = np.concatenate([g.flatten() for g in grads])

        # Input gradients
        grad_from_q = grad_Q @ self.W_q.T
        grad_from_kv = grad_K @ self.W_k.T + grad_V @ self.W_v.T
        if query_input is not None:
            # Cross-attention: separate gradients for query and key_value inputs
            return grad_from_q, grad_from_kv
        if key_value_input is not None:
            # Mixed attention: only Q comes from x
            return grad_from_q, grad_from_kv
        # Self-attention: x appears in Q, K and V (three-path chain rule)
        return (grad_from_q + grad_from_kv,)

    def _softmax(self, x, axis=-1):
        """Numerically stable softmax implementation."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _weight_matrices(self):
        mats = [self.W_q, self.W_k, self.W_v]
        if self.W_o is not None:
            mats.append(self.W_o)
        return mats

    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.

        Returns parameters in the order: [W_q, W_k, W_v, W_o] (W_o only if
        the layer has an output projection).

        :returns: 1D array of all trainable parameters
        :rtype: numpy.ndarray
        """
        return np.concatenate([W.flatten() for W in self._weight_matrices()])

    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.

        Updates parameters in the order: [W_q, W_k, W_v, W_o].

        :param value: 1D array of parameters to set
        :type value: numpy.ndarray
        :raises ValueError: If the parameter vector has incorrect length
        """
        mats = self._weight_matrices()
        expected_length = sum(W.size for W in mats)
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")
        start = 0
        new = []
        for W in mats:
            new.append(value[start:start + W.size].reshape(W.shape))
            start += W.size
        self.W_q, self.W_k, self.W_v = new[:3]
        if self.W_o is not None:
            self.W_o = new[3]


class MultiHeadAttentionLayer(Layer):
    """
    Multi-head attention layer built from multiple AttentionLayer instances.

    Each head projects the *full* d_model-dimensional input to its own
    d_k = d_model / n_heads dimensional queries, keys and values, so every
    head can attend using every input feature. The head outputs are
    concatenated and combined with an output projection W_o:

        MultiHead(X) = concat(head_1, ..., head_h) W_o

    Parameters
    ----------
    d_model : int
        The dimension of the model
    n_heads : int
        Number of attention heads
    dropout : float, optional
        Dropout rate, by default 0.0
    activation : Activation, optional
        Activation function for attention weights
    causal : bool, optional
        If True, use causal (autoregressive) masking in every head
    """

    def __init__(self, d_model, n_heads, dropout=0.0, activation=None, causal=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal

        # Each head maps d_model -> d_k and has no output projection of its own
        self.attention_heads = [AttentionLayer(d_model, dropout=dropout, activation=activation,
                                               d_k=self.d_k, output_projection=False, causal=causal)
                                for _ in range(n_heads)]

        # Output projection to combine all heads
        self.W_o = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))

        # Store for backward pass
        self.last_inputs = None
        self.last_output = None
        self.last_head_outputs = None
        self.last_concatenated = None

    @property
    def last_attention_weights(self):
        """Attention weights from the last forward pass, shape (batch, n_heads, seq_len, seq_len)."""
        if self.last_inputs is None:
            return None
        return np.stack([head.last_attention_weights for head in self.attention_heads], axis=1)

    def forward(self, x):
        """
        Forward pass through multiple attention heads (self-attention).

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (batch_size, seq_len, d_model)

        Returns
        -------
        np.ndarray
            Multi-head attention output of shape (batch_size, seq_len, d_model)
        """
        self.last_inputs = x

        # Every head sees the full input
        head_outputs = [head.forward(x) for head in self.attention_heads]

        concatenated = np.concatenate(head_outputs, axis=-1)
        output = concatenated @ self.W_o

        self.last_output = output
        self.last_head_outputs = head_outputs
        self.last_concatenated = concatenated
        return output

    def backward(self, grad_output):
        """
        Backward pass through multiple attention heads.

        Because x feeds every head, the input gradient is the *sum* of the
        input gradients from each head.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient from the next layer

        Returns
        -------
        tuple
            ``(grad_input,)``, the gradient with respect to the input
        """
        if self.last_inputs is None:
            raise ValueError("Must call forward before backward")
        grad_output = _unwrap_gradient(grad_output)

        # Output projection
        grad_W_o = np.einsum('bti,btj->ij', self.last_concatenated, grad_output)
        grad_concatenated = grad_output @ self.W_o.T

        grad_input = np.zeros_like(self.last_inputs)
        for i, head in enumerate(self.attention_heads):
            grad_head = grad_concatenated[..., i * self.d_k:(i + 1) * self.d_k]
            grad_input = grad_input + head.backward(grad_head)[0]

        self._parameter_gradients = np.concatenate(
            [head.gradients for head in self.attention_heads] + [grad_W_o.flatten()])

        return (grad_input,)

    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.

        Returns parameters from all attention heads plus output projection.

        Returns
        -------
        np.ndarray
            1D array of all trainable parameters
        """
        params = [head.parameters for head in self.attention_heads]
        params.append(self.W_o.flatten())
        return np.concatenate(params)

    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.

        Parameters
        ----------
        value : np.ndarray
            1D array of parameters to set
        """
        expected_length = sum(len(head.parameters) for head in self.attention_heads) + self.W_o.size
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")

        start = 0
        for head in self.attention_heads:
            head_size = len(head.parameters)
            head.parameters = value[start:start + head_size]
            start += head_size
        self.W_o = value[start:].reshape(self.W_o.shape)


class PositionalEncodingLayer(Layer):
    """
    Sinusoidal positional encoding layer for sequence data.

    This implements the standard sinusoidal positional encoding from
    "Attention Is All You Need" (Vaswani et al., 2017) as a Layer.

    Parameters
    ----------
    d_model : int
        The dimension of the model
    max_length : int, optional
        Maximum sequence length, by default 5000
    """

    def __init__(self, d_model, max_length=5000):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        # Create positional encoding matrix (static, no parameters)
        self.pe = self._create_positional_encoding(max_length, d_model)

    def _create_positional_encoding(self, max_length, d_model):
        """Create sinusoidal positional encoding matrix."""
        pe = np.zeros((max_length, d_model))
        position = np.arange(0, max_length, dtype=float).reshape(-1, 1)

        # Create the sinusoidal encoding
        div_term = np.exp(np.arange(0, d_model, 2) *
                         (-np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def forward(self, x):
        """
        Add positional encoding to input.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (batch_size, seq_len, d_model)

        Returns
        -------
        np.ndarray
            Input with positional encoding added
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]

    def backward(self, grad_output):
        """
        Backward pass for positional encoding.

        Since positional encoding is just addition, the gradient
        is passed through unchanged.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient from the next layer

        Returns
        -------
        np.ndarray
            Gradient with respect to input (same as grad_output)
        """
        return (grad_output,)

    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.

        Positional encoding has no trainable parameters.

        Returns
        -------
        np.ndarray
            Empty array (no parameters)
        """
        return np.array([])

    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.

        Positional encoding has no trainable parameters, so this
        just checks that an empty array is provided.

        Parameters
        ----------
        value : np.ndarray
            Should be empty array
        """
        if len(value) != 0:
            raise ValueError("PositionalEncodingLayer has no trainable parameters")


class ConvolutionalLayer(Layer):
    """
    Convolutional layer for processing 2D images.

    This layer implements 2D convolution with multiple filters, padding, and stride.
    It follows the standard CNN architecture where each filter learns to detect
    specific features in the input.

    Parameters
    ----------
    input_channels : int
        Number of input channels (e.g., 1 for grayscale, 3 for RGB)
    output_channels : int
        Number of output channels (number of filters)
    kernel_size : int or tuple
        Size of the convolution kernel (height, width)
    stride : int or tuple, optional
        Stride of the convolution, by default 1
    padding : int or tuple, optional
        Padding to add to input, by default 0
    activation : Activation, optional
        Activation function to apply after convolution
    """

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, activation=None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Handle kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Handle stride
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # Handle padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Initialize filters (kernels) and biases
        # Shape: (output_channels, input_channels, kernel_height, kernel_width)
        self.filters = np.random.normal(0, np.sqrt(2.0 / (input_channels * self.kernel_size[0] * self.kernel_size[1])),
                                      (output_channels, input_channels, self.kernel_size[0], self.kernel_size[1]))
        self.biases = np.zeros(output_channels)

        # Set up activation function
        if activation is None:
            self.activation = ReLUActivation()
        else:
            self.activation = activation

        # Store for backward pass
        self.last_input = None
        self.last_output = None
        self.last_conv_output = None

    def _pad(self, x):
        """Zero-pad the two spatial dimensions."""
        ph, pw = self.padding
        return np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')

    def _windows(self, x_padded):
        """
        All K_h x K_w patches of the padded input that the filters are applied to.

        Returns an array of shape (batch, in_channels, out_height, out_width, K_h, K_w)
        whose entry [b, k, h, w, i, j] is x_padded[b, k, h*S_h + i, w*S_w + j].
        """
        kh, kw = self.kernel_size
        sh, sw = self.stride
        windows = np.lib.stride_tricks.sliding_window_view(x_padded, (kh, kw), axis=(2, 3))
        return windows[:, :, ::sh, ::sw]

    def forward(self, x):
        """
        Forward pass of the convolutional layer.

        Computes, for every example b, output channel c and output position (h, w),

            z[b, c, h, w] = sum_{k, i, j} x_pad[b, k, h*S + i, w*S + j] * F[c, k, i, j] + bias[c]

        and then applies the activation function element-wise.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (batch_size, input_channels, height, width)

        Returns
        -------
        np.ndarray
            Output tensor of shape (batch_size, output_channels, out_height, out_width)
        """
        # Store input for backward pass
        self.last_input = x

        windows = self._windows(self._pad(x))
        conv_output = np.einsum('bkhwij,ckij->bchw', windows, self.filters) \
            + self.biases[None, :, None, None]

        # Apply activation function
        self.last_conv_output = conv_output
        output = self.activation.forward(conv_output) if hasattr(self.activation, 'forward') else conv_output
        self.last_output = output
        return output

    def backward(self, grad_output):
        """
        Backward pass of the convolutional layer.

        Uses

            dL/dF[c, k, i, j] = sum_{b, h, w} dL/dz[b, c, h, w] * x_pad[b, k, h*S + i, w*S + j]
            dL/dbias[c]       = sum_{b, h, w} dL/dz[b, c, h, w]
            dL/dx_pad[b, k, h*S + i, w*S + j] += sum_c dL/dz[b, c, h, w] * F[c, k, i, j]

        The filter and bias gradients are stored and available from ``gradients``.

        Parameters
        ----------
        grad_output : np.ndarray or tuple
            Gradient from the next layer

        Returns
        -------
        tuple
            Gradient with respect to input
        """
        if self.last_input is None:
            raise ValueError("Must call forward before backward")
        grad_output = _unwrap_gradient(grad_output)

        # Gradient through activation function
        if hasattr(self.activation, 'gradient'):
            grad_conv = grad_output * self.activation.gradient(self.last_conv_output)
        else:
            grad_conv = grad_output

        x_padded = self._pad(self.last_input)
        windows = self._windows(x_padded)

        # Parameter gradients: the shared filter accumulates over every position and example
        grad_filters = np.einsum('bchw,bkhwij->ckij', grad_conv, windows)
        grad_biases = grad_conv.sum(axis=(0, 2, 3))
        self._parameter_gradients = np.concatenate([grad_filters.flatten(), grad_biases.flatten()])

        # Input gradient: each filter element (i, j) sends gradient back to the
        # input positions it was multiplied with
        kh, kw = self.kernel_size
        sh, sw = self.stride
        out_h, out_w = grad_conv.shape[2], grad_conv.shape[3]
        grad_padded = np.zeros_like(x_padded, dtype=float)
        for i in range(kh):
            for j in range(kw):
                grad_padded[:, :, i:i + sh * out_h:sh, j:j + sw * out_w:sw] += \
                    np.einsum('bchw,ck->bkhw', grad_conv, self.filters[:, :, i, j])

        # Remove padding (explicit end indices so zero padding in one dimension works)
        ph, pw = self.padding
        height, width = self.last_input.shape[2], self.last_input.shape[3]
        grad_input = grad_padded[:, :, ph:ph + height, pw:pw + width]

        return (grad_input,)

    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.

        Returns
        -------
        np.ndarray
            1D array of all trainable parameters
        """
        return np.concatenate([self.filters.flatten(), self.biases.flatten()])

    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.

        Parameters
        ----------
        value : np.ndarray
            1D array of parameters to set
        """
        expected_length = self.filters.size + self.biases.size
        if len(value) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(value)}")

        # Set filters
        filters_size = self.filters.size
        self.filters = value[:filters_size].reshape(self.filters.shape)

        # Set biases
        self.biases = value[filters_size:].reshape(self.biases.shape)


class MaxPoolingLayer(Layer):
    """
    Max pooling layer for downsampling feature maps.

    This layer reduces the spatial dimensions by taking the maximum value
    in each pooling region, helping to reduce computational complexity
    and provide translation invariance.

    Parameters
    ----------
    pool_size : int or tuple
        Size of the pooling window (height, width)
    stride : int or tuple, optional
        Stride of the pooling operation, by default None (same as pool_size)
    """

    def __init__(self, pool_size, stride=None):
        super().__init__()

        # Handle pool size
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size

        # Handle stride (default to pool_size)
        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # Store for backward pass
        self.last_input = None
        self.last_output = None
        self.max_indices = None

    def forward(self, x):
        """
        Forward pass of the max pooling layer.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        np.ndarray
            Output tensor of shape (batch_size, channels, out_height, out_width)
        """
        # Store input for backward pass
        self.last_input = x

        ph, pw = self.pool_size
        sh, sw = self.stride

        # Every pooling region: shape (batch, channels, out_height, out_width, ph, pw)
        windows = np.lib.stride_tricks.sliding_window_view(x, (ph, pw), axis=(2, 3))[:, :, ::sh, ::sw]
        flat = windows.reshape(windows.shape[:4] + (ph * pw,))

        # Position of the maximum within each region (first one in case of ties)
        arg = np.argmax(flat, axis=-1)
        output = np.take_along_axis(flat, arg[..., None], axis=-1)[..., 0]

        # Convert to input coordinates for the backward pass
        out_height, out_width = output.shape[2], output.shape[3]
        rows = np.arange(out_height)[:, None] * sh + arg // pw
        cols = np.arange(out_width)[None, :] * sw + arg % pw
        self.max_indices = np.stack([rows, cols], axis=-1)

        self.last_output = output
        return output

    def backward(self, grad_output):
        """
        Backward pass of the max pooling layer.

        Each output gradient is routed to the input position that achieved the
        maximum in its pooling region; all other positions receive zero. If
        pooling regions overlap, gradients arriving at the same position are summed.

        Parameters
        ----------
        grad_output : np.ndarray or tuple
            Gradient from the next layer

        Returns
        -------
        tuple
            Gradient with respect to input
        """
        if self.last_input is None:
            raise ValueError("Must call forward before backward")
        grad_output = _unwrap_gradient(grad_output)

        grad_input = np.zeros_like(self.last_input, dtype=float)
        batch_size, channels, out_height, out_width = grad_output.shape
        b_idx = np.arange(batch_size)[:, None, None, None]
        c_idx = np.arange(channels)[None, :, None, None]
        np.add.at(grad_input, (b_idx, c_idx, self.max_indices[..., 0], self.max_indices[..., 1]), grad_output)

        return (grad_input,)

    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.

        Max pooling has no trainable parameters.

        Returns
        -------
        np.ndarray
            Empty array (no parameters)
        """
        return np.array([])

    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.

        Max pooling has no trainable parameters, so this
        just checks that an empty array is provided.

        Parameters
        ----------
        value : np.ndarray
            Should be empty array
        """
        if len(value) != 0:
            raise ValueError("MaxPoolingLayer has no trainable parameters")


class FlattenLayer(Layer):
    """
    Flatten layer for converting multi-dimensional tensors to 1D.

    This layer is commonly used to transition from convolutional layers
    to fully connected layers by flattening the spatial dimensions.

    Parameters
    ----------
    start_dim : int, optional
        First dimension to flatten, by default 1
    end_dim : int, optional
        Last dimension to flatten, by default -1
    """

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.original_shape = None

    def forward(self, x):
        """
        Forward pass of the flatten layer.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of any shape

        Returns
        -------
        np.ndarray
            Flattened tensor
        """
        # Store original shape for backward pass
        self.original_shape = x.shape

        # Flatten the specified dimensions
        if self.start_dim == 1 and self.end_dim == -1:
            # Standard flattening: keep batch dimension, flatten the rest
            return x.reshape(x.shape[0], -1)
        else:
            # Custom flattening
            shape = list(x.shape)
            flattened_dims = np.prod(shape[self.start_dim:self.end_dim+1])
            new_shape = shape[:self.start_dim] + [flattened_dims] + shape[self.end_dim+1:]
            return x.reshape(new_shape)

    def backward(self, grad_output):
        """
        Backward pass of the flatten layer.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient from the next layer

        Returns
        -------
        np.ndarray
            Gradient with respect to input
        """
        if self.original_shape is None:
            raise ValueError("Must call forward before backward")

        # Reshape gradient back to original shape
        grad_output = _unwrap_gradient(grad_output)
        return (grad_output.reshape(self.original_shape),)

    @property
    def parameters(self):
        """
        Get all trainable parameters as a 1D vector.

        Flatten layer has no trainable parameters.

        Returns
        -------
        np.ndarray
            Empty array (no parameters)
        """
        return np.array([])

    @parameters.setter
    def parameters(self, value):
        """
        Set all trainable parameters from a 1D vector.

        Flatten layer has no trainable parameters, so this
        just checks that an empty array is provided.

        Parameters
        ----------
        value : np.ndarray
            Should be empty array
        """
        if len(value) != 0:
            raise ValueError("FlattenLayer has no trainable parameters")




