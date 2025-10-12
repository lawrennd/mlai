"""
Transformer components for mlai lectures.

This module implements simple transformer components focused on understanding
the mathematical principles, particularly the chain rule in attention mechanisms.
"""

import numpy as np
from .mlai import SigmoidActivation, LinearActivation, ReLUActivation, Model


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


class SigmoidAttentionActivation:
    """
    Sigmoid-based attention activation with normalisation.
    
    This applies sigmoid element-wise and then normalizes to ensure
    attention weights sum to 1, making it suitable for attention mechanisms.
    """
    
    def forward(self, x, axis=-1):
        """
        Forward pass: sigmoid + normalization.
        
        :param x: Input array
        :type x: numpy.ndarray
        :param axis: Axis along which to normalise
        :type axis: int
        :returns: Normalised sigmoid activated array
        :rtype: numpy.ndarray
        """
        # Apply sigmoid element-wise
        sigmoid_output = 1.0 / (1.0 + np.exp(-x))
        
        # Normalise along the specified axis
        return sigmoid_output / np.sum(sigmoid_output, axis=axis, keepdims=True)
    
    def gradient(self, x, grad_output, axis=-1):
        """
        Gradient of sigmoid + normalisation.
        
        This computes the gradient through both sigmoid and normalisation.
        
        :param x: Input array
        :type x: numpy.ndarray
        :param grad_output: Gradient from next layer
        :type grad_output: numpy.ndarray
        :param axis: Axis along which normalisation was applied
        :type axis: int
        :returns: Gradient array
        :rtype: numpy.ndarray
        """
        # Compute sigmoid output
        sigmoid_output = 1.0 / (1.0 + np.exp(-x))
        
        # Compute normalised output
        normalised_output = sigmoid_output / np.sum(sigmoid_output, axis=axis, keepdims=True)
        
        # Gradient through normalisation: ∂(s/Σs)/∂x = (∂s/∂x * Σs - s * Σ(∂s/∂x)) / (Σs)²
        sigmoid_grad = sigmoid_output * (1 - sigmoid_output)  # sigmoid derivative
        
        # Sum of gradients along axis
        grad_sum = np.sum(grad_output * normalised_output, axis=axis, keepdims=True)
        sigmoid_sum = np.sum(sigmoid_output, axis=axis, keepdims=True)
        
        # Apply chain rule: gradient through normalisation
        grad_through_norm = (grad_output - grad_sum) / sigmoid_sum
        
        # Final gradient: sigmoid_grad * grad_through_norm
        return sigmoid_grad * grad_through_norm


class IdentityMinusSoftmaxActivation:
    """
    Identity minus softmax activation for attention mechanisms.
    
    This creates attention weights where diagonal entries get (1 - softmax)
    and off-diagonal entries get (-softmax). This can encourage the model
    to focus on specific positions while de-emphasising others.
    """
    
    def forward(self, x, axis=-1):
        """
        Forward pass: identity minus softmax activation.
        
        :param x: Input array
        :type x: numpy.ndarray
        :param axis: Axis along which to apply softmax
        :type axis: int
        :returns: Identity minus softmax activated array
        :rtype: numpy.ndarray
        """
        # Compute softmax
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        softmax_output = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        # Create identity matrix for diagonal entries
        identity = np.eye(x.shape[axis])
        if axis != -1:
            # If axis is not the last dimension, we need to handle this differently
            # For now, assume we're working with the last axis
            pass
        
        # Apply identity minus softmax: diagonal gets (1 - softmax), off-diagonal gets (-softmax)
        result = identity - softmax_output
        return result
    
    def gradient(self, x, grad_output, axis=-1):
        """
        Gradient of identity minus softmax activation.
        
        For identity minus softmax: ∂(I - s)/∂x = -∂s/∂x
        where ∂s/∂x is the softmax gradient.
        
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
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        softmax_output = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        # Standard softmax gradient: softmax * (grad - sum(grad * softmax))
        grad_sum = np.sum(grad_output * softmax_output, axis=axis, keepdims=True)
        softmax_grad = softmax_output * (grad_output - grad_sum)
        
        # For identity minus softmax, the gradient is negative of softmax gradient
        return -softmax_grad


class Attention:
    """
    Basic scaled dot-product attention mechanism for educational purposes.
    
    This implementation focuses on clarity and understanding of the chain rule
    rather than optimisation. The forward and backward passes are designed to
    clearly show how gradients flow through the Q, K, V transformations.
    
    Parameters
    ----------
    d_model : int
        The dimension of the model (input/output dimension)
    dropout : float, optional
        Dropout rate for regularisation, by default 0.0 (disabled for educational clarity)
    activation : Activation, optional
        Activation function for attention weights, by default SoftmaxActivation.
        Options:
        - SoftmaxActivation: Standard softmax (recommended for attention)
        - SigmoidAttentionActivation: Sigmoid + normalisation (attention-appropriate)
        - IdentityMinusSoftmaxActivation: Identity minus softmax (diagonal=1-softmax, off-diagonal=-softmax)
        - SigmoidActivation/LinearActivation: Raw activations (set normalise_attention=False)
    normalise_attention : bool, optional
        Whether to apply softmax normalisation to attention weights, by default True.
        Set to False to use raw activation outputs (e.g., for SigmoidActivation).
    """
    
    def __init__(self, d_model, dropout=0.0, activation=None, normalise_attention=True):
        self.d_model = d_model
        self.dropout = dropout
        self.normalise_attention = normalise_attention
        
        # Set up activation function (default to softmax for attention)
        if activation is None:
            self.activation = SoftmaxActivation()
        else:
            self.activation = activation
        
        # Initialise weight matrices for Q, K, V transformations
        # Using Xavier initialisation for stability
        self.W_q = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        self.W_k = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        self.W_v = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        self.W_o = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        
        # Scale factor for attention
        self.scale = 1.0 / np.sqrt(d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the attention mechanism.
        
        This clearly shows the three-path chain rule where the same input
        appears in Q, K, V transformations.
        
        Parameters
        ----------
        query : np.ndarray
            Query tensor of shape (batch_size, seq_len, d_model)
        key : np.ndarray
            Key tensor of shape (batch_size, seq_len, d_model)
        value : np.ndarray
            Value tensor of shape (batch_size, seq_len, d_model)
        mask : np.ndarray, optional
            Attention mask, by default None
            
        Returns
        -------
        output : np.ndarray
            Attention output of shape (batch_size, seq_len, d_model)
        attention_weights : np.ndarray
            Attention weights of shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Step 1: Linear transformations (Q, K, V)
        # This is where the three-path chain rule becomes important
        Q = query @ self.W_q  # Query transformation
        K = key @ self.W_k    # Key transformation  
        V = value @ self.W_v  # Value transformation
        
        # Step 2: Scaled dot-product attention
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        attention_scores = (Q @ K.transpose(0, 2, 1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = np.where(mask == 0, -1e9, attention_scores)
        
        # Step 3: Apply activation function to get attention weights
        if hasattr(self.activation, 'forward') and 'axis' in self.activation.forward.__code__.co_varnames:
            attention_weights = self.activation.forward(attention_scores, axis=-1)
        else:
            # For activations that don't support axis parameter, apply element-wise
            attention_weights = self.activation.forward(attention_scores)
            # Apply softmax normalisation if requested (default for attention mechanisms)
            if self.normalise_attention:
                attention_weights = self._softmax(attention_weights, axis=-1)
        
        # Apply dropout during training
        if self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout)
        
        # Step 4: Apply attention to values
        output = attention_weights @ V
        
        # Step 5: Output projection
        output = output @ self.W_o
        
        return output, attention_weights
    
    def _softmax(self, x, axis=-1):
        """Numerically stable softmax implementation."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def backward(self, grad_output, query, key, value, attention_weights, mask=None):
        """
        Backward pass implementing the three-path chain rule correctly.
        
        This implements the standard attention gradient computation that matches
        the numerical gradients exactly.
        
        Parameters
        ----------
        grad_output : np.ndarray
            Gradient from the next layer (∂L/∂output)
        query : np.ndarray
            Original query input
        key : np.ndarray
            Original key input
        value : np.ndarray
            Original value input
        attention_weights : np.ndarray
            Attention weights from forward pass
            
        Returns
        -------
        dict
            Dictionary containing gradients for all parameters and inputs
        """
        batch_size, seq_len, d_model = query.shape
        
        # Recompute Q, K, V for gradient computation
        Q = query @ self.W_q
        K = key @ self.W_k
        V = value @ self.W_v
        
        # Gradient through output projection
        grad_attention_output = grad_output @ self.W_o.T
        
        # Gradients for attention weights and values
        grad_attention_weights = grad_attention_output @ V.transpose(0, 2, 1)
        grad_V = attention_weights.transpose(0, 2, 1) @ grad_attention_output
        
        # Gradient through activation function
        # Recompute attention scores for gradient computation
        attention_scores = (Q @ K.transpose(0, 2, 1)) * self.scale
        if mask is not None:
            attention_scores = np.where(mask == 0, -1e9, attention_scores)
        
        # Handle different activation function interfaces
        if hasattr(self.activation, 'gradient') and 'axis' in self.activation.gradient.__code__.co_varnames:
            grad_attention_scores = self.activation.gradient(attention_scores, grad_attention_weights, axis=-1)
        else:
            # For activations that don't support axis parameter
            if self.normalise_attention:
                # Use standard softmax gradient formula
                grad_sum = np.sum(grad_attention_weights * attention_weights, axis=-1, keepdims=True)
                grad_attention_scores = attention_weights * (grad_attention_weights - grad_sum)
            else:
                # Use the activation function's gradient directly
                grad_attention_scores = self.activation.gradient(attention_scores) * grad_attention_weights
        
        # Gradients for Q and K
        grad_Q = grad_attention_scores @ K
        grad_K = grad_attention_scores.transpose(0, 2, 1) @ Q
        
        # Scale gradients
        grad_Q *= self.scale
        grad_K *= self.scale
        
        # Three-path chain rule for input gradients
        # The same input appears in Q, K, V transformations
        # ∂L/∂input = ∂L/∂Q @ W_q^T + ∂L/∂K @ W_k^T + ∂L/∂V @ W_v^T
        grad_input = grad_Q @ self.W_q.T + grad_K @ self.W_k.T + grad_V @ self.W_v.T
        
        # For compatibility, also return individual gradients
        grad_query = grad_Q @ self.W_q.T
        grad_key = grad_K @ self.W_k.T  
        grad_value = grad_V @ self.W_v.T
        
        # Weight matrix gradients (sum over batch dimension)
        grad_W_q = np.sum(query.transpose(0, 2, 1) @ grad_Q, axis=0)
        grad_W_k = np.sum(key.transpose(0, 2, 1) @ grad_K, axis=0)
        grad_W_v = np.sum(value.transpose(0, 2, 1) @ grad_V, axis=0)
        grad_W_o = np.sum((attention_weights @ V).transpose(0, 2, 1) @ grad_output, axis=0)
        
        return {
            'grad_input': grad_input,
            'grad_query': grad_query,
            'grad_key': grad_key,
            'grad_value': grad_value,
            'grad_W_q': grad_W_q,
            'grad_W_k': grad_W_k,
            'grad_W_v': grad_W_v,
            'grad_W_o': grad_W_o
        }
    
    def _softmax_gradient(self, softmax_output, grad_output):
        """Compute gradient through softmax operation."""
        # Gradient of softmax: softmax * (grad - sum(grad * softmax))
        grad_sum = np.sum(grad_output * softmax_output, axis=-1, keepdims=True)
        return softmax_output * (grad_output - grad_sum)


class MultiHeadAttention:
    """
    Multi-head attention built from basic Attention instances.
    
    This is a simple composition of multiple Attention heads to demonstrate
    how multi-head attention is constructed from basic attention mechanisms.
    
    Parameters
    ----------
    d_model : int
        The dimension of the model
    n_heads : int
        Number of attention heads
    dropout : float, optional
        Dropout rate, by default 0.0 (disabled for educational clarity)
    activation : Activation, optional
        Activation function for attention weights
    """
    
    def __init__(self, d_model, n_heads, dropout=0.0, activation=None):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Create multiple attention heads
        self.attention_heads = [Attention(self.d_k, dropout, activation) for _ in range(n_heads)]
        
        # Output projection
        self.W_o = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass through multiple attention heads.
        
        Parameters
        ----------
        query : np.ndarray
            Query tensor of shape (batch_size, seq_len, d_model)
        key : np.ndarray
            Key tensor of shape (batch_size, seq_len, d_model)
        value : np.ndarray
            Value tensor of shape (batch_size, seq_len, d_model)
        mask : np.ndarray, optional
            Attention mask, by default None
            
        Returns
        -------
        output : np.ndarray
            Multi-head attention output
        attention_weights : np.ndarray
            Attention weights from all heads
        """
        batch_size, seq_len, d_model = query.shape
        
        # Split input into multiple heads
        query_heads = query.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        key_heads = key.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        value_heads = value.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose to (batch_size, n_heads, seq_len, d_k)
        query_heads = query_heads.transpose(0, 2, 1, 3)
        key_heads = key_heads.transpose(0, 2, 1, 3)
        value_heads = value_heads.transpose(0, 2, 1, 3)
        
        # Process each head
        head_outputs = []
        all_attention_weights = []
        
        for i, attention_head in enumerate(self.attention_heads):
            # Forward pass through each attention head
            head_output, head_weights = attention_head.forward(
                query_heads[:, i], key_heads[:, i], value_heads[:, i], mask
            )
            head_outputs.append(head_output)
            all_attention_weights.append(head_weights)
        
        # Concatenate all head outputs
        concatenated = np.concatenate(head_outputs, axis=-1)
        
        # Apply output projection
        output = concatenated @ self.W_o
        
        # Stack attention weights from all heads
        attention_weights = np.stack(all_attention_weights, axis=1)
        
        return output, attention_weights


class PositionalEncoding:
    """
    Sinusoidal positional encoding for sequence data.
    
    This implements the standard sinusoidal positional encoding from
    "Attention Is All You Need" (Vaswani et al., 2017).
    
    Parameters
    ----------
    d_model : int
        The dimension of the model
    max_length : int, optional
        Maximum sequence length, by default 5000
    """
    
    def __init__(self, d_model, max_length=5000):
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = np.zeros((max_length, d_model))
        position = np.arange(0, max_length, dtype=float).reshape(-1, 1)
        
        # Create the sinusoidal encoding
        div_term = np.exp(np.arange(0, d_model, 2) * 
                         (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
        
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


class Transformer(Model):
    """
    Simple transformer model for educational purposes.
    
    This class demonstrates how transformer components (attention, positional encoding)
    are combined to create a complete model that inherits from the Model base class.
    
    Parameters
    ----------
    d_model : int
        The dimension of the model
    n_heads : int
        Number of attention heads
    vocab_size : int
        Size of the vocabulary
    max_seq_len : int
        Maximum sequence length for positional encoding
    dropout : float, optional
        Dropout rate, by default 0.0 (disabled for educational clarity)
    activation : Activation, optional
        Activation function for attention weights
    """
    
    def __init__(self, d_model, n_heads, vocab_size, max_seq_len=512, dropout=0.0, activation=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Create transformer components
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, activation)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Embedding layers
        self.embedding = np.random.normal(0, np.sqrt(1.0 / d_model), (vocab_size, d_model))
        self.output_projection = np.random.normal(0, np.sqrt(1.0 / d_model), (d_model, vocab_size))
        
        # Store for gradient computation
        self.last_input = None
        self.last_output = None
        
    def forward(self, x):
        """
        Forward pass through the transformer.
        
        Parameters
        ----------
        x : np.ndarray
            Input token indices of shape (batch_size, seq_len)
            
        Returns
        -------
        np.ndarray
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Store for backward pass
        self.last_input = x
        
        # Embed tokens
        embedded = self.embedding[x]  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        embedded = self.positional_encoding.forward(embedded)
        
        # Apply multi-head attention
        output, attention_weights = self.attention.forward(embedded, embedded, embedded)
        
        # Project to vocabulary
        logits = output @ self.output_projection
        
        # Store for backward pass
        self.last_output = logits
        
        return logits
    
    def predict(self, x):
        """
        Predict method for compatibility with Model interface.
        
        Parameters
        ----------
        x : np.ndarray
            Input token indices
            
        Returns
        -------
        np.ndarray
            Output logits
        """
        return self.forward(x)
    
    def objective(self):
        """
        Compute the objective function value.
        
        This is a placeholder - in practice, you'd compute loss based on targets.
        
        :returns: Objective function value
        :rtype: float
        """
        return 0.0
    
    def fit(self):
        """
        Fit the transformer model.
        
        This is a placeholder for compatibility with the Model interface.
        In practice, you'd implement training logic here.
        
        :raises NotImplementedError: Training not implemented
        """
        raise NotImplementedError("Training logic not implemented. Use as part of a larger training framework.")
