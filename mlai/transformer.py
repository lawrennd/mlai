"""
Transformer components for educational purposes.

This module implements simple transformer components focused on understanding
the mathematical principles, particularly the chain rule in attention mechanisms.
"""

import numpy as np


class Attention:
    """
    Basic scaled dot-product attention mechanism for educational purposes.
    
    This implementation focuses on clarity and understanding of the chain rule
    rather than optimization. The forward and backward passes are designed to
    clearly show how gradients flow through the Q, K, V transformations.
    
    Parameters
    ----------
    d_model : int
        The dimension of the model (input/output dimension)
    dropout : float, optional
        Dropout rate for regularization, by default 0.0 (disabled for educational clarity)
    """
    
    def __init__(self, d_model, dropout=0.0):
        self.d_model = d_model
        self.dropout = dropout
        
        # Initialize weight matrices for Q, K, V transformations
        # Using Xavier initialization for stability
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
        
        # Step 3: Softmax to get attention weights
        attention_weights = self._softmax(attention_scores, axis=-1)
        
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
    
    def backward(self, grad_output, query, key, value, attention_weights):
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
        
        # Gradient through softmax
        grad_sum = np.sum(grad_attention_weights * attention_weights, axis=-1, keepdims=True)
        grad_attention_scores = attention_weights * (grad_attention_weights - grad_sum)
        
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
    """
    
    def __init__(self, d_model, n_heads, dropout=0.0):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Create multiple attention heads
        self.attention_heads = [Attention(self.d_k, dropout) for _ in range(n_heads)]
        
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