#!/usr/bin/env python3
"""
Test a very simple attention case to debug the gradient computation.
"""

import numpy as np
from mlai.transformer import Attention
from mlai import finite_difference_gradient, verify_gradient_implementation

# Very simple case: 1x1x1
batch_size, seq_len, d_model = 1, 1, 1
X = np.array([[[1.0]]])  # Simple input
print(f"Input: {X}")

attention = Attention(d_model)
print(f"Weights: W_q={attention.W_q}, W_k={attention.W_k}, W_v={attention.W_v}, W_o={attention.W_o}")

# Forward pass
output, attn_weights = attention.forward(X, X, X)
print(f"Output: {output}")
print(f"Attention weights: {attn_weights}")

# Test input gradient with simple function
def simple_func(x_flat):
    x_reshaped = x_flat.reshape(batch_size, seq_len, d_model)
    output, _ = attention.forward(x_reshaped, x_reshaped, x_reshaped)
    return output[0, 0, 0]  # Just return scalar

print("\nTesting with simple scalar output...")
numerical_grad = finite_difference_gradient(simple_func, X.flatten())
print(f"Numerical gradient: {numerical_grad}")

# Analytical gradient
grad_output = np.ones_like(output)
gradients = attention.backward(grad_output, X, X, X, attn_weights)
analytical_grad = gradients['grad_query'].flatten()
print(f"Analytical gradient: {analytical_grad}")

print(f"Difference: {np.abs(numerical_grad - analytical_grad)}")

# Let's also check the intermediate computations
Q = X @ attention.W_q
K = X @ attention.W_k  
V = X @ attention.W_v
print(f"\nIntermediate values:")
print(f"Q: {Q}")
print(f"K: {K}")
print(f"V: {V}")

# Attention scores
attention_scores = (Q @ K.transpose(0, 2, 1)) * attention.scale
print(f"Attention scores: {attention_scores}")

# Softmax
attention_weights_manual = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
print(f"Manual attention weights: {attention_weights_manual}")
print(f"Computed attention weights: {attn_weights}")

