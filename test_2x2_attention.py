#!/usr/bin/env python3
"""
Test attention with 2x2 case where the three-path chain rule should be more apparent.
"""

import numpy as np
from mlai.transformer import Attention
from mlai import finite_difference_gradient, verify_gradient_implementation

# 2x2 case
batch_size, seq_len, d_model = 1, 2, 2
X = np.array([[[1.0, 2.0], [3.0, 4.0]]])
print(f"Input: {X}")

attention = Attention(d_model)
print(f"Weights:")
print(f"W_q: {attention.W_q}")
print(f"W_k: {attention.W_k}")
print(f"W_v: {attention.W_v}")
print(f"W_o: {attention.W_o}")

# Forward pass
output, attn_weights = attention.forward(X, X, X)
print(f"Output: {output}")
print(f"Attention weights: {attn_weights}")

# Test gradient for input
def attention_func(x_flat):
    x_reshaped = x_flat.reshape(batch_size, seq_len, d_model)
    output, _ = attention.forward(x_reshaped, x_reshaped, x_reshaped)
    return np.sum(output)

print("\nTesting input gradients...")
numerical_grad = finite_difference_gradient(attention_func, X.flatten())
print(f"Numerical gradient: {numerical_grad}")

# Analytical gradient
grad_output = np.ones_like(output)
gradients = attention.backward(grad_output, X, X, X, attn_weights)
analytical_grad = gradients['grad_query'].flatten()
print(f"Analytical gradient: {analytical_grad}")

print(f"Difference: {np.abs(numerical_grad - analytical_grad)}")

# Check if gradients are close
is_correct = verify_gradient_implementation(analytical_grad, numerical_grad, rtol=1e-3)
print(f"Gradient verification: {'✓ PASS' if is_correct else '✗ FAIL'}")

# Show the three-path chain rule
print(f"\nThree-path chain rule demonstration:")
print(f"grad_query: {gradients['grad_query']}")
print(f"grad_key: {gradients['grad_key']}")
print(f"grad_value: {gradients['grad_value']}")
total_grad = gradients['grad_query'] + gradients['grad_key'] + gradients['grad_value']
print(f"Total gradient (sum of three paths): {total_grad}")
