#!/usr/bin/env python3
"""
Debug gradient computation step by step.
"""

import numpy as np
from mlai.transformer import Attention

# Simple case
batch_size, seq_len, d_model = 1, 1, 1
X = np.array([[[1.0]]])

attention = Attention(d_model)
print(f"Input: {X}")
print(f"Weights: W_q={attention.W_q}, W_k={attention.W_k}, W_v={attention.W_v}, W_o={attention.W_o}")

# Forward pass
output, attn_weights = attention.forward(X, X, X)
print(f"Output: {output}")
print(f"Attention weights: {attn_weights}")

# Recompute intermediate values
Q = X @ attention.W_q
K = X @ attention.W_k
V = X @ attention.W_v
attention_scores = (Q @ K.transpose(0, 2, 1)) * attention.scale

print(f"\nIntermediate values:")
print(f"Q: {Q}")
print(f"K: {K}")
print(f"V: {V}")
print(f"Attention scores: {attention_scores}")

# Gradient computation step by step
grad_output = np.ones_like(output)
print(f"\nGradient computation:")
print(f"grad_output: {grad_output}")

# Step 1: Gradient through output projection
grad_attention_output = grad_output @ attention.W_o.T
print(f"grad_attention_output: {grad_attention_output}")

# Step 2: Gradient for value matrix
grad_V = grad_attention_output @ attn_weights.transpose(0, 2, 1)
print(f"grad_V: {grad_V}")

# Step 3: Gradient for attention matrix
grad_attention = grad_attention_output @ V.transpose(0, 2, 1)
print(f"grad_attention: {grad_attention}")

# Step 4: Gradient through softmax
grad_sum = np.sum(grad_attention * attn_weights, axis=-1, keepdims=True)
print(f"grad_sum: {grad_sum}")
grad_logits = attn_weights * (grad_attention - grad_sum)
print(f"grad_logits: {grad_logits}")

# Step 5: Gradients for Q and K
grad_Q = grad_logits @ K
grad_K = grad_logits.transpose(0, 2, 1) @ Q
print(f"grad_Q: {grad_Q}")
print(f"grad_K: {grad_K}")

# Scale gradients
grad_Q *= attention.scale
grad_K *= attention.scale
print(f"grad_Q (scaled): {grad_Q}")
print(f"grad_K (scaled): {grad_K}")

# Step 6: Input gradients
grad_query = grad_Q @ attention.W_q.T
grad_key = grad_K @ attention.W_k.T
grad_value = grad_V @ attention.W_v.T
print(f"grad_query: {grad_query}")
print(f"grad_key: {grad_key}")
print(f"grad_value: {grad_value}")

# Total gradient (three-path chain rule)
total_grad = grad_query + grad_key + grad_value
print(f"Total gradient: {total_grad}")

