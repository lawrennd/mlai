#!/usr/bin/env python3
"""
Test the softmax function directly.
"""

import numpy as np
from mlai.transformer import Attention

# Test softmax with simple case
attention = Attention(1)

# Simple 1D case
x = np.array([1.0])
print(f"Input: {x}")

# Manual softmax
exp_x = np.exp(x)
manual_softmax = exp_x / np.sum(exp_x)
print(f"Manual softmax: {manual_softmax}")

# Using the attention's softmax
result = attention._softmax(x, axis=-1)
print(f"Attention softmax: {result}")

# Test with 2D case
x_2d = np.array([[1.0, 2.0]])
print(f"\n2D input: {x_2d}")

manual_2d = np.exp(x_2d) / np.sum(np.exp(x_2d), axis=-1, keepdims=True)
print(f"Manual 2D softmax: {manual_2d}")

result_2d = attention._softmax(x_2d, axis=-1)
print(f"Attention 2D softmax: {result_2d}")

# Test the specific case from attention
attention_scores = np.array([[[-0.96524588]]])
print(f"\nAttention scores: {attention_scores}")

manual_attn = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
print(f"Manual attention weights: {manual_attn}")

result_attn = attention._softmax(attention_scores, axis=-1)
print(f"Attention weights: {result_attn}")

