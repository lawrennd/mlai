---
category: features
created: '2025-10-05'
id: 2025-10-05_teachable-attention-layers
last_updated: '2026-09-29'
owner: Neil Lawrence
priority: Medium
related_cips:
- '0005'
status: Completed
tags:
- backlog
- features
- transformer
- attention
- teaching
- gradients
title: Implement teachable attention layers for transformer chain-rule notes
---

# Task: Implement teachable attention layers for transformer chain-rule notes

## Description

Retrospective record of CIP-0005 delivery: add pedagogical scaled-dot-product attention, multi-head composition, and sinusoidal positional encoding to `mlai.neural_networks`, with explicit backward passes that expose the three-path chain rule used in transformer teaching material.

## Acceptance Criteria

- [x] `AttentionLayer` with clear Q/K/V forward and explicit backward (self / cross / mixed)
- [x] `MultiHeadAttentionLayer` composed from multiple `AttentionLayer` instances
- [x] `PositionalEncodingLayer` with sinusoidal encoding
- [x] Gradient tests including three-path chain-rule coverage
- [x] Components importable from `mlai` for `\loadcode` / teaching snippets
- [x] Snippets `simple-transformer-implementation.md` and `chain-rule-transformer-attention.md` reference the library layers

## Implementation Notes

- Delivered inside `mlai/neural_networks.py` rather than a separate `transformer` module so layers compose with the existing network stack
- Snippets live in the `snippets` repo under `_deepnn/includes/`

## Related

- CIP: 0005
- REQ: 0005
- Tests: `tests/unit/test_neural_networks.py` (`TestAttentionLayerGradients`, `TestMultiHeadAttentionLayerGradients`, `TestPositionalEncodingLayerGradients`)

## Progress Updates

### 2025-10-05
CIP proposed; implementation followed in `mlai`.

### 2026-09-29
Task created retrospectively as Completed when CIP-0005 was accepted and marked Implemented.
