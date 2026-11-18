---
category: features
created: '2026-11-18'
id: 2026-11-18_sparse-lq-grads-optimiser
last_updated: '2026-11-18'
owner: Neil D. Lawrence
priority: High
related_cips:
- '000C'
status: Completed
tags:
- backlog
- features
- gaussian-processes
- sparse
- gradients
- optimisation
title: SparseGP reverse-mode grads and shared optimiser surface
---

# Task: SparseGP reverse-mode grads and shared optimiser surface

## Description

Wire LQ + whitening cotangents into flat `parameters` / `gradients` (noise, kernel, `Z`, optional latent `X`) and show SGD/Adam reducing the bound.

## Acceptance Criteria

- [x] FD agreement on noise + kernel log-parameters
- [x] Adam reduces objective
- [x] Optional `latent_X` in the parameter vector

## Related

- CIP: 000C

## Progress Updates

### 2026-11-18
Implemented; tests in `tests/unit/test_sparse_gp.py`. Marked Completed.
