---
category: features
created: '2026-11-18'
id: 2026-11-18_scg-lbfgs-teaching-optimisers
last_updated: '2026-11-18'
owner: Neil D. Lawrence
priority: High
related_cips:
- '000D'
- '0007'
- '000C'
status: Completed
tags:
- backlog
- features
- optimization
- scg
- lbfgs
- gaussian-processes
- teaching
title: Teachable SCG and L-BFGS on the shared optimiser surface
---

# Task: Teachable SCG and L-BFGS on the shared optimiser surface

## Description

GP teaching (exact and sparse) should use Scaled Conjugate Gradients and L-BFGS, not Adam/SGD. Implement both on CIP-0007’s `parameters` / `gradients` / `objective` contract before migrating snippets such as `inducing-variables-demo.md`.

## Acceptance Criteria

- [x] `SCG` (Netlab/paramz lineage) with `optimize(model, …)`
- [x] `LBFGS` SciPy wrapper with the same API
- [x] Tests on quadratic bowl, `GP`, and `SparseGP`
- [x] Exported from `mlai`

## Related

- CIP: 000D
- REQ: 0007, 0009

## Progress Updates

### 2026-11-18
Implemented under CIP-000D. Marked Completed.
