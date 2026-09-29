---
category: features
created: '2026-09-29'
id: 2026-09-29_gp-shared-optimization-surface
last_updated: '2026-09-29'
owner: Neil D. Lawrence
priority: Medium
related_cips:
- '0007'
status: Proposed
tags:
- backlog
- features
- gaussian-processes
- optimization
- teaching
title: Expose GP parameters and gradients on the shared optimisation surface
---

# Task: Expose GP parameters and gradients on the shared optimisation surface

## Description

CIP-0007’s shared optimiser loop (`Optimiser.step` via `parameters` / `gradients`) already drives linear models and neural networks. Gaussian processes still sit outside that surface: `GP` exposes `objective()` but not the uniform parameter/gradient properties needed to run SGD/Adam (or teaching demos) the same way.

## Acceptance Criteria

- [ ] `GP` (or a clear hyperparameter-facing wrapper) exposes `parameters` and `gradients` consistent with other trainable models
- [ ] At least one kernel / likelihood hyperparameter path can be stepped with `SGD` or `Adam` from `mlai.optimisation`
- [ ] Unit or integration test shows loss/objective improving under the shared optimiser
- [ ] Teaching notes can refer to the same “fit by gradients” story used for linear / NN models

## Implementation Notes

- Prefer additive API; do not break existing GP fitting helpers
- Align shapes with how `SGD`/`Adam` already treat flat parameter vectors
- Finite-difference checks against `objective()` where practical

## Related

- CIP: 0007
- REQ: 0007

## Progress Updates

### 2026-09-29
Task created when CIP-0007 was accepted; residual Phase 1 GP work.
