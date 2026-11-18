---
category: features
created: '2026-09-29'
id: 2026-09-29_optimisation-extra-algorithms
last_updated: '2026-11-18'
owner: Neil D. Lawrence
priority: Low
related_cips:
- '0007'
- '000D'
status: Proposed
tags:
- backlog
- features
- optimization
- adam
- teaching
title: Optional extra optimisers and training utilities for mlai.optimisation
---

# Task: Optional extra optimisers and training utilities for mlai.optimisation

## Description

CIP-0007’s course-core surface already includes `SGD` and `Adam` with tests. The original CIP also listed RMSprop, Adagrad, AdamW, LBFGS, learning-rate schedules, early stopping, and gradient clipping. Those are useful but not required to satisfy REQ-0007’s “common first-order algorithms” bar once GP joins the shared surface.

## Acceptance Criteria

- [ ] Decide which extras are worth teaching (document in CIP-0007 or this task)
- [ ] Implement chosen algorithms as `Optimiser` subclasses in `mlai.optimisation`
- [ ] Unit tests for each added optimiser
- [ ] Optional: LR schedule / early stopping / gradient clipping helpers used by `train_model` or callers

## Implementation Notes

- Do not block CIP-0007 Implemented status on this task
- Prefer pedagogical clarity over parity with PyTorch’s full catalogue

## Related

- CIP: 0007
- REQ: 0007

## Progress Updates

### 2026-09-29
Task created when CIP-0007 was accepted; residual Phase 2 optional scope.

### 2026-11-18
SCG and L-BFGS for GP teaching moved to CIP-000D / `2026-11-18_scg-lbfgs-teaching-optimisers` (Completed). This task retains only RMSprop / Adagrad / AdamW / schedules / clipping if still desired.
