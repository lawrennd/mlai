---
category: bugs
created: '2026-09-29'
id: 2026-09-29_lagrange-parallel-vectors-stationarity
last_updated: '2026-09-29'
owner: Neil Lawrence
priority: Medium
related_cips:
- '0003'
status: Completed
tags:
- backlog
- bugs
- plotting
- lagrange
- teaching
title: Fix lagrange_parallel_vectors stationarity and frame animation
---

# Task: Fix lagrange_parallel_vectors stationarity and frame animation

## Description

Correct `mlai.plot.lagrange_parallel_vectors` so Stage 3 satisfies $\nabla f + \lambda\nabla g = 0$, and tidy the animation path (axis limits from all key vectors, morph frames that skip hold endpoints, clearer residual labelling).

Follow-on to commit `10ec266` (Lagrange multiplier vector plots).

## Acceptance Criteria

- [x] Default `grad_g_final` is $-\nabla f / \lambda$ (stationarity)
- [x] Reject $\lambda = 0$ with a clear error
- [x] Axis limits cover $\nabla f$, $\nabla g$, $\lambda\nabla g$, and residuals
- [x] Interpolation frames do not duplicate hold endpoints
- [x] Stage-3 zero residual shown without a separate `show_zero` flag

## Implementation Notes

- Changes are in `mlai/plot.py` (`lagrange_parallel_vectors`)
- Tallied under CIP-0003 as plot-module maintenance adjacent to ongoing `plot.py` docstring/API work

## Related

- CIP: 0003
- Prior commit: `10ec266` — Update with langrange multiplier vector plots

## Progress Updates

### 2026-09-29
Fix completed and recorded against CIP-0003.
