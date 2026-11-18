---
author: Neil D. Lawrence
created: "2026-11-18"
id: "000D"
last_updated: "2026-11-18"
owner: Neil D. Lawrence
status: Implemented
compressed: false
related_requirements:
- "0007"
- "0009"
related_cips:
- "0007"
- "000C"
tags:
- cip
- optimization
- scg
- lbfgs
- gaussian-processes
- teaching
title: Teaching SCG and L-BFGS on the shared optimiser surface
---

# CIP-000D: Teaching SCG and L-BFGS on the shared optimiser surface

> CIPs describe HOW to achieve requirements (WHAT). This CIP extends [CIP-0007](cip0007.md)’s shared `parameters` / `gradients` surface with the **full-batch** optimisers that GP teaching historically uses, so exact and sparse GP snippets need not call Adam/SGD or GPy’s internal `optimize`.

## Status

- [x] Proposed
- [x] Accepted
- [x] In Progress
- [x] Implemented
- [ ] Closed

## Summary

Add teachable **Scaled Conjugate Gradients (SCG)** and **L-BFGS** as `Optimiser` subclasses that drive any model exposing `parameters`, `gradients`, and `objective()` — especially `GP` and `SparseGP`. Prefer Netlab/paramz SCG numerics students already meet in SheffieldML tooling; wrap SciPy L-BFGS-B behind the same API for a clear quasi-Newton contrast.

**Requirements**: REQ-0007 (shared surface algorithms); REQ-0009 (sparse/latent GP demos need a GP-appropriate optimiser before snippet migration).

## Motivation

CIP-0007 shipped SGD and Adam so neural nets and linear models share one loop. Exact and sparse GPs are smooth, full-batch, low-dimensional objectives. Course narrative and GPy/GPc practice use **SCG** (and often L-BFGS), not stochastic first-order steps. Snippet migration (`inducing-variables-demo.md` and friends) should replace `m.optimize()` with `mlai` SCG/L-BFGS, not Adam.

## Detailed Description

### API

```python
opt = SCG()          # or LBFGS()
flog = opt.optimize(model, max_iters=100, verbose=False)
```

- `optimize` is the primary teaching entry (matches GPy’s “run until done” story).
- `step(model)` performs one SCG iteration (or one L-BFGS iteration budget) for demos that want a manual loop.
- Models must expose `parameters` / `gradients` / `objective()` (GP, SparseGP, LM, …).

### SCG (HOW)

Port the Netlab / paramz Scaled Conjugate Gradients algorithm (Møller 1993 via Nabney): directional finite-difference curvature, adaptive scale `β`, Polak–Ribière updates, restart every `n_params` successes. Keep the code readable; cite Netlab/`paramz.optimization.scg`.

### L-BFGS (HOW)

`scipy.optimize.minimize(..., method='L-BFGS-B')` with `fun`/`jac` bound to `model.objective` / `model.gradients` via temporary `parameters` assignment. Document SciPy as the linear-algebra backend (same spirit as “LAPACK owns dense LA”): we own the teaching wrapper, not a from-scratch BFGS textbook clone, unless a later CIP wants a pure-NumPy limited-memory sketch.

### Explicit non-goals

- RMSprop / Adagrad / AdamW / schedules (remain optional CIP-0007 backlog)
- Replacing SGD/Adam
- Line-search pedagogy beyond what SCG already embeds

## Implementation Plan

1. Add `SCG` and `LBFGS` to `mlai/optimisation.py`; export from `mlai`.
2. Unit tests: quadratic bowl; GP and SparseGP objective decreases; SCG `flog` finite.
3. Point CIP-0007 optional LBFGS backlog item at this CIP; update REQ-0007 progress note.

## Backward Compatibility

Additive. Existing SGD/Adam callers unchanged.

## Testing Strategy

- Synthetic quadratic with known minimum
- `GP` and `SparseGP` smoke: `optimize` reduces `objective()`
- Gradient path: SCG probes must restore `parameters` correctly after directional FD

## Related Requirements

- [REQ-0007](../requirements/req0007_shared-optimization-surface.md)
- [REQ-0009](../requirements/req0009_teachable-sparse-latent-gp.md)

## Implementation Status

- [x] CIP drafted and accepted for GP teaching path
- [x] `SCG` + `LBFGS` implemented
- [x] Tests (quadratic, GP, SparseGP)
- [x] Exports + backlog / CIP-0007 cross-links

## References

- Møller (1993), Scaled conjugate gradients
- Nabney, Netlab `scg.m`; paramz `paramz.optimization.scg`
- Nocedal & Wright — L-BFGS; SciPy `optimize.minimize` L-BFGS-B
- CIP-0007, CIP-000C; snippets `_gp/includes/inducing-variables-demo.md`
