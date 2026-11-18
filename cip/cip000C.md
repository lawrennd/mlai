---
author: Neil D. Lawrence
created: "2026-11-18"
id: "000C"
last_updated: "2026-11-18"
owner: Neil D. Lawrence
status: Proposed
compressed: false
blocked_by: "0007"
related_requirements:
- "0009"
- "0007"
related_cips:
- "0007"
tags:
- cip
- gaussian-processes
- sparse
- variational
- vfe
- dtc
- lq
- gplvm
- teaching
- numerics
title: Teaching LQ sparse VFE/DTC with gradients (and a minimal GP-LVM path)
---

# CIP-000C: Teaching LQ sparse VFE/DTC with gradients (and a minimal GP-LVM path)

> CIPs describe HOW to achieve requirements (WHAT). This CIP is the HOW for [REQ-0009](../requirements/req0009_teachable-sparse-latent-gp.md). It **depends on** finishing the exact-GP slice of [CIP-0007](cip0007.md) (shared `parameters` / `gradients` for `GP`) before sparse / latent models join the same optimiser loop.

## Status

- [x] Proposed - Initial idea documented
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed (blocked by CIP-0007 GP surface)

## Summary

Add a teachable sparse variational GP to `mlai` whose collapsed VFE/DTCVAR bound uses the **stable LQ factorisation** of stacked whitened features (Seeger et al. BLR trick), with reverse-mode gradients suitable for kernel, noise, inducing inputs \(Z\), and optionally latent inputs \(X\) (minimal GP-LVM). Prefer numerical clarity over GPy/GPc feature breadth. Cross-check against Gram+chol and the GPc CIP-001B prototype; do not port MPI, FITC, or uncertain-input machinery in the first cut.

**Requirements**: REQ-0009 (primary); REQ-0007 (shared surface reused, not redefined).

## Motivation

Exact `mlai.GP` teaches full NLL with an optional Cholesky path, but sparse variational inference and GP-LVM appear in the course narrative without an executable, inspectable core in the package. GPy’s VarDTC uses Gram+chol on \(I+AA^\top\), which squares the condition number of the whitened cross-covariance. GPc CIP-001B already documents and prototypes LQ of \([I,A]\) (optionally \(y\)-augmented) for the same bound. `mlai` needs the pedagogical strip-down of that idea: one factorisation students can follow, gradients on the CIP-0007 surface, and a latent-\(X\) demo that reuses the same objective.

## Detailed Description

### Scope boundary with CIP-0007

| Concern | Owner |
|---------|--------|
| Exact `GP` `parameters` / `gradients` + SGD/Adam on hyperparameters | **CIP-0007** (finish first) |
| Sparse VFE/DTCVAR forward bound (LQ), inducing \(Z\), reverse mode, optional latent \(X\) | **CIP-000C** (this CIP) |

CIP-000C assumes the exact GP already speaks the shared optimiser dialect so sparse / latent models can copy the same property shapes and teaching story.

### Bound (collapsed VFE / DTCVAR)

Whitened features (fixed inducing \(Z\), isotropic noise \(\sigma^2\)):

\[
A = \sigma^{-1} L_m^{-1} K_{mn},\qquad L_m=\mathrm{chol}(K_{mm}).
\]

Titsias / VFE:

\[
\mathcal{F}
= \log\mathcal{N}\bigl(y\,\big|\,0,\sigma^2(I+A^\top A)\bigr)
- \tfrac{1}{2\sigma^2}\operatorname{tr}K_{nn}
+ \tfrac12\|A\|_F^2.
\]

Plain DTC = drop the \(\operatorname{tr}K_{nn}\) correction (teaching toggle, not a second codepath).

### Numerics: LQ instead of Gram+chol (priority)

Do **not** form \(C=I+AA^\top\) then `chol(C)`. Factor the wide matrix whose Gram *is* that quantity:

\[
W=\begin{bmatrix}I_M & A\end{bmatrix}
= L\,Q
\]

(NumPy: QR of \(W^\top\); sign-normalise \(\mathrm{diag}(L)>0\); use \(\log|L_{ii}|\) in the bound). Optionally augment with a \(y\)-row so the Woodbury quadratic \(r^2\) falls out of the same factorisation (GPc CIP-001B §2). Cost \(\mathcal{O}(M^2(M+N))\), same order as Gram+chol, better \(\kappa\).

Ship a Gram+chol reference path only for parity tests on tiny problems, not as the default teaching API.

### Gradients (reverse mode)

Compose reverse mode through the forward graph (do not hand-derive Legacy \(\partial\mathcal{F}/\partial C^{-1}\) coordinates):

1. Seed cotangents from \(\sum_i\log|L_{ii}|\) and \(\tfrac12 r^2\) (plus \(\|A\|_F^2\) / trace terms outside the LQ).
2. **LQ adjoint** (Seeger et al. §4.3) \(\rightarrow\bar W\rightarrow\bar A\).
3. Whitening adjoints: `trsm` then Cholesky of \(K_{mm}\) \(\rightarrow\bar K_{mm},\bar K_{mn},\bar\sigma\).
4. Kernel / \(Z\) / optional \(X\) cotangents via existing `Kernel` evaluations (finite-difference verified first; analytic kernel grads where already natural).

Wire results into CIP-0007’s flat `parameters` / `gradients` vectors so `SGD` / `Adam` / `train_model` work unchanged.

### Minimal GP-LVM path

Unsupervised / latent case: treat rows of latent \(X\) as parameters of the same sparse (or, for tiny \(N\), exact) objective. First teaching demo: 2-D latent, small \(N\), fixed or jointly learned kernel lengthscales; no back-constraints, no ARD jungle, no GPy Bayesian GPLVM feature parity.

### Module layout (proposed)

- New module e.g. `mlai/sparse_gp.py` (or a clearly named class beside `GP` if kept small): construction, bound, predict, `parameters` / `gradients`.
- Reuse `Kernel` for \(K_{mm}\), \(K_{mn}\), \(\mathrm{diag}(K_{nn})\).
- Plots: ELBO / low-rank intuition via `plot.py` or course `teaching_plots` companions; do not block CIP on slide assets.
- Optional: notebook cells in mlatcl that `%load` the sparse class (out of repo or follow-on backlog).

### Explicit non-goals (first cut)

- FITC / PITC / PEP, heteroscedastic noise, uncertain inputs, MPI / minibatch SVI, multi-output, Deep GP layers
- Porting GPy `paramz` or GPc C++ / LAPACK `gelqf` (NumPy QR-of-transpose is enough for teaching)
- Replacing course GPy demos wholesale

## Implementation Plan

1. **Prerequisite**: Complete CIP-0007 exact-GP shared surface; confirm FD checks and one SGD/Adam hyperparameter path.
2. **Forward LQ bound**: port GPc `scripts/lq_sparse_prototype.py` algebra into `mlai` with real kernels; parity vs Gram+chol on random tiny \(N,M\).
3. **LQ + whitening reverse mode**: implement adjoints; FD-check \(\partial\mathcal{F}\) w.r.t. \(\sigma\), kernel params, \(Z\).
4. **Shared surface**: expose flat `parameters` / `gradients`; integration test with `SGD` or `Adam`.
5. **Latent \(X\)**: optional block in the parameter vector; smoke-test GP-LVM-style fit on a tiny synthetic dataset.
6. **Teaching polish**: docstrings with equations; minimal plot or notebook hook; document DTC vs VFE toggle.

## Backward Compatibility

Additive. Existing `GP` and GPy-optional tutorials remain. New imports are opt-in. No change to CIP-0007 optimiser API beyond consuming it.

## Testing Strategy

- Unit: `lq_factor` sign convention; log-det and \(r^2\) vs Gram+chol / dense NLL on well-conditioned toys
- FD: `mlai.utils.verify_gradient_implementation` on parameter blocks
- Integration: optimiser reduces the sparse objective; optional latent-\(X\) objective decreases on a fixed seed
- Stress (optional backlog): crowded \(Z\) — document failure modes; do not claim GPc CIP-001A coverage in v1

## Related Requirements

- [REQ-0009](../requirements/req0009_teachable-sparse-latent-gp.md) — teachable sparse / latent GP outcomes
- [REQ-0007](../requirements/req0007_shared-optimization-surface.md) — shared parameters / gradients loop (prerequisite surface)

## Implementation Status

- [ ] CIP-0007 exact GP surface available (blocker)
- [ ] LQ forward VFE/DTCVAR bound in `mlai`
- [ ] Reverse-mode grads (LQ + whitening) FD-verified
- [ ] Shared optimiser integration for \(\sigma\), kernel, \(Z\)
- [ ] Optional latent \(X\) / minimal GP-LVM demo
- [ ] Tests + teaching docstrings

## References

- GPc [CIP-001B](https://github.com/SheffieldML/GPc/blob/main/cip/cip001B.md) — LQ for collapsed VFE/DTC (production numerics source of truth)
- GPc `scripts/lq_sparse_prototype.py` — NumPy QR/LQ prototype
- Seeger, Hetzel, Dai, Meissner & Lawrence (2017), *Auto-differentiating linear algebra*, arXiv:1710.08717 §3.3, §4.3
- Titsias (2009), variational sparse GP / VFE
- GPy `VarDTC` — Gram+chol reference for ELBO parity only
- mlai backlog: `2026-09-29_gp-shared-optimization-surface` (CIP-0007)
