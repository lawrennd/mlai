---
author: Neil D. Lawrence
created: "2026-11-18"
id: "000E"
last_updated: "2026-11-18"
owner: Neil D. Lawrence
status: Accepted
compressed: false
related_requirements:
- "0007"
- "0009"
related_cips:
- "0007"
- "000C"
- "000D"
tags:
- cip
- gaussian-processes
- snippets
- gpy
- teaching
- migration
- scg
- lbfgs
title: Migrate GP teaching snippets from GPy to mlai
---

# CIP-000E: Migrate GP teaching snippets from GPy to mlai

> CIPs describe HOW to achieve requirements (WHAT). This CIP is the HOW for turning CIP-0007 / CIP-000C / CIP-000D library capability into lecture-facing snippets that no longer need GPy for exact and sparse GP regression demos.

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Migrate selected `_gp/` (and closely related) teaching snippets in the `snippets` repository from `GPy.models.GPRegression` / `SparseGPRegression` to `mlai.GP` / `mlai.SparseGP`, fitting hyperparameters with **SCG or L-BFGS** (`CIP-000D`), not Adam/SGD. Fix `mlai.plot.model_output` / `model_sample` so they accept mlai models before rewriting demos. Leave non-Gaussian likelihoods, Emukit wrappers, Bayesian GPLVM visualisation, and the GPy software tour on GPy.

**Requirements**: REQ-0007 (shared optimiser used in teaching material); REQ-0009 (sparse path demonstrable without GPy for the core lecture story).

## Motivation

CIP-0007, CIP-000C, and CIP-000D put exact GPs, sparse VFE/DTC, and teachable SCG/L-BFGS in `mlai`. Many lecture snippets still call GPy for ordinary regression fits (`gp-fit.md`, `inducing-variables-demo.md`, dataset-specific GP includes). That duplicates the teaching stack, blocks GPy-free classroom paths, and keeps DeepGP intros that `\include` those files on a GPy dependency for the shallow baseline.

Course practice for GP hyperparameters is full-batch SCG / L-BFGS (as in GPy / Netlab), not stochastic first-order methods. Snippet migration must use `SCG().optimize(...)` or `LBFGS().optimize(...)`.

## Detailed Description

### Scope

| In scope | Out of scope (this CIP) |
|----------|-------------------------|
| Exact GP regression demos that only need EQ (or simple) kernels + SCG/L-BFGS | Classification, Student-$t$, Laplace/EP |
| Sparse inducing-variable demo via `SparseGP` | Emukit / GPyOpt / `GPyModelWrapper` |
| Plot helper compatibility for `mlai.GP` / `SparseGP` | Rewriting `gpy-tutorial.md` (keep as GPy tour) |
| Dropping `\installcode{GPy}` / `gpy-software` includes where unused after migration | Full Bayesian GPLVM + GPy visualisers |
| | Multi-output ICM/LMC rewrite (follow-on) |
| | Mauna Loa / airline specialised kernel stacks (follow-on) |

Work spans two repositories: **`mlai`** (plot API + any small helpers) and **`snippets`** (markdown includes). Prefer surgical edits that preserve narrative voice and slide/notes macros.

### Phase 0 — Plot compatibility (`mlai`)

`mlai.plot.model_output` and `model_sample` currently assume GPy conventions and block clean migration:

1. **Targets**: accept `model.y` as well as `model.Y` (2D column layout for multi-output).
2. **Predict shapes**: normalise 1D `SparseGP.predict` returns to the 2D layout the plot helpers expect.
3. **`gpplot`**: do not gate the nice band plot on `GPY_AVAILABLE`; use `mlai.gp_tutorial.gpplot` whenever available.
4. **Inducing markers**: keep `hasattr(model, 'Z')` behaviour for `SparseGP`.

Without Phase 0, migrated snippets either crash or silently degrade when GPy is absent.

### Canonical fit pattern (after migration)

```python
from mlai import Kernel, eq_cov, GP, SCG  # or LBFGS, SparseGP

kernel = Kernel(eq_cov, lengthscale=1.0, variance=1.0)
model = GP(x, yhat, sigma2=1.0, kernel=kernel)
# optional: seed lengthscale / noise via kernel.parameters / model.sigma2
flog = SCG().optimize(model, max_iters=100)
yt_mean, yt_var = model.predict(xt)
```

Sparse demo adds `Z` and `SparseGP(..., approximation='vfe')`, with the same optimiser API. Prefer **SCG** as the default teaching optimiser; mention L-BFGS where a quasi-Newton contrast is useful.

### Migration batches (snippets)

#### Batch A — Exact GP regression (clear wins)

Replace `GPy.models.GPRegression` + `m.optimize()` with `mlai.GP` + `SCG`/`LBFGS`:

| Snippet | Notes |
|---------|--------|
| `_gp/includes/gp-fit.md` | Olympic marathon path; also via `olympic-marathon-gp.md` |
| `_gp/includes/step-function-gp.md` | Included by DeepGP step-function intro |
| `_gp/includes/della-gatta-gene-gp.md` | Lengthscale / noise initialisation → `kernel.parameters` / `sigma2` |
| `_gp/includes/motorcycle-helmet-gp.md` | DeepGP motorcycle baseline |
| `_gp/includes/robot-wireless-gp.md` | DeepGP robot baseline; multi-dim `y` careful with plot `output_dim` |
| `_gp/includes/erich-friedman-packing-gp.md` | Drop `\installcode{GPy}` if unused elsewhere in the talk path |

#### Batch B — Sparse inducing demo

| Snippet | Notes |
|---------|--------|
| `_gp/includes/inducing-variables-demo.md` | Full GP then `SparseGP`; fix inducing locations; compare bounds / log likelihoods; included from `approximate-gps*.md` |

This is the primary classroom payoff of CIP-000C + CIP-000D.

#### Batch C — Explicit non-goals (remain on GPy for now)

| Snippet / area | Reason |
|----------------|--------|
| `_gp/includes/non-gaussian-gps.md` | Classification / EP not in `mlai` |
| `_gp/includes/olympic-marathon-gp-student-t.md` | Non-Gaussian likelihood + Laplace |
| `_gp/includes/gpy-tutorial.md` | Product tour of GPy |
| `_gp/includes/other-gp-software.md` | Historical / ecosystem narrative |
| Emukit / UQ includes using `GPyModelWrapper` | External interface contract |
| `_gplvm/includes/*` Bayesian GPLVM + GPy visualise | Beyond minimal `latent_X` |
| `_deepgp/includes/*` kernel lists for DeepGP package | Still need GPy kern objects for that stack |

#### Follow-on (not this CIP unless accepted as stretch)

| Snippet | Dependency |
|---------|------------|
| `_gp/includes/olympic-sprint-gp.md` | Teach ICM/LMC via `icm_cov` / `lmc_cov` with a clear API story |
| `_gp/includes/mauna-loa-gp.md`, `box-jenkins-airline-gp.md` | Periodic / ratquad / bias sums already partly in `mlai` kernels — careful reconstruction |
| `_gp/includes/gpy-emulation.md` | Emulation narrative; lower priority |

### Tenet balance

- **Compose, don't monolith**: snippets keep calling focused `mlai` pieces (`Kernel`, `GP`, `SparseGP`, `SCG`) rather than a new mega-helper.
- **Explicit over implicit**: show optimiser construction and `optimize(...)` rather than hiding fit behind a GPy-like `model.optimize()` shim unless a thin alias proves necessary for readability.
- **Single source, multiple contexts**: migrated includes remain shared by talks/notes/DeepGP intros.

## Implementation Plan

1. **Accept this CIP** after review of batches and non-goals.
2. **Phase 0 (`mlai`)**: fix `model_output` / `model_sample`; add/extend unit tests with a tiny `GP` and `SparseGP` (no GPy).
3. **Batch A (`snippets`)**: migrate exact GP includes; remove obsolete GPy installs/imports; keep figure filenames stable where possible.
4. **Batch B (`snippets`)**: migrate `inducing-variables-demo.md` to `SparseGP` + SCG/L-BFGS.
5. **Smoke-check**: rebuild or execute representative notebook/talk paths that `\include` the migrated files (at least olympic marathon / approximate GPs / one DeepGP baseline include).
6. **Docs compression (after close)**: short note in `mlai` formal docs that lecture GP fits use `SCG`/`LBFGS` and point at the snippet patterns — mark `compressed: true` when done.

## Backward Compatibility

- **Library**: Phase 0 should remain compatible with GPy models (`model.Y`, existing predict shapes) so unmigrated snippets keep working.
- **Snippets**: talks that still assume GPy in the same notebook may need import cleanup; DeepGP paths that only used GPy for the shallow baseline become GPy-light until the deep model section.
- **No GPy API shim** inside `mlai` is required; teaching code switches explicitly to `mlai` names.

## Testing Strategy

- **mlai**: unit tests for plot helpers with `GP` and `SparseGP`; existing SCG/L-BFGS and sparse GP tests remain green.
- **Snippets**: manual or scripted execution of migrated code cells where the course build allows; visual check that SVG outputs still regenerate for key figures.
- **Regression**: at least one path run *without* GPy installed to confirm Batch A/B no longer import it.

## Related Requirements

- [REQ-0007](../requirements/req0007_shared-optimization-surface.md) — teaching material uses the shared optimiser surface (SCG/L-BFGS for GPs).
- [REQ-0009](../requirements/req0009_teachable-sparse-latent-gp.md) — sparse lecture path without GPy for the core demo.

## Implementation Status

- [ ] Phase 0: `model_output` / `model_sample` accept `mlai.GP` and `SparseGP`
- [ ] Batch A: exact GP regression snippets migrated
- [ ] Batch B: `inducing-variables-demo.md` migrated
- [ ] Smoke-check representative includes / builds
- [ ] (Stretch) Batch follow-on only if explicitly accepted

## References

- CIP-0007 (shared surface), CIP-000C (SparseGP), CIP-000D (SCG / L-BFGS)
- Snippets inventory (2026-11-18): `_gp/includes/gp-fit.md`, `step-function-gp.md`, `della-gatta-gene-gp.md`, `motorcycle-helmet-gp.md`, `robot-wireless-gp.md`, `erich-friedman-packing-gp.md`, `inducing-variables-demo.md`
- PR #11 — CIP-000D merge enabling this work
