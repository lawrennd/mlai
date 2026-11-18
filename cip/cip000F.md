---
author: Neil D. Lawrence
created: "2026-11-18"
id: "000F"
last_updated: "2026-11-18"
owner: Neil D. Lawrence
status: Proposed
compressed: false
related_requirements:
- "0007"
- "0009"
related_cips:
- "0007"
- "000C"
- "000D"
- "000E"
tags:
- cip
- paramz
- parameters
- transforms
- display
- gaussian-processes
- teaching
title: Paramz-lite named parameters and teaching display
---

# CIP-000F: Paramz-lite named parameters and teaching display

> CIPs describe HOW to achieve requirements (WHAT). This CIP makes the shared optimiser surface from [CIP-0007](cip0007.md) **inspectable in the Paramz/GPy teaching style**: named parameters, simple positivity transforms, and a printable table — without porting Paramz itself. Optimisers from CIP-0007 / [CIP-000D](cip000D.md) keep driving a packed 1D `parameters` / `gradients` vector.

## Status

- [x] Proposed
- [ ] Accepted
- [ ] In Progress
- [ ] Implemented
- [ ] Closed
- [ ] Rejected
- [ ] Deferred

## Summary

Introduce a **teaching-scale Paramz-lite** layer: named parameters with optional `+ve` (log/exp) transforms, hierarchical paths for display (`gp.rbf.lengthscale`), and a text/HTML table (`Name | Value | Constraint`) when students `print` a model. Refactor `GP` and `SparseGP` off hand-rolled log-space packing onto this layer first. Leave SGD/Adam/SCG/L-BFGS APIs unchanged.

**Requirements**: REQ-0007 (shared surface stays one loop; becomes more transparent); REQ-0009 (sparse/latent GP demos gain GPy-like inspectability before snippet migration).

## Motivation

CIP-0007 and CIP-000D give course code a shared fit loop and GP-appropriate optimisers. What students still miss from GPy/paramz is the **parameter story at the REPL**: names, positivity, and a readable table — not a silent float vector. Hand-rolled `log(sigma2)` packing in `GP` already encodes Paramz’s idea (optimise unconstrained, store natural values) but hides it. A small, explicit layer makes the transform and the names part of the teaching surface without pulling in Paramz’s full `Parameterized` / prior / tying machinery.

Aligns with **Mathematical Transparency**, **Clarity Over Cleverness**, and **Educational Focus**.

## Detailed Description

### Design choice: scope (2) — display + transforms

Agreed teaching subset (not a Paramz port):

| Include | Exclude (this CIP) |
|---------|-------------------|
| Named `Param` with natural value | Priors column / prior objects |
| `+ve` transform (log ↔ exp) | Tied parameters |
| Hierarchical path for display | Full `constrain_*` API zoo |
| Table `__repr__` / `_repr_html_` | Freeze/fix beyond a simple flag if needed later |
| Pack/unpack to 1D for existing optimisers | GPy-compatible class hierarchy |
| Gradients in **unconstrained** space via chain rule | MCMC / HMC parameter hooks |

### API sketch

```python
# Natural values with named paths; optimisers still see packed unconstrained vector
print(gp)
# Name                |   Value  |  Constraint
# gp.sigma2           |   1.000  |  +ve
# gp.rbf.variance     |   1.000  |  +ve
# gp.rbf.lengthscale  |   1.000  |  +ve

opt = SCG()
opt.optimize(gp)   # unchanged: model.parameters / model.gradients
```

### HOW (components)

1. **`Param`** — scalar (and small array where needed, e.g. inducing `Z`) holding a natural value, optional transform (`None` or `positive` / log-exp), and a name segment.
2. **`ParamGroup` (or mixin on `Model`)** — ordered collection; builds dotted paths; implements pack (`parameters` getter), unpack (setter), and unconstrained `gradients` aggregation via chain rule when the model supplies natural-space grads (or FD as today).
3. **Display** — `__repr__` / `_repr_html_` table: Name, Value, Constraint (`+ve` or blank). Teaching-first formatting; no Prior/Tied columns in v1.
4. **Migration** — `GP` then `SparseGP` replace manual `np.log` / `np.exp` packing with `Param`/`ParamGroup`. LM/NN may keep flat vectors until a follow-on task if display value is low there.
5. **Optimisers** — no API change; CIP-000D SCG/L-BFGS and CIP-0007 SGD/Adam keep calling `parameters` / `gradients` / `objective()`.

### Explicit non-goals

- Depending on or vendoring `paramz` / GPy
- Prior and tied-parameter columns
- Replacing Netlab-style pack/unpak as the optimiser contract
- Rewriting neural-net training display in this CIP

## Implementation Plan

1. Add a small module (e.g. `mlai/parameters.py`) with `Param`, group/mixin, and display helpers; export what teaching notebooks need.
2. Unit tests: pack/unpack round-trip; `+ve` stays positive under unconstrained steps; `__repr__` contains names and `+ve`.
3. Migrate `GP` onto the layer; keep finite-difference / optimiser tests green.
4. Migrate `SparseGP` (including `Z` as unconstrained array params).
5. Smoke: `SCG`/`LBFGS` still reduce `GP` / `SparseGP` objectives; update docstrings to show `print(model)` in teaching examples.

## Backward Compatibility

- **Optimiser contract preserved**: 1D `parameters` / `gradients` remain the driver interface.
- **Natural attributes** (`gp.sigma2`, `kernel.parameters[...]`) should keep working for existing snippets where practical; prefer adapters over silent renames.
- Additive exports; no removal of SGD/Adam/SCG/LBFGS.

## Testing Strategy

- Round-trip pack/unpack for GP and SparseGP layouts
- Positivity: after random unconstrained updates, natural values remain `> 0`
- Display: fixed toy model → stable table substring checks (text and HTML)
- Regression: existing optimisation unit tests (quadratic, GP, SparseGP) still pass
- Optional: finite-difference check that unconstrained gradients match chain rule through `+ve`

## Related Requirements

- [REQ-0007](../requirements/req0007_shared-optimization-surface.md)
- [REQ-0009](../requirements/req0009_teachable-sparse-latent-gp.md)

## Implementation Status

- [x] CIP drafted (Proposed)
- [ ] Accepted
- [ ] `Param` / group + display helpers
- [ ] `GP` migrated
- [ ] `SparseGP` migrated
- [ ] Optimiser regression tests green
- [ ] Exports / teaching docstring examples

## References

- SheffieldML paramz / GPy parameter printout (`Name | Value | Constraint | Prior | Tied to`)
- CIP-0007 shared surface; CIP-000D SCG/L-BFGS; CIP-000C sparse LQ path
- Netlab `pak` / `unpak` convention (retained as optimiser packing)
