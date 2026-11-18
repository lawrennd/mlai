---
category: features
created: "2026-11-18"
id: 2026-11-18_cip000E-batch-a-exact-gp-snippets
last_updated: "2026-11-18"
owner: Neil D. Lawrence
priority: High
related_cips:
- "000E"
status: Completed
tags:
- backlog
- features
- gaussian-processes
- snippets
- gpy
- teaching
title: Batch A — migrate exact GP regression snippets to mlai
dependencies:
- 2026-11-18_cip000E-phase0-plot-mlai-gp
---

# Task: Batch A — migrate exact GP regression snippets to mlai

## Description

Replace `GPy.models.GPRegression` + `optimize()` in exact GP teaching includes with `mlai.GP` + `SCG` (or `LBFGS`). Drop unused GPy install/import lines. Preserve figure filenames where possible.

## Acceptance Criteria

- [x] `_gp/includes/gp-fit.md` uses `mlai.GP` + SCG/L-BFGS
- [x] `_gp/includes/step-function-gp.md` migrated
- [x] `_gp/includes/della-gatta-gene-gp.md` migrated (lengthscale/noise init via mlai)
- [x] `_gp/includes/motorcycle-helmet-gp.md` migrated
- [x] `_gp/includes/robot-wireless-gp.md` migrated
- [x] `_gp/includes/erich-friedman-packing-gp.md` migrated
- [x] Migrated paths do not import GPy

## Implementation Notes

Edits land in the `snippets` repository. Depends on Phase 0 plot compatibility. Default optimiser: SCG.

Note: `plot.multiple_optima` (still called from della-gatta) remains GPy-backed inside `mlai.plot` for now; the three gene-expression fits themselves no longer use GPy.

## Related

- CIP: 000E
- Snippets branch: `cip000E-migrate-gp-snippets`

## Progress Updates

### 2026-11-18
Task created (Ready) with CIP-000E acceptance.

### 2026-11-18
Batch A snippets migrated on snippets branch; marked Completed.
