---
category: features
created: "2026-11-18"
id: 2026-11-18_cip000E-batch-b-sparse-inducing-demo
last_updated: "2026-11-18"
owner: Neil D. Lawrence
priority: High
related_cips:
- "000E"
status: Ready
tags:
- backlog
- features
- gaussian-processes
- sparse
- snippets
- teaching
title: Batch B — migrate inducing-variables demo to SparseGP
dependencies:
- 2026-11-18_cip000E-phase0-plot-mlai-gp
- 2026-11-18_cip000E-batch-a-exact-gp-snippets
---

# Task: Batch B — migrate inducing-variables demo to SparseGP

## Description

Rewrite `_gp/includes/inducing-variables-demo.md` from GPy full + sparse regression to `mlai.GP` / `mlai.SparseGP` with SCG/L-BFGS, including fixed then free inducing inputs and a likelihood/bound comparison.

## Acceptance Criteria

- [ ] Full GP baseline uses `mlai.GP` + SCG/L-BFGS
- [ ] Sparse path uses `SparseGP` with explicit `Z`
- [ ] Demo still shows constrained then unconstrained inducing optimisation
- [ ] Log-likelihood / bound comparison retained in spirit
- [ ] No GPy import in this include

## Implementation Notes

Primary classroom payoff of CIP-000C + CIP-000D. Included from `approximate-gps*.md`.

## Related

- CIP: 000E

## Progress Updates

### 2026-11-18
Task created (Ready) with CIP-000E acceptance.
