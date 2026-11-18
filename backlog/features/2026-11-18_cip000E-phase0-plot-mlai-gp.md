---
category: features
created: "2026-11-18"
id: 2026-11-18_cip000E-phase0-plot-mlai-gp
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
- plot
- teaching
title: Phase 0 — plot helpers accept mlai GP and SparseGP
dependencies: []
---

# Task: Phase 0 — plot helpers accept mlai GP and SparseGP

## Description

Make `mlai.plot.model_output` and `model_sample` work with `mlai.GP` and `mlai.SparseGP` without requiring GPy, while remaining compatible with existing GPy-based snippets.

## Acceptance Criteria

- [ ] Targets read from `model.y` or `model.Y` (normalised to 2D columns)
- [ ] Predict mean/variance 1D returns are reshaped for plotting
- [ ] `gpplot` is used without gating on `GPY_AVAILABLE`
- [ ] Inducing markers still appear when `model.Z` is present
- [ ] Unit tests cover a tiny `GP` and `SparseGP` path (GPy not required)

## Implementation Notes

Blocker for Batch A/B snippet migration. Keep GPy models working for unmigrated includes.

## Related

- CIP: 000E

## Progress Updates

### 2026-11-18
Task created (Ready) with CIP-000E acceptance.
