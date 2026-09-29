---
category: features
created: '2026-09-29'
id: 2026-09-29_hmc-contour-trajectory-plots
last_updated: '2026-09-29'
owner: Neil D. Lawrence
priority: Medium
related_cips:
- '0008'
status: Proposed
tags:
- backlog
- features
- hmc
- plotting
- teaching
title: Add HMC leapfrog trajectory and trace plot helpers
---

# Task: Add HMC leapfrog trajectory and trace plot helpers

## Description

Extend `mlai.plot` so HMC teaching demos can show leapfrog trajectory segments on the same contour style as `regression_contour*`, plus simple trace plots for $H$ and parameters.

## Acceptance Criteria

- [ ] Contour helper overlays leapfrog path segments for a 2D potential
- [ ] Trace helper plots $H$ and/or parameter components from an HMC run
- [ ] Helpers compose with existing contour / figure-writing conventions
- [ ] Basic unit or smoke tests for the new plot entry points

## Implementation Notes

- Depends on a usable sampler API from `2026-09-29_hmc-core-sampler` (can stub trajectories for early plot work if needed)
- Keep API small; match docstring quality expectations from CIP-0003

## Related

- CIP: 0008
- REQ: 0008
- Depends on: 2026-09-29_hmc-core-sampler (preferred)

## Progress Updates

### 2026-09-29
Task created when CIP-0008 was accepted.
