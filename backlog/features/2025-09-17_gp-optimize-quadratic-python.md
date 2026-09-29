---
category: features
created: '2025-09-17'
id: 2025-09-17_gp-optimize-quadratic-python
last_updated: '2026-09-29'
owner: Neil Lawrence
priority: Medium
related_cips:
- '0004'
status: Completed
tags:
- backlog
- features
- plotting
- gaussian-processes
- teaching
title: Implement gp_optimize_quadratic Python teaching animation
---

# Task: Implement gp_optimize_quadratic Python teaching animation

## Description

Retrospective record of CIP-0004 delivery: replace the MATLAB GP data-fit quadratic animation with `mlai.plot.gp_optimize_quadratic`, SVG frame outputs compatible with the docs pipeline, and unit tests. Teaching snippet `_gp/includes/gp-optimize-data-fit.md` now calls the Python plot helper.

## Acceptance Criteria

- [x] `gp_optimize_quadratic` available in `mlai.plot`
- [x] Elliptical contours, eigenvalue axes, data-point and rotation frames supported
- [x] SVG naming compatible with `gp-optimise-quadratic{NNN}.svg` docs usage
- [x] Unit tests cover basic, custom, no-frames, directory creation, and mathematical checks
- [x] Snippet uses `\plotcode{plot.gp_optimize_quadratic(...)}` rather than MATLAB

## Implementation Notes

- Primary commit: `37bf566` — Implement `gp_optimize_quadratic()` with comprehensive tests
- Snippet lives in the `snippets` repo (`_gp/includes/gp-optimize-data-fit.md`)

## Related

- CIP: 0004
- REQ: 0004
- Tests: `tests/unit/test_plot.py` (`TestGpOptimizeQuadratic`)

## Progress Updates

### 2025-09-17
CIP proposed; implementation followed in `mlai`.

### 2026-09-29
Task created retrospectively as Completed when CIP-0004 was accepted and marked Implemented.
