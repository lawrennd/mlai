---
id: "0004"
title: "Teaching visualizations regenerate in an open Python toolchain"
status: "Proposed"
priority: "Medium"
created: "2026-09-29"
last_updated: "2026-09-29"
related_tenets:
- vibesafe-mlai-tenets
stakeholders:
- educators
- students
tags:
- visualization
- teaching
- open-science
- reproducibility
---

# REQ-0004: Teaching visualizations regenerate in an open Python toolchain

## Description

Animated and diagrammatic teaching figures used with `mlai` can be produced and refreshed with the same open Python stack that builds the documentation. Educators are not blocked by proprietary plotting toolchains when updating mathematical progressions (contours, geometry of fit, parameter trajectories, and similar sequences) across topics.

**Why this matters**: Open Science and Sharing, Reproducibility, and Educational Focus — teaching assets should stay editable wherever the course runs.

**Who benefits**: Lecturers maintaining notes across modules; students reproducing figures locally.

## Acceptance Criteria

- [ ] Teaching figure sequences used in notes are regenerable with the project's Python plotting stack
- [ ] Outputs integrate with the existing documentation figure workflow
- [ ] Regenerating a sequence does not require a proprietary plotting environment
- [ ] The same workflow can serve more than one teaching topic (not a one-off script per lecture only)

## Notes

CIP-0004 is one HOW (replacing a legacy MATLAB GP optimization animation). The requirement is the open, reusable visualization outcome across teaching material.

## References

- **Related Tenets**: vibesafe-mlai-tenets (open science, reproducibility, educational focus)
- **Implements via**: CIP-0004 (initial instance)

## Progress Updates

### 2026-09-29
Requirement extracted from CIP-0004, then generalized away from a single algorithm/topic.

### 2026-09-29
CIP-0004 accepted and marked Implemented (GP quadratic teaching animation now Python/SVG). REQ-0004 stays Proposed: it still asks for the open-toolchain outcome across teaching topics, not only this GP instance.
