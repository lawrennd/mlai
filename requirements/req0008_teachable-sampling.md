---
id: "0008"
title: "Students can explore sampling-based inference inside the teaching stack"
status: "Proposed"
priority: "Medium"
created: "2026-09-29"
last_updated: "2026-09-29"
related_tenets:
- vibesafe-mlai-tenets
stakeholders:
- students
- educators
tags:
- inference
- sampling
- bayesian
- teaching
---

# REQ-0008: Students can explore sampling-based inference inside the teaching stack

## Description

After point-estimate fitting (losses, gradients, downhill motion), learners can continue into sampling-based inference using small, readable components in the same ecosystem as existing contour and optimization demos. Production inference libraries remain available elsewhere; this requirement is about a first teachable contact with drawing samples from a target defined by a potential (or equivalent) that connects to familiar training losses.

**Why this matters**: Educational Focus and Mathematical Transparency; Clarity Over Cleverness over research-library machinery.

**Who benefits**: Students in Bayesian / energy-based lecture arcs; educators contrasting point estimates with posterior samples without leaving `mlai`.

## Acceptance Criteria

- [ ] At least one teachable sampling method is available from `mlai` for notebooks and lectures
- [ ] The interface makes the link between a potential (or target density) and familiar loss/energy language clear
- [ ] Classroom-useful diagnostics (e.g. accept rate or energy/trace summaries where relevant) are available
- [ ] At least one demo path connects samples or trajectories to existing contour-style teaching plots
- [ ] Further sampling methods can be added under the same teaching pattern without inventing a separate stack

## Notes

CIP-0008 is one HOW (Neal-style Hamiltonian Monte Carlo). The requirement is the cross-method teaching outcome for sampling-based inference.

## References

- **Related Tenets**: vibesafe-mlai-tenets (educational focus, mathematical transparency, clarity)
- **Implements via**: CIP-0008 (initial instance)

## Progress Updates

### 2026-09-29
Requirement extracted from CIP-0008, then generalized away from a single sampler.

### 2026-09-29 (CIP-0008 Accepted)
CIP-0008 accepted and marked In Progress with backlog for core sampler, contour/trace plots, and teaching/lecture wiring. This requirement stays **Proposed** until at least one teachable sampling method ships in `mlai`.
