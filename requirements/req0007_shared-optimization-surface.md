---
id: "0007"
title: "Trainable models expose a shared optimization surface"
status: "Proposed"
priority: "Medium"
created: "2026-09-29"
last_updated: "2026-09-29"
related_tenets:
- vibesafe-mlai-tenets
stakeholders:
- students
- educators
- contributors
tags:
- optimization
- models
- teaching
---

# REQ-0007: Trainable models expose a shared optimization surface

## Description

Linear models, neural networks, and other trainable objects in `mlai` can be driven by the same conceptual optimization loop: parameters and gradients are available uniformly, and course-relevant first-order methods can be applied without per-model bespoke training code. Students see one story of “fit by gradients” across model families.

**Why this matters**: Mathematical Transparency and Clarity Over Cleverness — one interface for the shared mathematics of gradient-based learning; Good Python Practices for less duplicated training logic.

**Who benefits**: Students comparing model classes; educators demonstrating optimizers once; contributors adding algorithms without forking every model.

## Acceptance Criteria

- [ ] Trainable model types used in teaching expose parameters and gradients in a consistent way
- [ ] At least the common first-order algorithms needed for courses can run against that shared surface
- [ ] Adding or demonstrating a new optimizer does not require rewriting each model’s training method
- [ ] Teaching material can show the same optimizer on more than one model family

## Notes

Extracted from CIP-0007. Choice of particular optimizers and exact property names are HOW.

## References

- **Related Tenets**: vibesafe-mlai-tenets (mathematical transparency, clarity, good Python practices, educational focus)
- **Implements via**: CIP-0007

## Progress Updates

### 2026-09-29
Requirement extracted from CIP-0007.
