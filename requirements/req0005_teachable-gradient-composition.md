---
id: "0005"
title: "Students can study multi-path gradient structure with teachable models"
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
- gradients
- teaching
- composition
---

# REQ-0005: Students can study multi-path gradient structure with teachable models

## Description

Learners working through chain-rule and compositional-model material can inspect small implementations whose forward and backward behaviour make shared or multi-path gradient structure explicit. The outcome is conceptual understanding of how gradients flow through composed transforms, not a production deep-learning stack.

**Why this matters**: Mathematical Transparency and Educational Focus — prefer illustrative implementations over opaque optimized ones (Clarity Over Cleverness).

**Who benefits**: Students in gradient / architecture units; educators tying notes to runnable library code.

## Acceptance Criteria

- [ ] Teachable compositional models used in course material are available inside `mlai`
- [ ] Forward computation of the composed transforms is readable in the teaching implementation
- [ ] Multi-path or shared-input gradient structure can be demonstrated without a production framework
- [ ] Teaching notes can reference the library components without leaving the `mlai` ecosystem
- [ ] The same pedagogical pattern can cover more than one architecture family over time

## Notes

CIP-0005 is one HOW (simple attention for transformer chain-rule notes). The requirement is the cross-architecture learning outcome.

## References

- **Related Tenets**: vibesafe-mlai-tenets (mathematical transparency, educational focus, clarity)
- **Implements via**: CIP-0005 (initial instance)

## Progress Updates

### 2026-09-29
Requirement extracted from CIP-0005, then generalized away from a single architecture.

### 2026-09-29 (CIP-0005 Implemented)
CIP-0005 accepted and marked Implemented: `AttentionLayer` / `MultiHeadAttentionLayer` / `PositionalEncodingLayer` ship with gradient tests and snippet usage. This requirement stays **Proposed** — the transformer attention work is one HOW instance; broader multi-architecture teachable gradient composition is still open.
