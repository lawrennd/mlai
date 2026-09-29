---
category: features
created: '2026-09-29'
id: 2026-09-29_hmc-teaching-example
last_updated: '2026-09-29'
owner: Neil D. Lawrence
priority: Medium
related_cips:
- '0008'
status: Completed
tags:
- backlog
- features
- hmc
- teaching
- snippets
title: Wire HMC teaching example and lecture placeholders
---

# Task: Wire HMC teaching example and lecture placeholders

## Description

Add at least one logistic / small-MLP HMC vs SGD teaching path using existing `mlai` models where possible, and replace lecture placeholders (e.g. `_ml/includes/from-loss-to-hamiltonian.md`) with live `mlai` imports once the sampler exists.

## Acceptance Criteria

- [x] Example shows SGD point estimate vs HMC samples / posterior predictive on a small problem
- [x] Commentary links $V=-\log p(q\mid\mathcal{D})$ to lecture energy language
- [x] Snippet or notebook imports the real `mlai` HMC API (no placeholder-only stubs)
- [x] Example is reproducible with a fixed seed

## Implementation Notes

- Snippet updated in `snippets` repo: `_ml/includes/from-loss-to-hamiltonian.md`
- Quadratic trajectory demo + logistic SGD vs HMC traces
- Library regression: `TestHMCTeachingExample` in `tests/unit/test_hmc.py`

## Related

- CIP: 0008
- REQ: 0008
- Depends on: 2026-09-29_hmc-core-sampler
- Snippet: `_ml/includes/from-loss-to-hamiltonian.md`

## Progress Updates

### 2026-09-29
Task created when CIP-0008 was accepted.

### 2026-09-29
Completed: lecture placeholder replaced; logistic teaching test added.
