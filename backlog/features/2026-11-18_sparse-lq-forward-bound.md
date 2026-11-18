---
category: features
created: '2026-11-18'
id: 2026-11-18_sparse-lq-forward-bound
last_updated: '2026-11-18'
owner: Neil D. Lawrence
priority: High
related_cips:
- '000C'
status: Completed
tags:
- backlog
- features
- gaussian-processes
- sparse
- lq
- teaching
title: LQ forward VFE/DTCVAR bound in mlai
---

# Task: LQ forward VFE/DTCVAR bound in mlai

## Description

Port the GPc LQ-of-`[I,A]` collapsed VFE/DTCVAR forward bound into `mlai` with real kernels, and parity-check against Gram+chol on tiny problems.

## Acceptance Criteria

- [x] `lq_factor` with positive diagonal and `W = L Q` reconstruction
- [x] `SparseGP` objective matches Gram+chol reference
- [x] VFE vs DTC toggle

## Related

- CIP: 000C

## Progress Updates

### 2026-11-18
Implemented in `mlai/sparse_gp.py` with unit tests. Marked Completed.
