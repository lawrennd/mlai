---
author: Neil D. Lawrence
category: features
created: "2026-11-13"
id: 2026-11-13_cip000B-phaseB-mid-modules
last_updated: "2026-11-13"
owner: Neil D. Lawrence
priority: Medium
related_cips:
- "000B"
status: Completed
tags:
- backlog
- features
- testing
- coverage
title: CIP-000B Phase B — raise loss/hmc/optimisation/data coverage
---

# Task: CIP-000B Phase B — raise loss/hmc/optimisation/data coverage

## Description

Grow unit tests for `loss`, `hmc`, `optimisation`, and `data` toward ≥75% each (or document justified exceptions). Prefer public teaching APIs; keep heavy HMC statistical cases under the `slow` marker.

## Acceptance Criteria

- [x] Each of `loss.py`, `hmc.py`, `optimisation.py`, `data.py` ≥75% or exception noted in CIP-000B
- [x] Existing unit files extended rather than new frameworks
- [x] Default CI (`not slow`) stays green and fast

## Related

- CIP: 000B

## Progress Updates

### 2026-11-13
Phase 0 baseline already showed loss/optimisation 100%, hmc 95%, data 92%. No further work required. Status → Completed.
