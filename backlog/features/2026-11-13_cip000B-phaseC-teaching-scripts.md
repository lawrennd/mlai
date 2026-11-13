---
author: Neil D. Lawrence
category: features
created: "2026-11-13"
id: 2026-11-13_cip000B-phaseC-teaching-scripts
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
title: CIP-000B Phase C — raise gp_tutorial and mountain_car coverage
---

# Task: CIP-000B Phase C — raise gp_tutorial and mountain_car coverage

## Description

Drive `gp_tutorial` and `mountain_car` toward ≥80% each with mocked display/control loops so teaching scripts are exercised without interactive sessions.

## Acceptance Criteria

- [x] `gp_tutorial.py` ≥80%
- [x] `mountain_car.py` ≥80%
- [x] Tests use mocks/tempdirs; no interactive control loops required

## Related

- CIP: 000B

## Progress Updates

### 2026-11-13
`gp_tutorial` now always exported and at **100%** cover. `mountain_car` dynamics tests raised cover from 14% → **24%**; remaining surface is GPy controllers/animation. Status → In Progress.

### 2026-11-13 (later)
Expanded `test_mountain_car.py` with mocked env/GPy/display paths; `v_simulation` returns a scalar for NumPy 2 assignment safety. `mountain_car` cover **~99%**. Status → Completed.
