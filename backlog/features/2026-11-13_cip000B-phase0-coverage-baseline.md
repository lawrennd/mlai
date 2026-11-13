---
author: Neil D. Lawrence
category: features
created: "2026-11-13"
id: 2026-11-13_cip000B-phase0-coverage-baseline
last_updated: "2026-11-13"
owner: Neil D. Lawrence
priority: High
related_cips:
- "000B"
status: Completed
tags:
- backlog
- features
- testing
- coverage
title: CIP-000B Phase 0 — remasure coverage baseline and inventory gaps
---

# Task: CIP-000B Phase 0 — remasure coverage baseline and inventory gaps

## Description

Remeasure package coverage on post-CIP-000A `main` with the CI-equivalent unit suite (`pytest -m "not slow"`), record totals in CIP-000B, and produce an uncovered-line inventory for P0/P1 modules (`plot`, `loss`, `hmc`, `optimisation`, `data`).

## Acceptance Criteria

- [x] Fresh `coverage.xml` / term report from `uv run pytest tests/unit -m "not slow" --cov=mlai`
- [x] Package and per-module rates recorded in CIP-000B progress notes
- [x] Short inventory of largest uncovered regions in `plot.py` (function-level)
- [x] Existing plot backlog retargeted to CIP-000B

## Related

- CIP: 000B
- Depends on: CIP-000A merged (PR #7)

## Progress Updates

### 2026-11-13
Baseline: 77% overall; plot 68%. Inventory pointed at `covariance_capacity`, holdout/LOO/CV fit, `basis`, `computing_covariance`, multi-output helpers, `mountain_car`. Status → Completed.
