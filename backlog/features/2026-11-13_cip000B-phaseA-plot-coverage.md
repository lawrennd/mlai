---
author: Neil D. Lawrence
category: features
created: "2026-11-13"
id: 2026-11-13_cip000B-phaseA-plot-coverage
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
- plot
title: CIP-000B Phase A — raise plot.py coverage to ≥80%
---

# Task: CIP-000B Phase A — raise plot.py coverage to ≥80%

## Description

Extend `tests/unit/test_plot.py` so `mlai/plot.py` reaches ≥80% line coverage under the default unit suite. Prefer logic and call-shape assertions with mocked matplotlib; avoid pixel-perfect gates. This task absorbs the remaining work from `2025-07-16_plot-py-testing`.

## Acceptance Criteria

- [x] `plot.py` line coverage ≥80% on `uv run pytest tests/unit -m "not slow" --cov=mlai`
- [x] New tests follow existing mock/Agg patterns in `test_plot.py`
- [x] No new display-dependent failures (no required writable `../diagrams`)
- [x] `2025-07-16_plot-py-testing` updated when the 80% gate holds

## Related

- CIP: 000B
- Related: `2025-07-16_plot-py-testing`
- Depends on: Phase 0 baseline inventory

## Progress Updates

### 2026-11-13
`plot.py` reached **81%**. Unskipped/rewrote holdout/LOO/CV/`basis`/`computing_covariance`/`kern_circular_sample` and added `covariance_capacity`. Status → Completed.
