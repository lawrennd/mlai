---
author: Neil D. Lawrence
category: features
created: "2026-11-13"
id: 2026-11-13_cip000B-phaseD-coverage-floor
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
title: CIP-000B Phase D — enable CI coverage floor
---

# Task: CIP-000B Phase D — enable CI coverage floor

## Description

Once Phases A–C leave package line coverage stably ≥85%, lock that floor into `.coveragerc` (`fail_under`) so regressions fail the default test job, and document how to read `htmlcov/`.

## Acceptance Criteria

- [x] `.coveragerc` sets `fail_under = 85`
- [x] Default `uv run pytest -m "not slow"` (CI-equivalent) stays green above the floor
- [x] README testing section mentions the floor and `htmlcov/`

## Related

- CIP: 000B

## Progress Updates

### 2026-11-13
Enabled after Phase C raised `mountain_car` to ~99% and package cover to ~88%. Status → Completed.
