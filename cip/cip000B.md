---
author: Neil D. Lawrence
created: "2026-11-13"
id: "000B"
last_updated: "2026-11-13"
owner: Neil D. Lawrence
status: In Progress
compressed: false
related_requirements:
- "0002"
related_cips:
- "0002"
- "000A"
tags:
- cip
- testing
- coverage
- pytest
- quality
title: Raise package test coverage for teaching-critical surfaces
---

# CIP-000B: Raise package test coverage for teaching-critical surfaces

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Run a focused coverage campaign on `mlai` so the automated suite exercises the modules that still leave most statements untested. The test *framework* from [CIP-0002](cip0002.md) is already in place; this CIP is the depth campaign: raise measurable line coverage on the large and teaching-facing surfaces, then optionally lock a floor into CI once the numbers are stable.

**Which requirements does this CIP address?** [REQ-0002](../requirements/req0002_automated-verification.md) (coverage / quality signal visible and actionable; core units and teaching workflows covered).

## Motivation

A recent full-package `pytest --cov=mlai` run on the CIP-000A branch reported about **70%** overall line coverage (**3465 / 4948** statements). That is enough to catch many regressions in GPs, networks, and utils, and too thin for confidence when editing the big teaching helpers.

Two facts drive the design:

1. **Statement weight.** `mlai/plot.py` alone is ~5400 lines. At ~64% cover it still dominates the uncovered residual. Raising overall coverage without a plot campaign is arithmetic theatre.
2. **Teaching surfaces with thin tests.** Several smaller modules land well below the package average despite existing `tests/unit/test_*.py` files: `mountain_car` (~14%), `gp_tutorial` (~19%), `data` (~21%), `optimisation` (~23%), `hmc` (~34%), `loss` (~43%). These are the APIs students and notebooks lean on.

CIP-0002 already named this gap (plot depth, optional-deps hygiene). The open backlog item [2025-07-16_plot-py-testing](../backlog/features/2025-07-16_plot-py-testing.md) targets `plot.py` specifically. This CIP sequences that work with the other low modules into one coverage programme with clear phases and a package-level outcome, instead of leaving depth as an open-ended CIP-0002 residual.

## Detailed Description

### Baseline (remeasure at kickoff)

Numbers below are from the CIP-000A worktree unit run (`pytest` with `pytest.ini` coverage defaults). Remeasure on the branch used for implementation (prefer `main` after [CIP-000A](cip000A.md) / PR #7 merges) and record the fresh totals in this CIP's progress notes.

| Module | Approx. cover | Approx. size (LOC) | Priority |
|--------|---------------|--------------------|----------|
| `plot.py` | ~64% | ~5400 | P0 — dominates residual |
| `loss.py` | ~43% | ~370 | P1 |
| `hmc.py` | ~34% | ~300 | P1 — extend existing suite |
| `optimisation.py` | ~23% | ~250 | P1 |
| `data.py` | ~21% | ~290 | P1 |
| `gp_tutorial.py` | ~19% | ~50 | P2 — small, high leverage |
| `mountain_car.py` | ~14% | ~330 | P2 |
| Core (`gaussian_processes`, `neural_networks`, `dimred`, `utils`, …) | ≥90% | — | Hold the line |

Package target for this CIP: **≥85% overall** line coverage under the default `pytest` invocation (`not slow` where CI uses that marker), with **`plot.py` ≥80%**. Stretch: ≥90% overall if plot and teaching modules cooperate without brittle visual tests.

### Design principles

1. **Logic over pixels.** Prefer asserting return values, array shapes, raised errors, and that matplotlib APIs were called with expected arguments. Avoid making pixel-perfect image comparison the default gate (optional `pytest-mpl` only where a teaching figure is the contract).
2. **Mock the display.** Keep Agg / non-interactive backends; write plot outputs under temp dirs. Do not require a writable `../diagrams` tree for unit coverage.
3. **Skip clearly for optionals.** GPy / Daft / JAX paths stay `skip` or soft-fail with a documented reason (REQ-0002). Coverage omit rules stay honest — do not `pragma: no cover` teaching code just to inflate the rate.
4. **Parity with CIP-0002 structure.** New tests live under `tests/unit/` (and `tests/integration/` only when a multi-module workflow is the point). No second framework.
5. **Coverage floor is late.** Publish numbers and HTML reports from the start; add a CI `fail_under` (or equivalent) only after Phase C, once the floor will not flap on Python/matplotlib minor versions.

### Phased campaign

**Phase 0 — Baseline and inventory**

- Remeasure coverage on the implementation branch; paste totals into this CIP.
- Produce an uncovered-line inventory for P0/P1 modules (`coverage report --show-missing` or HTML drill-down).
- Absorb / retarget [2025-07-16_plot-py-testing](../backlog/features/2025-07-16_plot-py-testing.md) under this CIP once Accepted (update `related_cips` on that task).

**Phase A — `plot.py` (≥80%)**

- Extend `tests/unit/test_plot.py` by cluster (GP visuals, neural-net figures, HMC helpers, optimisation surfaces, misc utilities).
- Reuse fixtures; assert side effects via mocks where drawing is incidental.
- Close or narrow the plot backlog item when the 80% gate holds locally and in CI.

**Phase B — Mid modules (loss, hmc, optimisation, data)**

- Grow existing unit files: `test_loss.py`, `test_hmc.py`, `test_optimisation.py`, `test_data.py`.
- Prefer public teaching APIs and error paths over private helpers.
- Keep HMC statistical / slow cases under the `slow` marker so default CI stays fast.

**Phase C — Teaching scripts (`gp_tutorial`, `mountain_car`)**

- Drive the small `gp_tutorial` surface to high cover with mocks.
- Cover mountain-car dynamics and policy helpers without requiring interactive control loops.

**Phase D — Package floor (optional tripwire)**

- If overall ≥85% is stable across Python versions in CI, add a documented coverage floor (e.g. `--cov-fail-under=85` or `.coveragerc` `fail_under`) on the default test job.
- Document how to read `htmlcov/` in the README testing section if anything is still unclear after CIP-0002.

### Explicitly out of scope

- Filling empty `tests/property/`, `tests/docs/`, or performance suites (remain CIP-0002 follow-ons).
- Tightening ruff advisory rules (E501 / C901) — separate from coverage.
- Requiring optional GPy/Daft installs in CI solely to raise cover on import branches.

## Implementation Plan

1. **Accept this CIP** and create backlog tasks per phase (plot campaign first).
2. **Phase 0**: Remeasure baseline after CIP-000A lands; inventory missing lines.
3. **Phase A**: Raise `plot.py` to ≥80%; update plot backlog status.
4. **Phase B**: Raise loss / hmc / optimisation / data toward ≥75% each (or document justified exceptions).
5. **Phase C**: Raise `gp_tutorial` and `mountain_car` toward ≥80% each.
6. **Phase D** (optional): Enable CI coverage floor once green across the matrix.
7. **Verify**: Default `uv run pytest` (CI-equivalent markers) meets package ≥85%; record final table in this CIP; move to Implemented → Closed after review.

## Backward Compatibility

No public API changes are required. Tests and CI config only. A later `fail_under` floor can fail PRs that drop coverage; that is intentional and should be announced in the PR that enables Phase D.

## Testing Strategy

- Local: `uv run pytest` (inherits `pytest.ini` coverage) and `uv run pytest -m "not slow"` to match CI.
- CI: existing test workflow must stay green; coverage XML/HTML artifacts remain available for review.
- Acceptance for this CIP: package line rate ≥85%, `plot.py` ≥80%, no new flaky display-dependent failures, optional-deps still skip cleanly.

## Related Requirements

- [REQ-0002](../requirements/req0002_automated-verification.md) — automated verification; actionable coverage signal; teaching workflows covered.

Related HOW already in flight:

- [CIP-0002](cip0002.md) — pytest framework (this CIP is the depth campaign under that roof).
- [CIP-000A](cip000A.md) — ruff lint cutover (orthogonal; merge first so coverage work sits on the current tool stack).

## Implementation Status

- [x] Draft CIP (Proposed)
- [x] Accept CIP and create phase backlog tasks
- [x] Phase 0 baseline remasurement
- [x] Phase A `plot.py` ≥80%
- [x] Phase B mid modules (already met on baseline; see progress)
- [ ] Phase C teaching scripts (`mountain_car` still below 80%)
- [ ] Phase D coverage floor (optional)
- [x] Package ≥85% verified in CI-equivalent run

## Progress Updates

### 2026-11-13
Accepted and moved to In Progress after CIP-000A (PR #7) merged. Linked CIP-0002 residual plot-coverage item and existing `2025-07-16_plot-py-testing` backlog under this campaign. Phase backlog tasks created; Phase 0 remasurement started on post-ruff `main`.

**Phase 0 baseline** (`uv run pytest tests/unit -m "not slow" --cov=mlai` on post-ruff `main`): **77%** overall; `plot.py` 68%; `gp_tutorial` 19%; `mountain_car` 14%. Mid modules already strong (`loss`/`optimisation` 100%, `hmc` 95%, `data` 92%) — Phase B target effectively pre-met.

**First implementation pass** (same command after new tests): **85%** overall; `plot.py` **81%**; `gp_tutorial` **100%**; `mountain_car` 24%. Changes: always export `gp_tutorial` (no GPy gate); real `gp_tutorial` tests; mountain-car dynamics tests; unskip/rewrite plot helpers (`covariance_capacity`, `holdout_fit`/`loo_fit`/`cv_fit`, `basis`, `computing_covariance`, `kern_circular_sample`).

Remaining for Phase C: `mountain_car` (GPy-heavy controllers/animation) toward ≥80%. Phase D coverage floor still deferred until that settles.

## References

- Backlog: [2025-07-16_plot-py-testing](../backlog/features/2025-07-16_plot-py-testing.md)
- Coverage config: `.coveragerc`, `pytest.ini`
- CIP-000A PR: https://github.com/lawrennd/mlai/pull/7
