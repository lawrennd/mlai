---
author: Neil D. Lawrence
created: "2026-11-13"
id: "000A"
last_updated: "2026-11-13"
owner: Neil D. Lawrence
status: Implemented
compressed: false
related_requirements:
- "0002"
related_cips:
- "0002"
- "0009"
tags:
- cip
- lint
- ruff
- flake8
- ci
- infrastructure
title: Replace flake8 with ruff for linting
---

# CIP-000A: Replace flake8 with ruff for linting

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Succeed flake8 with [ruff](https://docs.astral.sh/ruff/) as the lint tool for `mlai` contributor installs and CI. Keep the current two-pass policy (hard-fail on selected syntax/undefined-name classes; advisory pass for complexity and line length) as the first cutover. Optional `ruff format` is out of scope for this CIP.

This CIP depends on [CIP-0009](cip0009.md) (Poetry → uv). Implement after that cutover is merged so the lint job and `dev` group already use uv.

**Which requirements does this CIP address?** [REQ-0002](../requirements/req0002_automated-verification.md) (automated verification still includes a CI lint gate).

## Motivation

After CIP-0009, flake8 lives in the `dev` dependency group and `lint.yml` runs `uv run flake8` twice: a hard fail on `E9,F63,F7,F82`, then an `exit-zero` statistics pass with `max-complexity=10` and `max-line-length=127`. That policy is sound; the engine is the weak part.

Ruff implements the flake8 rule surface we use, is one Astral-stack tool alongside uv, and removes a separate flake8 install path. Contributors get faster local lint and a single config home in `pyproject.toml`.

Keeping this as its own CIP avoids mixing a lint-engine decision into the packaging cutover PR.

## Detailed Description

### What stays

| Topic | Policy |
|-------|--------|
| Lint paths | `mlai/` (same as today) |
| Hard-fail select | Syntax errors and selected pyflakes classes equivalent to `E9,F63,F7,F82` |
| Advisory pass | Complexity and line-length statistics; non-blocking (`exit-zero` today) |
| CI matrix | Python 3.11 and 3.12 on the Lint workflow |
| Contributor bootstrap | `uv sync` then `uv run …` (post CIP-0009) |

### What changes

| Area | Today (flake8) | Target (ruff) |
|------|----------------|---------------|
| Dev dependency | `flake8>=7` in `[dependency-groups] dev` | `ruff` in `dev`; remove flake8 |
| CI command | Two `uv run flake8 …` lines | Two `uv run ruff check …` lines (or one check + one stats-oriented invocation) that preserve hard-fail vs advisory behaviour |
| Config | CLI flags only | Prefer `[tool.ruff]` / `[tool.ruff.lint]` in `pyproject.toml` for line length, select, and complexity so local and CI match |
| Docs | Mentions of flake8 (if any) | Point to `uv run ruff check mlai` |

### Accept decisions (proposed)

1. **Engine:** ruff only; drop flake8 from the lock.
2. **Parity first:** Match current CI semantics before tightening rules.
3. **Config in `pyproject.toml`:** line length 127; hard-fail select aligned with `E9,F63,F7,F82`; complexity limit 10 for the advisory pass.
4. **Format:** do not enable `ruff format` or CI format checks in this CIP.
5. **Scope:** `mlai/` package only (not `tests/` unless Accept revises that).

### Files this cutover touches

- `pyproject.toml` (`dev` group + `[tool.ruff*]`)
- `uv.lock`
- `.github/workflows/lint.yml`
- README / contributor notes if they name flake8
- This CIP and a backlog task once Accepted

## Implementation Plan

1. **Accept** the decisions above (or revise select / path / format scope).
2. **Config.** Add `[tool.ruff]` settings that encode the Accept decisions.
3. **Dependencies.** Replace flake8 with ruff in `dev`; `uv lock`.
4. **CI.** Point `lint.yml` at `uv run ruff check` with hard-fail and advisory passes preserved.
5. **Docs.** Update any flake8 install or run instructions.
6. **Verify.** Local `uv run ruff check mlai` matches CI; Lint workflow green on the PR.

## Backward Compatibility

Contributors who run `flake8` directly switch to `uv run ruff check mlai`. CI behaviour for the hard-fail select should stay equivalent. Broader ruff defaults must not be enabled accidentally in a way that fails the advisory pass if that pass remains `exit-zero`.

No effect on the published `mlai` wheel or on students using `pip install mlai`.

## Testing Strategy

- `uv sync` installs ruff; flake8 is absent from the lock.
- `uv run ruff check mlai` with the hard-fail select exits non-zero only on real hits in that select.
- Lint workflow green on 3.11 and 3.12.
- Spot-check that a deliberate undefined name still fails the hard pass.

## Related Requirements

- [REQ-0002](../requirements/req0002_automated-verification.md): CI still runs an automated quality gate on proposed changes.
- [CIP-0009](cip0009.md): uv packaging and CI bootstrap (prerequisite).
- [CIP-0002](cip0002.md): test/CI quality culture this lint gate supports.

## Implementation Status

- [x] Draft CIP-000A (Proposed)
- [x] Accept decisions reviewed
- [x] Backlog task created (after Accept)
- [x] `pyproject.toml` + `uv.lock` cut over
- [x] `lint.yml` cut over
- [x] Docs updated
- [x] Drop stale `__all__` names that ruff F822 reports and flake8 7 does not, so the hard-fail gate stays green
- [ ] CI Lint green on the cutover pull request

## References

- Current lint job: `.github/workflows/lint.yml` (flake8 two-pass; post-0009: `uv run flake8`)
- Prerequisite: [CIP-0009](cip0009.md) / https://github.com/lawrennd/mlai/pull/6
- Ruff: https://docs.astral.sh/ruff/
- Ruff flake8 compatibility: https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8
