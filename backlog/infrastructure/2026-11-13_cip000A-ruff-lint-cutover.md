---
category: infrastructure
created: "2026-11-13"
id: 2026-11-13_cip000A-ruff-lint-cutover
last_updated: "2026-11-13"
owner: Neil D. Lawrence
priority: High
related_cips:
- "000A"
status: Completed
tags:
- backlog
- infrastructure
- lint
- ruff
- ci
title: CIP-000A flake8 to ruff lint cutover
---

# Task: CIP-000A flake8 to ruff lint cutover

## Description

Execute the Accepted decisions in [CIP-000A](../../cip/cip000A.md): replace
flake8 with ruff in the dev group, encode the hard-fail select, line length,
and complexity limit in `pyproject.toml`, and point the Lint workflow at
`ruff check` without enabling the formatter.

## Acceptance Criteria

- [x] `ruff` is in the `dev` group and flake8 is absent from `uv.lock`
- [x] `[tool.ruff]` sets line length 127; the default select is `E9,F63,F7,F82`
- [x] Advisory CI pass selects `E,F,W,C90,E501` with McCabe complexity 10 and
      exits zero
- [x] README points contributors at `uv run ruff check mlai`
- [x] Local hard-fail `ruff check` is clean, and a deliberate undefined name
      still fails that select

## Implementation Notes

- Do not enable `ruff format` or a format check in CI
- Lint path stays `mlai` (not `tests/`)
- Default `ruff check` is the hard-fail gate; the advisory pass overrides
  `--select` on the command line so ruff's broader defaults stay off
- Ruff F822 flags three `__all__` names flake8 7 does not
  (`SimpleDropoutNeuralNetwork`, `NonparametricDropoutNeuralNetwork` live in
  `experimental`; `radial_multivariate` lives in `linear_models`). Those
  entries were removed so the hard-fail gate stays green, matching current CI

## Related

- CIP: 000A
- Requirement: 0002
- Prerequisite CIP: 0009

## Progress Updates

### 2026-11-13

Task created with the CIP-000A implementation on `cip000A-ruff-lint`. Local
`uv lock`, hard-fail check, advisory `--exit-zero` pass, and an undefined-name
probe are done. Awaiting Lint workflow on the pull request before CIP close.

Follow-up on the same branch: advisory whitespace, unused names, shadowed
definitions, and the small syntax rules in the advisory select are cleared.
E501 and C901 remain, and the advisory CI pass still exits zero.
