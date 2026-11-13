---
category: infrastructure
created: "2026-11-13"
id: 2026-11-13_cip0009-uv-packaging-cutover
last_updated: "2026-11-13"
owner: Neil D. Lawrence
priority: High
related_cips:
- "0009"
status: Completed
tags:
- backlog
- infrastructure
- packaging
- uv
- poetry
- ci
title: CIP-0009 Poetry to uv packaging cutover
---

# Task: CIP-0009 Poetry to uv packaging cutover

## Description

Execute the Accepted decisions in [CIP-0009](../../cip/cip0009.md): convert
`pyproject.toml` to PEP 621 with hatchling, commit `uv.lock`, remove
Poetry tooling, cut CI and docs over to uv, and verify pytest plus wheel
contents locally.

## Acceptance Criteria

- [x] `pyproject.toml` uses `[project]`, `[dependency-groups]`, hatchling, and
      renamed extras (`graphical-models`, `mountain-car`, `gp-tutorials`)
- [x] `uv.lock` present; Poetry metadata gone (`poetry.lock` was gitignored)
- [x] Tests, lint, docs, and publish workflows use `astral-sh/setup-uv` and
      `uv sync` / `uv run` / `uv build` / `uv publish`
- [x] README and installation docs lead with `uv sync` and keep pip editable
- [x] `uv sync`, `uv run pytest -m "not slow"`, and `uv build` succeed; wheel
      contains `mlai/` and not `tests/`

## Implementation Notes

- Keep runtime dependencies unpinned in `pyproject.toml`
- Put flake8 in the `dev` group
- Release version check compares tag, `[project].version`, and
  `mlai.__version__`
- Map `secrets.PYPI_API_TOKEN` to `UV_PUBLISH_TOKEN`
- Set `MPLBACKEND=Agg` in test and publish workflows; allow Agg
  `plt.show`/`plt.pause` UserWarnings in the workflow-script stderr check

## Related

- CIP: 0009
- Requirement: 0002

## Progress Updates

### 2026-11-13

Task created after CIP-0009 Accept. Implementation completed on
`cip0009-uv-packaging`: metadata, workflows, docs, lockfile, and local
verification. PR #6 CI green (tests + lint on 3.11/3.12); CIP-0009 Closed.
