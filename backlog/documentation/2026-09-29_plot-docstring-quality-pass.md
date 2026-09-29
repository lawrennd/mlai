---
category: documentation
created: '2026-09-29'
id: 2026-09-29_plot-docstring-quality-pass
last_updated: '2026-09-29'
owner: Neil Lawrence
priority: Medium
related_cips:
- '0003'
status: Proposed
tags:
- backlog
- documentation
- plotting
- docstrings
- sphinx
title: Raise remaining mlai.plot docstrings to Sphinx teaching quality
---

# Task: Raise remaining mlai.plot docstrings to Sphinx teaching quality

## Description

All public callables in `mlai.plot` now have *some* docstring, but many remain too thin for teaching/API use (no `:param:` / Parameters block, one-line stubs). CIP-0003 batches covered a large Sphinx-style set; this task finishes the quality pass and clears the known duplicate `kronecker_illustrate` definition.

## Acceptance Criteria

- [ ] Duplicate `kronecker_illustrate` definitions resolved to a single canonical function
- [ ] Public plot functions/classes that are still prose-only or stub-length gain Sphinx/reST parameter (and return) documentation where they take arguments
- [ ] Autodoc for `mlai.plot` still builds cleanly
- [ ] CIP-0003 remaining-function checklist updated to match reality after the pass

## Implementation Notes

- Prefer small batches with `search_replace` (large-file risk noted in CIP-0003)
- Priority stubs from CIP-0003 residual list include Kronecker helpers, `network`/`layer`, GP chain diagrams, and several difficulty/perceptron helpers
- `direcotry` typo from CIP-0003 appears already fixed; re-check when editing those functions

## Related

- CIP: 0003
- REQ: 0003
- Completed adjacent: `2026-09-29_lagrange-parallel-vectors-stationarity`

## Progress Updates

### 2026-09-29
Task created after CIP-0003 acceptance; audit found 85/85 public functions documented at some level, ~33 still lacking Sphinx-style parameter docs, and two `kronecker_illustrate` definitions still present.
