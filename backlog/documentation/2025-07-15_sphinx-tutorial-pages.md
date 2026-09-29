---
category: documentation
created: '2025-07-15'
id: 2025-07-15_sphinx-tutorial-pages
last_updated: '2026-09-29'
owner: Neil Lawrence
priority: Medium
related_cips:
- '0001'
status: Completed
tags:
- backlog
- documentation
- tutorials
- sphinx
title: Add Sphinx tutorial pages aligned with tested workflows
---

# Task: Add Sphinx tutorial pages aligned with tested workflows

## Description

Retrospective record of CIP-0001 tutorial authorship: Sphinx tutorial pages for basis functions, linear regression, logistic regression, perceptron, and related index wiring, kept in sync with tested tutorial code.

## Acceptance Criteria

- [x] Tutorial RST pages exist under `docs/tutorials/`
- [x] Tutorials index lists the main educational workflows
- [x] Tutorial content matches code exercised by integration/workflow tests
- [x] API tutorial stubs (e.g. mountain car, deep GP, GP) linked from docs where applicable

## Implementation Notes

- Primary commit: `da2f251` (2025-07-15)
- Remaining documentation of *how tutorials are tested* is tracked separately in `2025-07-16_tutorial-documentation-integration.md` (CIP-0001 and CIP-0002)

## Related

- CIP: 0001
- Docs: `docs/tutorials/`
- Related open task: `2025-07-16_tutorial-documentation-integration.md`

## Progress Updates

### 2025-07-15
Tutorial pages added and aligned with tested code.

### 2026-09-29
Task created retrospectively as Completed for CIP-0001 backlog coverage.
