---
author: Neil D. Lawrence
category: bugs
created: '2025-10-05'
id: 2025-10-05_plotting-test-failures
last_updated: '2026-09-29'
owner: Neil D. Lawrence
priority: Medium
related_cips:
- '0002'
status: Proposed
tags:
- backlog
- bugs
- testing
- plotting
- matplotlib
title: Fix remaining plotting test failures
---

# Task: Fix remaining plotting test failures

## Description

Several plotting tests are still failing after fixing the perceptron plotting issues. These failures appear to be related to plotting functionality that may have dependency issues or configuration problems.

## Current Status

- **Perceptron plotting tests**: ✅ Fixed (3/3 passing)
- **Other plotting tests**: ❌ Still failing (~7-10 tests)
- **Examples of failing tests**:
  - `test_vertical_chain`
  - `test_horizontal_chain` 
  - `test_init_regression`
  - `test_regression_contour_fit`
  - `test_basis`

## Acceptance Criteria

- [ ] All plotting tests should pass or be properly skipped
- [ ] Plotting functionality should work for educational purposes
- [ ] Clear error messages for plotting failures
- [ ] Documentation of plotting requirements

## Implementation Notes

- Investigate specific plotting test failures
- Check for missing plotting dependencies (matplotlib, seaborn, etc.)
- Consider mocking plotting functions in tests where appropriate
- Ensure plotting works for core educational functionality

## Related

- CIP: 0002 (Comprehensive Test Framework)
- Perceptron plotting tests were successfully fixed
- Core mlai functionality is working
- Transformer implementation is independent of plotting issues

## Progress Updates

### 2025-10-05
Task created to address remaining plotting test failures after fixing perceptron plotting issues.

### 2026-09-29
Linked to CIP-0002 via `related_cips` after CIP-0002 acceptance.