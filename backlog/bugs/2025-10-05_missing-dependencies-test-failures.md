---
author: "AI Assistant"
created: "2025-10-05"
id: "2025-10-05_missing-dependencies-test-failures"
last_updated: "2025-10-05"
status: proposed
tags:
- backlog
- bugs
- testing
- dependencies
- gpy
- daft
title: "Fix test failures due to missing optional dependencies (GPy, Daft)"
---

# Task: Fix test failures due to missing optional dependencies

## Description

Several tests are failing because optional dependencies (GPy, Daft) are not available in the current environment. These dependencies are required for certain advanced functionality but should not block core functionality testing.

## Current Status

- **GPy**: Not available - causes Gaussian Process related test failures
- **Daft**: Not available - causes probabilistic graphical model test failures
- **Impact**: ~15-20 test failures related to missing dependencies

## Acceptance Criteria

- [ ] Tests should gracefully skip when optional dependencies are missing
- [ ] Core functionality tests should pass regardless of optional dependencies
- [ ] Clear documentation of which features require which dependencies
- [ ] CI/CD should handle missing optional dependencies appropriately

## Implementation Notes

- Use `pytest.skipif` decorators for tests requiring optional dependencies
- Add dependency checks in test setup
- Consider making dependency installation optional in CI
- Update documentation to clarify optional vs required dependencies

## Related

- Core mlai functionality should work without these dependencies
- Transformer implementation is independent of these dependencies
- Educational materials should not require these dependencies

## Progress Updates

### 2025-10-05
Task created to address missing dependency test failures.
