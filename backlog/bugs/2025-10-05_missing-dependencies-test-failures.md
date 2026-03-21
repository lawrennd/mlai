---
owner: "Neil D. Lawrence"
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

- Use `unittest.mock.patch.dict('sys.modules', {'GPy': MagicMock()})` to mock GPy at import
  time, allowing GPy-dependent modules to be tested without GPy installed — see
  `tests/unit/test_deepgp_tutorial.py` and `tests/unit/test_mountain_car.py` for the pattern
- Evict cached module entries from `sys.modules` before each patched import to ensure a fresh load
- Use `pytest.skipif` or `pytest.importorskip` only as a fallback when mocking is impractical
- Update documentation to clarify optional vs required dependencies
- Consider making dependency installation optional in CI

## Related

- Core mlai functionality should work without these dependencies
- Transformer implementation is independent of these dependencies
- Educational materials should not require these dependencies
- `backlog/bugs/2026-03-21_mountain-car-stale-mlai-import.md` — GPy mocking pattern first
  applied to `mountain_car` while fixing a stale import bug

## Progress Updates

### 2025-10-05
Task created to address missing dependency test failures.

### 2026-03-21
GPy mocking pattern established in `tests/unit/test_mountain_car.py` (commit `970c4bb`).
`tests/unit/test_deepgp_tutorial.py` already used this approach; `mountain_car` is now the
second module covered. The pattern to follow for remaining GPy-dependent modules:

```python
with patch.dict('sys.modules', {'GPy': MagicMock()}):
    import mlai.<module> as mod
```

Remaining acceptance criteria still open: Daft dependency, CI/CD configuration, and coverage
of other GPy-dependent modules (`gp_tutorial`).
