---
id: "2025-10-12_coverage-collection-numpy-interference"
title: "Fix Coverage Collection Interference with NumPy"
status: "Proposed"
priority: "Medium"
created: "2025-10-12"
last_updated: "2025-10-12"
owner: "Neil D. Lawrence"
github_issue: ""
dependencies: ""
tags:
- backlog
- bug
- testing
- coverage
- numpy
- matplotlib
---

# Task: Fix Coverage Collection Interference with NumPy

## Description

The test suite fails when run with coverage collection due to interference between pytest-cov and NumPy's internal state. This causes matplotlib subplot creation to fail with a `TypeError: float() argument must be a string or a real number, not '_NoValueType'` error.

The issue occurs when:
1. **Coverage collection** instruments the code to track execution
2. **NumPy module reloading** occurs during coverage collection (warning: "The NumPy module was reloaded (imported a second time)")
3. **Matplotlib** uses numpy's `np.min()` function with `initial=_NoValue` parameter
4. **Coverage instrumentation** corrupts numpy's internal state, causing `_NoValue` to become `_NoValueType` instead of the expected value
5. **Matplotlib fails** when trying to create subplots because `np.min()` can't handle the corrupted `_NoValueType`

**Affected Tests**: 7 tests fail when run with coverage (TestNeuralNetworkVisualizations and TestWardsMethod classes)

**Error**: `TypeError: float() argument must be a string or a real number, not '_NoValueType'`
**Location**: `numpy.core._methods.py:45` in `_amin()` function
**Trigger**: `matplotlib.gridspec.py:658` in `get_position()` when calling `fig_bottoms[rows].min()`

**Current Workaround**: Tests are currently skipped with `@pytest.mark.skip()` to allow the test suite to pass.

**⚠️ REJECTED APPROACH**: Test skipping is not an acceptable solution as it:
- Hides the underlying problem rather than fixing it
- Reduces test coverage and confidence
- Prevents proper validation of visualization functionality
- Creates technical debt that will need to be addressed later

## Acceptance Criteria

- [ ] All tests pass when run with coverage collection (NO test skipping)
- [ ] No numpy module reloading warnings
- [ ] Coverage collection works without interfering with matplotlib/numpy
- [ ] Test suite maintains 82% coverage
- [ ] No test isolation issues
- [ ] All visualization tests run and pass with coverage
- [ ] All Ward's method tests run and pass with coverage

## Implementation Notes

**Potential Solutions:**

1. **Update pytest-cov configuration** in `pytest.ini`:
   ```ini
   [tool:pytest]
   addopts = --cov=mlai.mlai --cov-report=term-missing --cov-omit=*/tests/*
   ```

2. **Add coverage exclusions** for problematic modules:
   ```ini
   [coverage:run]
   omit = 
       */tests/*
       */test_*
       */conftest.py
   ```

3. **Use a different coverage approach** that doesn't interfere with numpy

4. **Update dependency versions** to more compatible combinations:
   - Current: NumPy 1.26.4, Matplotlib 3.10.6, SciPy 1.16.0
   - Test with different versions to find compatible combinations

5. **Add coverage collection isolation** to prevent numpy state corruption

**Technical Details:**
- Environment: Python 3.11.7, pytest 8.4.1, pytest-cov 6.2.1
- Dependencies: NumPy 1.26.4, Matplotlib 3.10.6, SciPy 1.16.0
- Issue Type: Test environment interference
- Impact: 7 failing tests out of 164 total (157 passing)

## Related

- CIP: 0006 (Refactor mlai.py into Modular Structure)
- Test Coverage: 82% coverage achieved before issue
- Core Functionality: All ML algorithms work correctly (157/164 tests pass)

## Progress Updates

### 2025-10-12
- Issue identified and documented
- Root cause analysis completed
- Test skipping approach REJECTED as unacceptable
- Ready for proper implementation (no workarounds)
