---
author: "Neil Lawrence"
created: "2025-07-16"
id: "2025-07-16_plot-py-testing"
last_updated: "2025-07-16"
status: proposed
tags:
- backlog
- features
- testing
- plot
- coverage
title: "Build Comprehensive Tests for plot.py"
---

# Task: Build Comprehensive Tests for plot.py


## Description
The `plot.py` module currently has very low test coverage (8%) and needs comprehensive testing to ensure reliability and maintainability. This module contains critical plotting functionality used throughout the MLAI tutorials and examples.

## Acceptance Criteria
- [ ] Unit tests for all major plotting functions
- [x] Tests for plot customization options
- [x] Tests for error handling and edge cases
- [x] Mock tests for matplotlib dependencies
- [ ] Integration tests for plot generation workflows
- [ ] Tests for plot file output and saving
- [ ] Coverage target: >80% for plot.py
- [ ] Documentation of test patterns for plotting code

## Implementation Notes
- Use matplotlib testing utilities for plot verification
- Mock matplotlib backend to avoid display issues in CI
- Test both interactive and non-interactive plotting modes
- Focus on testing plot generation logic rather than visual appearance
- Use pytest fixtures for common plot data and configurations
- Consider using pytest-mpl for image comparison tests
- Test plot file output using temporary directories

## Current State
- plot.py has 1906 statements with only 8% coverage
- Most plotting functions are untested
- No integration tests for plot workflows
- Limited error handling tests

## Related
- CIP: 0002 (Comprehensive Test Framework)
- Current Coverage: 8% (1906 statements, 1752 missed)
- Target Coverage: >80%

## Progress Updates

### 2025-07-16
Task created with Proposed status.

### 2025-07-16 (Updated)
*Significant Progress Made:*
- *Created 35 comprehensive unit tests* covering foundational plotting infrastructure:
  - Plot utilities (constants, pred_range function)
  - Matrix plotting (all display types: values, entries, image, patch, colorpatch)
  - Network visualization (network and layer classes)
  - Model output functions (model_output, model_sample)
  - Error handling and edge cases
  - Performance testing (large matrices, high point counts)
  - Integration testing (matplotlib, IPython, optional dependencies)
- *Fixed critical bugs in plot.py*:
  - String formatting issue with integer values in matrix function
  - Made model_output and model_sample return axis objects
  - Improved parameter handling for highlight_row, highlight_col, zoom_row, zoom_col
  - Maintained backward compatibility with ":" string parameter
- *All 35 tests passing* (100% success rate)
- *Status*: Still in progress - tested ~6-8 functions out of ~60+ total functions
- *Remaining Work*: Need to test remaining ~50+ functions including covariance_capacity, prob_diagram, perceptron visualization, regression plotting, model evaluation functions, statistical plots, kernel visualization, and advanced visualization functions 