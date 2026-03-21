---
id: "2026-03-21_mountain-car-stale-mlai-import"
title: "Fix stale mlai.mlai import in mountain_car.py"
status: "Completed"
priority: "High"
created: "2026-03-21"
last_updated: "2026-03-21"
category: "bugs"
---

# Task: Fix stale mlai.mlai import in mountain_car.py

## Description

`mlai/mountain_car.py` contained a stale import:

```python
import mlai.mlai as ma
```

This referred to a `mlai.py` submodule that no longer exists inside the `mlai` package. When GPy is installed, importing `mlai` would fail with:

```
ModuleNotFoundError: No module named 'mlai.mlai'
```

The functions actually used via the `ma` alias (`write_figure` and `write_animation_html`) both live in `mlai/utils.py`.

## Acceptance Criteria

- [x] `import mlai` succeeds when GPy is installed
- [x] `mountain_car` module imports correctly as part of the GPy-conditional block in `__init__.py`
- [x] `ma.write_figure` and `ma.write_animation_html` resolve to the correct functions in `mlai.utils`
- [x] Tests cover the import and the alias binding, running without GPy via a mock

## Implementation Notes

- Changed line 16 of `mlai/mountain_car.py` from `import mlai.mlai as ma` to `import mlai.utils as ma`
- Added `tests/unit/test_mountain_car.py` using `unittest.mock.patch.dict` to inject a mock GPy so tests run in environments without GPy installed
- Same mocking pattern already used by `tests/unit/test_deepgp_tutorial.py`

## Related

- Commit: `970c4bb` — Fix stale mlai.mlai import in mountain_car and add tests

## Progress Updates

### 2026-03-21
Bug identified during notebook usage with GPy available. Fix applied and tests added in commit `970c4bb`. Task created retrospectively as Completed.
