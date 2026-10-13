---
category: bugs
created: '2026-10-13'
id: 2026-10-13_deepgp-pinball-shape-mismatch
last_updated: '2026-10-13'
owner: Neil D. Lawrence
priority: High
related_cips: []
status: Completed
title: Fix visualize_pinball ribbon shape mismatch for non-flat predict outputs
---

# Task: Fix visualize_pinball ribbon shape mismatch

## Description

`mlai.deepgp_tutorial.visualize_pinball` failed when layer `predict` returned
non-flat mean/variance (common GPy shapes `(n, 1)` or `(1, n)`). `scale_data`
squeezed the means, but `y_std` was left multi-dimensional, so

```python
stdx = np.sqrt(stdy) * y_std[i]
```

produced a `(50, N)` array and `ax.fill` raised:

```text
ValueError: x and y must have same first dimension, but have shapes (50, 100) and (100,)
```

Reported against GPy as [discussion #1001](https://github.com/SheffieldML/GPy/discussions/1001)
(`error_pinball_DGP`); the helpers live in `mlai`, not in GPy or PyDeepGP.

## Acceptance Criteria

- [x] `y_std` is forced to a 1-D length-`n` vector before ribbon construction
- [x] `yt_mean` / `yt_var` from `layer.predict` are squeezed before use
- [x] Unit regression covers `(n, 1)`, `(1, n)`, and flat predict shapes
- [x] `ax.fill` always receives matching 1-D polygon coordinates

## Implementation Notes

- Fix in `mlai/deepgp_tutorial.py` nested `pinball` helper and the predict loop
- Tests in `tests/unit/test_deepgp_tutorial.py`

## Related

- GitHub: [SheffieldML/GPy discussion #1001](https://github.com/SheffieldML/GPy/discussions/1001)
- Reply draft (post pending `gh auth login`): `backlog/bugs/2026-10-13_gpy-discussion-1001-reply.md`
- GPSS Deep GP notebooks rebuild after this fix
- Snippets: `_deepgp/includes/deepgp-enhance.md` / `deep-gp-setup-code.md`

## Progress Updates

### 2026-10-13

Bug diagnosed from GPy discussion #1001 and GPSS Olympic marathon pinball path.
Fix and regression tests applied; task marked Completed.
