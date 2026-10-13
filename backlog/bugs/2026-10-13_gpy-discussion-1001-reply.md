---
category: bugs
created: '2026-10-13'
id: 2026-10-13_gpy-discussion-1001-reply
last_updated: '2026-10-13'
owner: Neil D. Lawrence
priority: Medium
related_cips: []
status: Completed
title: Post reply on GPy discussion #1001 (pinball)
---

# Task: Post reply on GPy discussion #1001

## Description

Post a maintainer reply on
[SheffieldML/GPy discussion #1001](https://github.com/SheffieldML/GPy/discussions/1001)
pointing reporters at the `mlai` pinball fix (PR
[#1](https://github.com/lawrennd/mlai/pull/1)).

## Draft reply

```markdown
Thanks for the report — and sorry this sat so long.

The failure is not in GPy (or PyDeepGP). `visualize_pinball` lives in the
teaching helpers in [`lawrennd/mlai`](https://github.com/lawrennd/mlai)
(`mlai.deepgp_tutorial`), which the GPSS Deep GP notebooks monkey-patch onto
`deepgp.DeepGP`. When `layer.predict` returned non-flat mean/variance
(typical GPy shapes like `(n, 1)` or `(1, n)`), the uncertainty ribbon path
built a 2-D polygon and `ax.fill` raised the shape mismatch you saw.

That is fixed in `mlai` (squeeze/`reshape` of `y_std` and mean/var used for
ribbons only; next-layer inputs stay column vectors). There is a regression
test in `tests/unit/test_deepgp_tutorial.py`. After `pip install -U` from
`git+https://github.com/lawrennd/mlai.git` (or an editable install of main),
`m.visualize_pinball(...)` should work again on the Olympic marathon Deep GP
example.

GPSS lecture notebooks / snippets were also refreshed so they import from
`mlai.deepgp_tutorial` rather than a missing bare `deepgp_tutorial` module.
No GPy or PyDeepGP code change is required for this issue.
```

## Acceptance Criteria

- [x] Reply posted on discussion #1001
- [x] Reply links the mlai fix PR (`lawrennd/mlai#1`)

## Progress Updates

### 2026-10-13

Draft prepared. Automatic post initially blocked by Cursor seatbelt proxy
403 (misreported as invalid `gh` token).

### 2026-10-13 (later)

Posted via unsandboxed `gh` GraphQL after bridle path validated:
https://github.com/SheffieldML/GPy/discussions/1001#discussioncomment-18905521
Marked Completed.
