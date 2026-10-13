---
category: bugs
created: '2026-10-13'
id: 2026-10-13_gpy-discussion-1001-reply
last_updated: '2026-10-13'
owner: Neil D. Lawrence
priority: Medium
related_cips: []
status: Ready
title: Post reply on GPy discussion #1001 (pinball)
---

# Task: Post reply on GPy discussion #1001

## Description

`gh` auth for `lawrennd` is currently invalid (`gh auth login` needed), so the
reply below was not posted automatically. Paste it into
[SheffieldML/GPy discussion #1001](https://github.com/SheffieldML/GPy/discussions/1001).

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

- [ ] Reply posted on discussion #1001
- [ ] Link this backlog item from the discussion if useful

## Progress Updates

### 2026-10-13

Draft prepared. Automatic post blocked: `gh` reports invalid keyring token for
`lawrennd` (`gh auth login -h github.com` required).
