---
category: features
created: '2026-09-29'
id: 2026-09-29_hmc-statistical-validation-tests
last_updated: '2026-09-29'
owner: Neil D. Lawrence
priority: Medium
related_cips:
- '0008'
status: Proposed
tags:
- backlog
- features
- hmc
- testing
- statistics
- mcmc
title: Statistical validation tests for teachable HMC
---

# Task: Statistical validation tests for teachable HMC

## Description

CIP-0008’s unit suite covers code paths and an **isotropic** $\mathcal{N}(0,I)$ moment check. That is not enough to claim sampler correctness under elongated / highly correlated targets, or to catch integrator and Metropolis bookkeeping bugs that moments can miss.

This task specifies a **layered** validation suite: deterministic integrator checks (always on), teaching-grade moment checks on a correlated Gaussian (default CI), and stronger / stress checks on the slow lane. Tolerances, budgets, and run policy are explicit so failures stay interpretable and Monte Carlo flake is controlled.

## Test layers

| Layer | Purpose | Default CI | Slow lane |
|---|---|---|---|
| A. Leapfrog reversibility | Integrator correctness | Yes | — |
| B. Energy / Metropolis sanity | $\Delta H$ bookkeeping | Yes | — |
| C. Correlated Gaussian moments | Anisotropic sampling | Yes | Tighter twin |
| D. 1D marginal vs analytic | Stronger than moments | No | Yes |
| E. Neal’s funnel (optional) | Varying curvature stress / teaching limit | No | Optional |
| F. ESS floor (optional) | Efficiency, not correctness | No | Soft optional |

**Out of scope for this CIP gate:** multimodal mixtures as pass/fail, multi-chain R-hat / Gelman–Rubin, NUTS benchmarks.

---

## Layer A — Leapfrog reversibility (default CI)

Deterministic. No Monte Carlo budget.

1. Pick $q_0$, $p_0$, $\varepsilon$, $L$, diagonal `mass`.
2. $(q_L, p_L) \leftarrow \mathrm{leapfrog}(q_0, p_0)$.
3. $(q', p') \leftarrow \mathrm{leapfrog}(q_L, -p_L)$ with the same $\varepsilon$, $L$, mass.
4. Assert $\|q' - q_0\|_\infty \le 10^{-10}$ (float64) and $\|p' + p_0\|_\infty \le 10^{-10}$.

Run for at least: isotropic quadratic $V$, and the $\rho=0.95$ quadratic $V$ from Layer C. Catches drift in the half-step / path-recording leapfrog variants.

---

## Layer B — Energy / Metropolis sanity (default CI)

1. **Tiny-$\varepsilon$ conservation** (already present): keep / extend so $|\Delta H| < 10^{-6}$ over a short trajectory. Do not remove.
2. **Creutz-style check** (additive): over $M \ge 200$ independent leapfrog proposals from $q\sim\mathcal{N}(0,I)$ (or the correlated target) with fixed modest $\varepsilon$, $L$:
   - Record $\Delta H = H(q',p') - H(q,p)$ for each proposal (before accept/reject).
   - Assert $\bigl| \frac1M \sum_m e^{-\Delta H_m} - 1 \bigr| \le 0.15$ at default budget; $\le 0.05$ on slow with $M \ge 2000$.
   - Guards sign errors in $\Delta H$ / accept probability. If this conflicts with a deliberately large $\varepsilon$, document hyperparameters used for the check (prefer a regime where accept rate is high).

---

## Layer C — Highly correlated bivariate Gaussian (primary statistical gate)

### Target

- $q \sim \mathcal{N}(0, \Sigma)$ with
  $$
  \Sigma = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}, \quad \rho = 0.95
  $$
- $V(q) = \frac12 q^\top \Sigma^{-1} q$, $\nabla V(q) = \Sigma^{-1} q$
- Fixed seed; document `step_size`, `n_steps`, `mass` (start `mass=1`; if accept rate collapses, try diagonal mass from $\mathrm{diag}(\Sigma)$ or $\mathrm{diag}(\Sigma^{-1})$ and document — no NUTS)

### Estimands

| Estimand | Population | Estimator |
|---|---|---|
| Mean | $0$ | Sample mean |
| Variances | $1$ | Sample variance (`ddof=1` / `np.cov`) |
| Covariance | $\rho=0.95$ | Off-diagonal of sample covariance |
| Accept rate | Tuned band | `HMCResult.accept_rate` |

Optional stretch: parametrize $\rho \in \{0.9, 0.95, 0.99\}$; only $\rho=0.95$ is required in default CI.

### Tolerances and budget

Teaching-grade absolute tolerances (detect gross failure), not asymptotic MCMC theory.

**Default CI**

| Quantity | Budget | Tolerance |
|---|---|---|
| $\|\bar q\|_\infty$ | $N=5000$ after $N_{\mathrm{warm}}=500$ discarded | $\le 0.12$ |
| $|\widehat{\mathrm{Var}}(q_i)-1|$ | same | $\le 0.20$ |
| $|\widehat{\mathrm{Cov}}(q_1,q_2)-0.95|$ | same | $\le 0.08$ |
| Accept rate | same chain | $\in [0.5, 0.95]$ |

**Slow** (`@pytest.mark.slow`)

| Quantity | Budget | Tolerance |
|---|---|---|
| Mean / var / cov | $N=20000$, $N_{\mathrm{warm}}=2000$ | Mean $\le 0.06$; var $\le 0.10$; cov $\le 0.04$ |
| Accept rate | same | Prefer $\in [0.6, 0.9]$ |

Keep existing isotropic `test_gaussian_target_moments`; Layer C is additive.

---

## Layer D — 1D marginal vs analytic (slow)

On the same $\rho=0.95$ chain (or a dedicated slow run):

- Project onto a coordinate or the leading eigenvector of $\Sigma$.
- Compare empirical CDF to $\mathcal{N}(0,1)$ via **KS** (`scipy.stats.kstest`) or a QQ max-gap proxy.
- **Tolerance:** KS $p$-value not required (seed-fixed $p$ is awkward); prefer statistic threshold e.g. $D_n \le 0.03$ at $N=20000$, or QQ absolute deviation $\le 0.08$ on the central 1%–99% percentiles.
- Mark `@pytest.mark.slow`. Skip if `scipy` unavailable only if the rest of the suite already treats scipy as required (it is, via `plot` / stack — use scipy).

---

## Layer E — Neal’s funnel (optional slow)

- Standard Neal funnel potential in 2D (or 10D if cheap enough).
- **Not** a Close gate for CIP-0008 unless teaching materials claim funnel competence.
- Soft checks: finite samples, accept rate in a documented band, no NaNs; optionally note bias in the neck as expected for diagonal-mass HMC.
- `@pytest.mark.slow`; may live behind a comment / separate test class `TestHMCNealFunnel`.

---

## Layer F — ESS floor (optional soft slow)

- After Layer C slow chain: estimate bulk ESS for each coordinate (e.g. `arviz` if already a dep, or a tiny ACF-based helper — **do not** add a heavy dependency solely for this).
- Soft assert `ESS / N > 0.05` (order-of-magnitude “not stuck”).
- Efficiency only; never relax Layer C tolerances because ESS is low — retune $\varepsilon$, $L$, mass instead.

---

## When tests run

| Lane | Includes |
|---|---|
| **Default local / PR CI** | Layers A, B, C (default budget); isotropic moment test |
| **Slow** (`pytest -m slow`) | C extended, D; optionally E, F |
| **PR CI** | Slow **off** unless a workflow already runs `-m slow` |

```python
@pytest.mark.slow
def test_hmc_correlated_gaussian_extended(...): ...
```

`slow` is already registered in `pytest.ini`.

### Flake policy

- Fixed integer `random_state` required for stochastic layers.
- On suspected flake: increase $N$ or widen tolerance only after documenting Monte Carlo SE — no silent retries, no `@pytest.mark.flaky`.

---

## Acceptance Criteria

- [ ] Layer A: reversibility tests for isotropic and $\rho=0.95$ potentials
- [ ] Layer B: tiny-$\varepsilon$ conservation retained; Creutz-style $\widehat{\mathbb{E}}[e^{-\Delta H}] \approx 1$ with documented tolerances
- [ ] Layer C: default-CI correlated Gaussian mean / var / cov / accept-rate within tables
- [ ] Layer C: `@pytest.mark.slow` extended-budget twin
- [ ] Layer D: slow marginal QQ or KS-style check vs analytic $\mathcal{N}(0,1)$
- [ ] Existing isotropic moment test retained
- [ ] Layers E–F either implemented as optional slow tests or explicitly deferred in a progress note
- [ ] CIP-0008 Testing Strategy updated with a short pointer to these layers / tolerances
- [ ] Default CI runs non-slow layers only

## Implementation Notes

- Primary home: `tests/unit/test_hmc.py` (classes e.g. `TestLeapfrogReversibility`, `TestHMCEnergySanity`, `TestHMCCorrelatedGaussian`, `TestHMCMarginalSlow`)
- Prefer analytic $2\times2$ $\Sigma^{-1}$ for Layer C
- Default correlated test runtime target: ≲5s on a laptop when practical
- Teaching clarity over library parity: document mass / step choices in test docstrings

## Related

- CIP: 0008
- REQ: 0008
- Depends on: `2026-09-29_hmc-core-sampler` (Completed)
- Related tests: `tests/unit/test_hmc.py` (`test_gaussian_target_moments`, `test_tiny_step_conserves_hamiltonian`)

## Progress Updates

### 2026-09-29
Task created for correlated-Gaussian moments, tolerances, and CI vs slow policy.

### 2026-09-29
Expanded to layered suite: reversibility (A), Creutz / $\Delta H$ sanity (B), correlated moments (C), slow marginal vs analytic (D), optional funnel (E) and ESS floor (F); multimodal / R-hat explicitly out of scope.
