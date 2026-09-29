#!/usr/bin/env python3
"""
Tests for teachable Hamiltonian Monte Carlo (CIP-0008).

Statistical layers A–F follow backlog
``2026-09-29_hmc-statistical-validation-tests``.
"""

import unittest

import numpy as np
import pytest
from scipy import stats

import mlai
from mlai.hmc import leapfrog, kinetic_energy


def _standard_normal_potential(q):
    return 0.5 * float(np.dot(q, q))


def _standard_normal_grad(q):
    return np.asarray(q, dtype=float)


# Layer C target: N(0, Σ) with ρ = 0.95
_CORR_RHO = 0.95
_CORR_PREC = 1.0 / (1.0 - _CORR_RHO**2)
_CORR_SIGMA_INV = _CORR_PREC * np.array(
    [[1.0, -_CORR_RHO], [-_CORR_RHO, 1.0]], dtype=float
)
_CORR_SIGMA = np.array([[1.0, _CORR_RHO], [_CORR_RHO, 1.0]], dtype=float)


def _correlated_potential(q):
    q = np.asarray(q, dtype=float)
    return 0.5 * float(q @ _CORR_SIGMA_INV @ q)


def _correlated_grad(q):
    return _CORR_SIGMA_INV @ np.asarray(q, dtype=float)


def _ess_acf(x, max_lag=None):
    """Bulk ESS from truncated positive ACF (no arviz dependency)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return float('nan')
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0.0:
        return float('nan')
    if max_lag is None:
        max_lag = min(n - 1, max(50, n // 5))
    # Biased ACF via FFT-free loop; n is modest in tests
    acf_sum = 0.0
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(x[:-lag], x[lag:]) / (n * var))
        if rho <= 0.0:
            break
        acf_sum += 2.0 * rho
    return n / (1.0 + acf_sum)


class TestKineticAndLeapfrog(unittest.TestCase):
    """Unit tests for kinetic energy and leapfrog primitives."""

    def test_kinetic_energy_identity_mass(self):
        p = np.array([3.0, 4.0])
        mass = np.ones(2)
        self.assertAlmostEqual(kinetic_energy(p, mass), 0.5 * 25.0)

    def test_kinetic_energy_scaled_mass(self):
        p = np.array([2.0, 2.0])
        mass = np.array([4.0, 1.0])
        # 0.5 * (4/4 + 4/1) = 0.5 * (1 + 4) = 2.5
        self.assertAlmostEqual(kinetic_energy(p, mass), 2.5)

    def test_leapfrog_n_steps_one_matches_manual(self):
        q = np.array([1.0, -0.5])
        p = np.array([0.2, 0.3])
        mass = np.ones(2)
        eps = 0.05

        q_out, p_out = leapfrog(q, p, _standard_normal_grad, eps, 1, mass)

        # Manual single leapfrog step for V(q)=0.5||q||^2, ∇V=q, M=I
        p_half = p - 0.5 * eps * q
        q_new = q + eps * p_half
        p_new = p_half - 0.5 * eps * q_new
        np.testing.assert_allclose(q_out, q_new)
        np.testing.assert_allclose(p_out, p_new)

    def test_leapfrog_record_path_length(self):
        q = np.zeros(2)
        p = np.array([1.0, -1.0])
        mass = np.ones(2)
        q_out, p_out, path = leapfrog(
            q, p, _standard_normal_grad, 0.1, 5, mass, record_path=True
        )
        self.assertEqual(path.shape, (6, 2))
        np.testing.assert_allclose(path[0], q)
        np.testing.assert_allclose(path[-1], q_out)
        self.assertEqual(p_out.shape, (2,))


class TestHamiltonianMonteCarlo(unittest.TestCase):
    """Tests for HamiltonianMonteCarlo sampler."""

    def test_exports_on_mlai(self):
        self.assertTrue(hasattr(mlai, 'HamiltonianMonteCarlo'))
        self.assertTrue(hasattr(mlai, 'HMCResult'))
        self.assertTrue(hasattr(mlai, 'leapfrog'))
        self.assertTrue(hasattr(mlai, 'kinetic_energy'))

    def test_invalid_step_size(self):
        with self.assertRaises(ValueError):
            mlai.HamiltonianMonteCarlo(
                _standard_normal_potential, _standard_normal_grad, step_size=0.0
            )

    def test_invalid_n_steps(self):
        with self.assertRaises(ValueError):
            mlai.HamiltonianMonteCarlo(
                _standard_normal_potential, _standard_normal_grad, n_steps=0
            )

    def test_mass_shape_mismatch(self):
        hmc = mlai.HamiltonianMonteCarlo(
            _standard_normal_potential, _standard_normal_grad, mass=[1.0, 2.0]
        )
        with self.assertRaises(ValueError):
            hmc.sample(q0=np.zeros(3), n_samples=2, random_state=0)

    def test_seed_reproducibility(self):
        hmc = mlai.HamiltonianMonteCarlo(
            _standard_normal_potential,
            _standard_normal_grad,
            step_size=0.1,
            n_steps=10,
        )
        a = hmc.sample(q0=np.zeros(2), n_samples=50, random_state=42)
        b = hmc.sample(q0=np.zeros(2), n_samples=50, random_state=42)
        np.testing.assert_array_equal(a.samples, b.samples)
        self.assertEqual(a.accept_rate, b.accept_rate)
        np.testing.assert_array_equal(a.hamiltonian_trace, b.hamiltonian_trace)

    def test_gaussian_target_moments(self):
        """Empirical mean near 0 and covariance near I for N(0,I)."""
        hmc = mlai.HamiltonianMonteCarlo(
            _standard_normal_potential,
            _standard_normal_grad,
            step_size=0.15,
            n_steps=10,
        )
        result = hmc.sample(q0=np.zeros(2), n_samples=4000, random_state=0)
        mean = result.samples.mean(axis=0)
        cov = np.cov(result.samples, rowvar=False)

        np.testing.assert_allclose(mean, np.zeros(2), atol=0.15)
        np.testing.assert_allclose(cov, np.eye(2), atol=0.25)
        self.assertGreater(result.accept_rate, 0.5)
        self.assertLessEqual(result.accept_rate, 1.0)
        self.assertEqual(result.n_accepted, int(round(result.accept_rate * 4000)))

    def test_hamiltonian_trace_stored(self):
        hmc = mlai.HamiltonianMonteCarlo(
            _standard_normal_potential, _standard_normal_grad, step_size=0.1, n_steps=5
        )
        result = hmc.sample(q0=np.ones(2), n_samples=20, random_state=1)
        self.assertIsNotNone(result.hamiltonian_trace)
        self.assertEqual(result.hamiltonian_trace.shape, (20,))
        self.assertTrue(np.all(np.isfinite(result.hamiltonian_trace)))

    def test_no_hamiltonian_trace_when_disabled(self):
        hmc = mlai.HamiltonianMonteCarlo(
            _standard_normal_potential, _standard_normal_grad
        )
        result = hmc.sample(
            q0=np.zeros(2), n_samples=10, random_state=0, store_hamiltonian=False
        )
        self.assertIsNone(result.hamiltonian_trace)

    def test_tiny_step_conserves_hamiltonian(self):
        """Very small ε should keep |ΔH| tiny over a leapfrog trajectory."""
        hmc = mlai.HamiltonianMonteCarlo(
            _standard_normal_potential,
            _standard_normal_grad,
            step_size=1e-4,
            n_steps=20,
        )
        q0 = np.array([0.5, -0.3])
        rng = np.random.default_rng(0)
        mass = np.ones(2)
        p0 = rng.normal(0.0, 1.0, size=2)
        h0 = hmc.hamiltonian(q0, p0, mass)
        q1, p1 = leapfrog(q0, p0, _standard_normal_grad, 1e-4, 20, mass)
        h1 = hmc.hamiltonian(q1, p1, mass)
        self.assertLess(abs(h1 - h0), 1e-6)

    def test_tuned_accept_rate_band(self):
        hmc = mlai.HamiltonianMonteCarlo(
            _standard_normal_potential,
            _standard_normal_grad,
            step_size=0.1,
            n_steps=10,
        )
        result = hmc.sample(q0=np.zeros(3), n_samples=500, random_state=7)
        self.assertGreater(result.accept_rate, 0.7)
        self.assertLessEqual(result.accept_rate, 1.0)

    def test_return_trajectory(self):
        hmc = mlai.HamiltonianMonteCarlo(
            _standard_normal_potential,
            _standard_normal_grad,
            step_size=0.1,
            n_steps=4,
        )
        result, trajectories = hmc.sample(
            q0=np.zeros(2), n_samples=3, random_state=0, return_trajectory=True
        )
        self.assertEqual(result.samples.shape, (3, 2))
        self.assertEqual(len(trajectories), 3)
        self.assertEqual(trajectories[0].shape, (5, 2))

    def test_diagonal_mass_runs(self):
        hmc = mlai.HamiltonianMonteCarlo(
            _standard_normal_potential,
            _standard_normal_grad,
            step_size=0.1,
            n_steps=5,
            mass=[1.0, 2.0],
        )
        result = hmc.sample(q0=np.zeros(2), n_samples=100, random_state=3)
        self.assertEqual(result.samples.shape, (100, 2))
        self.assertGreater(result.accept_rate, 0.0)


class TestHMCTeachingExample(unittest.TestCase):
    """Teaching path: logistic SGD point estimate vs HMC samples."""

    def test_logistic_sgd_vs_hmc(self):
        from mlai import LR, Basis, linear

        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 1))
        y = (1.2 * X[:, 0] + 0.2 * rng.normal(size=50) > 0).astype(float).reshape(-1, 1)
        model = LR(X, y, Basis(linear, number=2))

        def potential(q):
            model.parameters = q
            return -float(model.log_likelihood())

        def grad_potential(q):
            model.parameters = q
            return np.asarray(model.gradients, dtype=float)

        q = np.zeros(2)
        for _ in range(200):
            q = q - 0.05 * grad_potential(q)
        v_sgd = potential(q)

        hmc = mlai.HamiltonianMonteCarlo(
            potential, grad_potential, step_size=0.04, n_steps=8
        )
        result = hmc.sample(q0=q, n_samples=150, random_state=1)
        self.assertGreater(result.accept_rate, 0.5)
        self.assertEqual(result.samples.shape, (150, 2))
        # Posterior mean should stay in the same basin as the SGD point
        self.assertLess(np.linalg.norm(result.samples.mean(0) - q), 3.0)
        self.assertLess(v_sgd, potential(np.zeros(2)))


class TestLeapfrogReversibility(unittest.TestCase):
    """Layer A: leapfrog-forward then reverse recovers (q, p)."""

    def _assert_reversible(self, grad_fn, q0, p0, step_size, n_steps, mass):
        q_l, p_l = leapfrog(q0, p0, grad_fn, step_size, n_steps, mass)
        q_b, p_b = leapfrog(q_l, -p_l, grad_fn, step_size, n_steps, mass)
        self.assertLessEqual(np.max(np.abs(q_b - q0)), 1e-10)
        self.assertLessEqual(np.max(np.abs(p_b + p0)), 1e-10)

    def test_reversible_isotropic_quadratic(self):
        mass = np.ones(2)
        self._assert_reversible(
            _standard_normal_grad,
            q0=np.array([0.7, -1.2]),
            p0=np.array([0.3, 0.5]),
            step_size=0.2,
            n_steps=15,
            mass=mass,
        )

    def test_reversible_correlated_quadratic(self):
        mass = np.ones(2)
        self._assert_reversible(
            _correlated_grad,
            q0=np.array([0.4, -0.9]),
            p0=np.array([-0.2, 0.6]),
            step_size=0.15,
            n_steps=12,
            mass=mass,
        )

    def test_reversible_with_record_path_endpoint(self):
        """Path-recording leapfrog must share the same endpoint map."""
        mass = np.ones(2)
        q0 = np.array([1.0, 0.25])
        p0 = np.array([-0.4, 0.8])
        q_a, p_a = leapfrog(q0, p0, _correlated_grad, 0.1, 8, mass)
        q_b, p_b, path = leapfrog(
            q0, p0, _correlated_grad, 0.1, 8, mass, record_path=True
        )
        np.testing.assert_allclose(q_a, q_b)
        np.testing.assert_allclose(p_a, p_b)
        np.testing.assert_allclose(path[-1], q_a)
        q_r, p_r = leapfrog(q_a, -p_a, _correlated_grad, 0.1, 8, mass)
        self.assertLessEqual(np.max(np.abs(q_r - q0)), 1e-10)
        self.assertLessEqual(np.max(np.abs(p_r + p0)), 1e-10)


class TestHMCEnergySanity(unittest.TestCase):
    """Layer B: tiny-ε conservation (retained above) + Creutz-style ΔH check."""

    def test_creutz_expectation_default(self):
        """E[exp(-ΔH)] ≈ 1 for equilibrium starts (high-accept ε, L).

        Hyperparameters: ε=0.1, L=10, M=200, mass=I; q ~ N(0, Σ) with ρ=0.95,
        p ~ N(0, I). Tolerance |mean - 1| ≤ 0.15.
        """
        rng = np.random.default_rng(0)
        mass = np.ones(2)
        eps, n_steps, m = 0.1, 10, 200
        exps = np.empty(m)
        for i in range(m):
            q = rng.multivariate_normal(np.zeros(2), _CORR_SIGMA)
            p = rng.normal(size=2)
            h0 = _correlated_potential(q) + kinetic_energy(p, mass)
            q1, p1 = leapfrog(q, p, _correlated_grad, eps, n_steps, mass)
            h1 = _correlated_potential(q1) + kinetic_energy(p1, mass)
            exps[i] = np.exp(-(h1 - h0))
        self.assertLessEqual(abs(float(exps.mean()) - 1.0), 0.15)

    @pytest.mark.slow
    def test_creutz_expectation_slow(self):
        """Creutz with M=2000 and |mean - 1| ≤ 0.05 (same ε, L as default)."""
        rng = np.random.default_rng(0)
        mass = np.ones(2)
        eps, n_steps, m = 0.1, 10, 2000
        exps = np.empty(m)
        for i in range(m):
            q = rng.multivariate_normal(np.zeros(2), _CORR_SIGMA)
            p = rng.normal(size=2)
            h0 = _correlated_potential(q) + kinetic_energy(p, mass)
            q1, p1 = leapfrog(q, p, _correlated_grad, eps, n_steps, mass)
            h1 = _correlated_potential(q1) + kinetic_energy(p1, mass)
            exps[i] = np.exp(-(h1 - h0))
        self.assertLessEqual(abs(float(exps.mean()) - 1.0), 0.05)


class TestHMCCorrelatedGaussian(unittest.TestCase):
    """Layer C: highly correlated bivariate Gaussian moments."""

    # Tuned for accept ∈ [0.5, 0.95] with mass=I (documented choice).
    _STEP = 0.3
    _N_STEPS = 10

    def test_correlated_gaussian_moments_default(self):
        """ρ=0.95, N=5000 after warm=500; teaching-grade absolute tolerances.

        step_size=0.3, n_steps=10, mass=1 (identity). Accept band [0.5, 0.95].
        """
        hmc = mlai.HamiltonianMonteCarlo(
            _correlated_potential,
            _correlated_grad,
            step_size=self._STEP,
            n_steps=self._N_STEPS,
            mass=1.0,
        )
        warm = 500
        n = 5000
        result = hmc.sample(
            q0=np.zeros(2), n_samples=warm + n, random_state=0
        )
        samples = result.samples[warm:]
        mean = samples.mean(axis=0)
        cov = np.cov(samples, rowvar=False)

        self.assertLessEqual(np.max(np.abs(mean)), 0.12)
        self.assertLessEqual(np.max(np.abs(np.diag(cov) - 1.0)), 0.20)
        self.assertLessEqual(abs(cov[0, 1] - _CORR_RHO), 0.08)
        self.assertGreaterEqual(result.accept_rate, 0.5)
        self.assertLessEqual(result.accept_rate, 0.95)

    @pytest.mark.slow
    def test_correlated_gaussian_moments_extended(self):
        """Slow twin: N=20000 after warm=2000; tighter tolerances."""
        hmc = mlai.HamiltonianMonteCarlo(
            _correlated_potential,
            _correlated_grad,
            step_size=self._STEP,
            n_steps=self._N_STEPS,
            mass=1.0,
        )
        warm = 2000
        n = 20000
        result = hmc.sample(
            q0=np.zeros(2), n_samples=warm + n, random_state=1
        )
        samples = result.samples[warm:]
        mean = samples.mean(axis=0)
        cov = np.cov(samples, rowvar=False)

        self.assertLessEqual(np.max(np.abs(mean)), 0.06)
        self.assertLessEqual(np.max(np.abs(np.diag(cov) - 1.0)), 0.10)
        self.assertLessEqual(abs(cov[0, 1] - _CORR_RHO), 0.04)
        self.assertGreaterEqual(result.accept_rate, 0.6)
        self.assertLessEqual(result.accept_rate, 0.9)


class TestHMCMarginalSlow(unittest.TestCase):
    """Layer D: 1D marginal vs analytic N(0,1) (slow)."""

    @pytest.mark.slow
    def test_coordinate_marginal_ks(self):
        """KS D_n ≤ 0.03 for q_0 ~ N(0,1) on ρ=0.95 HMC chain (N=20000).

        Uses the same hyperparameters as Layer C (ε=0.3, L=10, mass=I).
        Threshold is on the KS statistic, not a p-value (seed-fixed p is awkward;
        dependent samples make p-values invalid anyway).
        """
        hmc = mlai.HamiltonianMonteCarlo(
            _correlated_potential,
            _correlated_grad,
            step_size=0.3,
            n_steps=10,
            mass=1.0,
        )
        warm = 2000
        n = 20000
        result = hmc.sample(
            q0=np.zeros(2), n_samples=warm + n, random_state=1
        )
        samples = result.samples[warm:]
        d_n = float(stats.kstest(samples[:, 0], 'norm').statistic)
        self.assertLessEqual(d_n, 0.03)
        # QQ proxy on central 1%–99% percentiles
        pct = np.linspace(0.01, 0.99, 99)
        emp = np.quantile(samples[:, 0], pct)
        theo = stats.norm.ppf(pct)
        self.assertLessEqual(np.max(np.abs(emp - theo)), 0.08)


class TestHMCNealFunnel(unittest.TestCase):
    """Layer E (optional): soft Neal funnel stress — not a CIP-0008 Close gate."""

    @pytest.mark.slow
    def test_funnel_finite_samples(self):
        """2D Neal funnel: finite samples, no NaNs, documented accept band.

        V(x, z) = 0.5 z^2 + 0.5 (x^2 exp(-z) + z) with q=[x, z].
        Diagonal-mass HMC is known to struggle in the neck; we only check
        that the chain stays finite with a non-vanishing accept rate.
        """

        def potential(q):
            x, z = float(q[0]), float(q[1])
            return 0.5 * z * z + 0.5 * (x * x * np.exp(-z) + z)

        def grad(q):
            x, z = float(q[0]), float(q[1])
            return np.array(
                [x * np.exp(-z), z + 0.5 - 0.5 * x * x * np.exp(-z)],
                dtype=float,
            )

        hmc = mlai.HamiltonianMonteCarlo(
            potential, grad, step_size=0.05, n_steps=10, mass=1.0
        )
        result = hmc.sample(q0=np.zeros(2), n_samples=2000, random_state=0)
        self.assertTrue(np.all(np.isfinite(result.samples)))
        self.assertGreater(result.accept_rate, 0.2)
        self.assertLessEqual(result.accept_rate, 1.0)


class TestHMCESSFloor(unittest.TestCase):
    """Layer F (optional soft): ESS / N > 0.05 after Layer C slow budget."""

    @pytest.mark.slow
    def test_ess_floor_correlated(self):
        hmc = mlai.HamiltonianMonteCarlo(
            _correlated_potential,
            _correlated_grad,
            step_size=0.3,
            n_steps=10,
            mass=1.0,
        )
        warm = 2000
        n = 20000
        result = hmc.sample(
            q0=np.zeros(2), n_samples=warm + n, random_state=1
        )
        samples = result.samples[warm:]
        for j in range(2):
            ess = _ess_acf(samples[:, j])
            self.assertGreater(ess / n, 0.05)


if __name__ == '__main__':
    unittest.main()
