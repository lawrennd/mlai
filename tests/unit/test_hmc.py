#!/usr/bin/env python3
"""
Tests for teachable Hamiltonian Monte Carlo (CIP-0008).
"""

import unittest

import numpy as np

import mlai
from mlai.hmc import leapfrog, kinetic_energy


def _standard_normal_potential(q):
    return 0.5 * float(np.dot(q, q))


def _standard_normal_grad(q):
    return np.asarray(q, dtype=float)


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


if __name__ == '__main__':
    unittest.main()
