"""Tests for CIP-000C teachable LQ sparse VFE/DTC."""

import numpy as np
import pytest

import mlai
from mlai import (
    SparseGP,
    Kernel,
    eq_cov,
    lq_factor,
    Adam,
    finite_difference_gradient,
    verify_gradient_implementation,
)


def _toy_model(latent_X=False, approximation='vfe', optimize_Z=True, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(20, 1)
    y = np.sin(X[:, 0]) + 0.05 * rng.randn(20)
    Z = np.linspace(-2, 2, 5)[:, None]
    kernel = Kernel(eq_cov, variance=1.0, lengthscale=1.0)
    return SparseGP(
        X, y, Z, sigma2=0.1, kernel=kernel,
        approximation=approximation, latent_X=latent_X, optimize_Z=optimize_Z,
    )


class TestLQFactor:
    def test_reconstructs_and_positive_diag(self):
        rng = np.random.RandomState(1)
        W = rng.randn(4, 12)
        L, Q = lq_factor(W)
        assert np.all(np.diag(L) > 0)
        np.testing.assert_allclose(L @ Q, W, atol=1e-12)
        np.testing.assert_allclose(Q @ Q.T, np.eye(4), atol=1e-12)


class TestSparseGPForward:
    def test_lq_matches_gram_chol(self):
        model = _toy_model()
        np.testing.assert_allclose(
            model.objective(), model.gram_chol_neg_bound(), atol=1e-10
        )

    def test_dtc_differs_from_vfe(self):
        vfe = _toy_model(approximation='vfe')
        dtc = _toy_model(approximation='dtc')
        assert vfe.objective() != pytest.approx(dtc.objective())

    def test_predict_shapes(self):
        model = _toy_model()
        mu, var = model.predict(np.linspace(-1, 1, 7)[:, None])
        assert mu.shape == (7,)
        assert var.shape == (7,)
        assert np.all(var >= 0)

    def test_predict_matches_dtc_closed_form(self):
        """Mean/var must match DTC algebra (no spurious 1/σ on the mean)."""
        model = _toy_model(seed=2)
        X_test = np.linspace(-1.5, 1.5, 11)[:, None]
        mu, var = model.predict(X_test)

        Kmm, Kmn, _ = model._kernel_blocks()
        Kms = model.kernel.K(model.Z, X_test)
        Kss = model.kernel.diag(X_test).ravel()
        sigma2 = model.sigma2
        # μ* = K_*m (σ² Kmm + Kmn Knm)^{-1} Kmn y
        Sigma = sigma2 * Kmm + Kmn @ Kmn.T
        mu_ref = Kms.T @ np.linalg.solve(Sigma, Kmn @ model.y)
        # Latent var: k** - Q** + K_*m (Kmm + Kmn Knm/σ²)^{-1} K_m*
        Qdiag = np.sum(
            np.linalg.solve(Kmm, Kms) * Kms, axis=0
        )
        cov_u = np.linalg.inv(Kmm + (Kmn @ Kmn.T) / sigma2)
        var_ref = Kss - Qdiag + np.sum((Kms.T @ cov_u) * Kms.T, axis=1)

        np.testing.assert_allclose(mu, mu_ref, atol=1e-10)
        np.testing.assert_allclose(var, var_ref, atol=1e-10)
        # Guard the historical 1/σ mean bug when noise ≠ 1
        assert not np.allclose(mu, mu_ref / np.sqrt(sigma2), atol=1e-5)

class TestSparseGPSharedSurface:
    def test_parameters_roundtrip(self):
        model = _toy_model()
        original = model.parameters.copy()
        model.parameters = original + 0.01
        np.testing.assert_allclose(model.parameters, original + 0.01)
        model.parameters = original
        np.testing.assert_allclose(model.parameters, original)

    def test_gradients_match_finite_differences(self):
        model = _toy_model()
        # Check a reduced block: noise + kernel (skip Z for speed/stability)
        n_log = 1 + len(model._scalar_kernel_param_names())

        def objective_func(log_params):
            full = model.parameters.copy()
            full[:n_log] = log_params
            model.parameters = full
            return model.objective()

        analytical = model.gradients[:n_log].copy()
        numerical = finite_difference_gradient(
            objective_func, model.parameters[:n_log].copy(), h=1e-5
        )
        assert verify_gradient_implementation(
            analytical, numerical, rtol=2e-2, atol=1e-3
        )

    def test_adam_reduces_objective(self):
        model = _toy_model()
        # Perturb lengthscale / noise away from a reasonable regime
        p = model.parameters.copy()
        p[0] = np.log(0.5)
        p[2] = np.log(0.2)
        model.parameters = p
        initial = model.objective()
        opt = Adam(learning_rate=0.05)
        for _ in range(40):
            opt.step(model)
        assert model.objective() < initial

    def test_latent_X_in_parameter_vector(self):
        model = _toy_model(latent_X=True)
        assert model.parameters.size == (
            1 + 2 + model.Z.size + model.X.size
        )
        initial = model.objective()
        opt = Adam(learning_rate=0.05)
        for _ in range(50):
            opt.step(model)
        assert np.isfinite(model.objective())
        assert model.objective() < initial

    def test_optimize_Z_false_excludes_inducing(self):
        model = _toy_model(optimize_Z=False)
        Z0 = model.Z.copy()
        assert model.parameters.size == 1 + len(model._scalar_kernel_param_names())
        assert model.gradients.size == model.parameters.size
        p = model.parameters.copy()
        p[0] = np.log(0.5)
        model.parameters = p
        np.testing.assert_allclose(model.Z, Z0)
        from mlai import SCG
        SCG().optimize(model, max_iters=20)
        np.testing.assert_allclose(model.Z, Z0)

    def test_param_layout_respects_optimize_Z_and_latent_X(self):
        free = _toy_model(optimize_Z=True)
        names = [name for name, _ in free._param_layout()]
        assert names[0] == 'sigma2'
        assert 'Z' in names
        assert 'X' not in names

        fixed = _toy_model(optimize_Z=False)
        assert 'Z' not in [name for name, _ in fixed._param_layout()]

        latent = _toy_model(optimize_Z=True, latent_X=True)
        latent_names = [name for name, _ in latent._param_layout()]
        assert 'Z' in latent_names and 'X' in latent_names
