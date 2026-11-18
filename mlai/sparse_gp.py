"""
Sparse variational Gaussian processes (teachable LQ / VFE path).

Collapsed VFE/DTCVAR with LQ factorisation of stacked whitened features
``W = [I, A]`` (Seeger et al. 2017 BLR trick), as planned in CIP-000C.
"""

from __future__ import annotations

import inspect

import numpy as np

from .models import ProbMapModel
from .parameters import POSITIVE, Param, ParamGroup

__all__ = [
    'lq_factor',
    'lq_adjoint',
    'SparseGP',
]


def lq_factor(W):
    """
    Factor wide ``W`` (m×n, n≥m) as ``W = L @ Q`` with ``diag(L) > 0``.

    Uses QR of ``W.T`` (NumPy has no ``gelqf``). Rows of ``Q`` are orthonormal.

    :param W: Wide matrix
    :type W: numpy.ndarray
    :returns: ``(L, Q)`` with ``L`` lower triangular m×m and ``Q`` m×n
    :rtype: tuple
    """
    Qr, Rr = np.linalg.qr(W.T, mode='reduced')
    L = Rr.T.copy()
    Q = Qr.T.copy()
    signs = np.sign(np.diag(L))
    signs[signs == 0.0] = 1.0
    # Column-scale L and row-scale Q so W = L @ Q is preserved and diag(L) > 0.
    L *= signs[np.newaxis, :]
    Q *= signs[:, np.newaxis]
    return L, Q


def lq_adjoint(L, Q, L_bar, Q_bar=None):
    """
    Reverse mode for ``W = L @ Q`` (Seeger et al. §4.3).

    :param L: Lower factor from :func:`lq_factor`
    :param Q: Orthonormal-row factor from :func:`lq_factor`
    :param L_bar: Cotangent w.r.t. ``L``
    :param Q_bar: Cotangent w.r.t. ``Q`` (optional, default zero)
    :returns: Cotangent w.r.t. ``W``
    :rtype: numpy.ndarray
    """
    if Q_bar is None:
        Q_bar = np.zeros_like(Q)
    M = L.T @ L_bar - Q_bar @ Q.T
    # copyltu(M): copy strict lower triangle into strict upper
    M = np.tril(M) + np.tril(M, -1).T
    rhs = Q_bar + M @ Q
    return np.linalg.solve(L.T, rhs)


def _chol_lower_adjoint(L, L_bar):
    """Giles/Murray adjoint for ``K = L @ L.T`` given cotangent ``L_bar``."""
    Phi = np.tril(L.T @ L_bar)
    Phi = Phi - 0.5 * np.diag(np.diag(Phi))
    tmp = np.linalg.solve(L.T, Phi.T).T
    K_bar = np.linalg.solve(L.T, tmp)
    return 0.5 * (K_bar + K_bar.T)


class SparseGP(ProbMapModel):
    """
    Collapsed sparse variational GP (VFE / DTC) with LQ numerics.

    Whitened features ``A = σ⁻¹ L_m⁻¹ K_{mn}`` with ``L_m = chol(K_{mm})``.
    The bound uses LQ of ``[I, A]`` (optionally y-augmented for the quadratic)
    instead of Cholesky of ``I + AAᵀ``.

    :param X: Training inputs (N×D)
    :param y: Targets (N,) or (N×1)
    :param Z: Inducing inputs (M×D)
    :param sigma2: Observation noise variance
    :param kernel: Covariance function (:class:`~mlai.gaussian_processes.Kernel`)
    :param approximation: ``'vfe'`` (default) or ``'dtc'``
    :param latent_X: If True, rows of ``X`` join the shared parameter vector
    :param optimize_Z: If False, inducing inputs ``Z`` stay fixed (excluded
        from :attr:`parameters` / :attr:`gradients`)
    :param jitter: Diagonal jitter on ``K_{mm}`` (default ``1e-8``, matching
        GPy ``VarDTC.const_jitter``)
    """

    def __init__(
        self,
        X,
        y,
        Z,
        sigma2,
        kernel,
        approximation='vfe',
        latent_X=False,
        optimize_Z=True,
        jitter=1e-8,
    ):
        if approximation not in ('vfe', 'dtc'):
            raise ValueError("approximation must be 'vfe' or 'dtc'")
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).ravel()
        self.Z = np.asarray(Z, dtype=float).copy()
        self.sigma2 = float(sigma2)
        self.kernel = kernel
        self.approximation = approximation
        self.latent_X = bool(latent_X)
        self.optimize_Z = bool(optimize_Z)
        self.jitter = float(jitter)
        self._materialize_kernel_defaults()
        self.name = f'SparseGP_{approximation}_{kernel.function.__name__}'
        self.objective_name = 'Negative Variational Bound'
        self._update_bound()

    # ------------------------------------------------------------------
    # Kernel defaults (same idea as exact GP)
    # ------------------------------------------------------------------
    def _materialize_kernel_defaults(self):
        try:
            signature = inspect.signature(self.kernel.function)
        except (TypeError, ValueError):
            return
        for name, param in signature.parameters.items():
            if name in ('x', 'x_prime'):
                continue
            if name in self.kernel.parameters:
                continue
            if param.default is inspect.Parameter.empty:
                continue
            if np.isscalar(param.default):
                self.kernel.parameters[name] = float(param.default)

    def _scalar_kernel_param_names(self):
        names = []
        for key, value in self.kernel.parameters.items():
            if np.isscalar(value) or (
                isinstance(value, np.ndarray) and value.shape == ()
            ):
                names.append(key)
        return names

    # ------------------------------------------------------------------
    # Kernels and whitening
    # ------------------------------------------------------------------
    def _kernel_blocks(self):
        Kmm = self.kernel.K(self.Z, self.Z)
        Kmm = Kmm + self.jitter * np.eye(Kmm.shape[0])
        Kmn = self.kernel.K(self.Z, self.X)
        diag_Knn = self.kernel.diag(self.X).ravel()
        return Kmm, Kmn, diag_Knn

    def _whiten(self, Kmm, Kmn):
        Lm = np.linalg.cholesky(Kmm)
        # A = σ⁻¹ Lm^{-1} Kmn
        A = np.linalg.solve(Lm, Kmn) / np.sqrt(self.sigma2)
        return Lm, A

    def _update_bound(self):
        """Recompute cached bound pieces from current parameters."""
        Kmm, Kmn, diag_Knn = self._kernel_blocks()
        Lm, A = self._whiten(Kmm, Kmn)
        m, n = A.shape
        sigma = np.sqrt(self.sigma2)
        ys = self.y / sigma

        W = np.hstack([np.eye(m), A])
        L, Q = lq_factor(W)
        logdet = 2.0 * np.sum(np.log(np.abs(np.diag(L))))

        # Woodbury quadratic via triangular solve (same as y-augmented LQ r²)
        c = np.linalg.solve(L, A @ ys)
        r2 = float(ys @ ys - c @ c)

        froA = float(np.linalg.norm(A, 'fro') ** 2)
        trKnn = float(np.sum(diag_Knn))

        # ELBO F (up to teaching constants); objective = -F
        nll_gauss = (
            0.5 * n * np.log(2.0 * np.pi)
            + 0.5 * n * np.log(self.sigma2)
            + 0.5 * logdet
            + 0.5 * r2
        )
        if self.approximation == 'vfe':
            # -F = nll_gauss + (1/(2σ²))tr Knn - (1/2)||A||_F²
            neg_bound = nll_gauss + 0.5 * trKnn / self.sigma2 - 0.5 * froA
        else:
            neg_bound = nll_gauss

        self.Kmm = Kmm
        self.Kmn = Kmn
        self.diag_Knn = diag_Knn
        self.Lm = Lm
        self.A = A
        self.L = L
        self.Q = Q
        self.logdet = logdet
        self.r2 = r2
        self.c = c
        self.froA = froA
        self.trKnn = trKnn
        self._neg_bound = float(neg_bound)

    # ------------------------------------------------------------------
    # Shared optimiser surface (Paramz-lite)
    # ------------------------------------------------------------------
    def _kernel_display_name(self):
        if getattr(self.kernel, 'name', None):
            return str(self.kernel.name)
        if getattr(self.kernel, 'shortname', None):
            return str(self.kernel.shortname)
        fname = getattr(self.kernel.function, '__name__', 'kern')
        return fname.replace('_cov', '').replace('exponentiated_quadratic', 'rbf')

    def _param_layout(self):
        """Describe flat unconstrained parameter blocks."""
        blocks = [('sigma2', 1)]
        for name in self._scalar_kernel_param_names():
            blocks.append((f'kernel:{name}', 1))
        if self.optimize_Z:
            blocks.append(('Z', self.Z.size))
        if self.latent_X:
            blocks.append(('X', self.X.size))
        return blocks

    def _expected_parameter_size(self):
        return self.param_group().size

    def param_group(self):
        """
        Paramz-lite group: ``+ve`` noise/kernel; unconstrained ``Z`` / ``X``.

        Natural attributes remain the source of truth; the group is built on
        demand for packing, gradient scaling, and display.
        """
        root = 'sparse_gp'
        kern = self._kernel_display_name()
        group = ParamGroup(name=root)
        group.add(
            Param('sigma2', self.sigma2, transform=POSITIVE, path=f'{root}.sigma2')
        )
        for name in self._scalar_kernel_param_names():
            group.add(
                Param(
                    name,
                    float(self.kernel.parameters[name]),
                    transform=POSITIVE,
                    path=f'{root}.{kern}.{name}',
                )
            )
        if self.optimize_Z:
            group.add(Param('Z', self.Z, transform=None, path=f'{root}.Z'))
        if self.latent_X:
            group.add(Param('X', self.X, transform=None, path=f'{root}.X'))
        return group

    @property
    def parameters(self):
        """
        Flat parameter vector for shared optimisers.

        Layout: ``[log σ², log θ…, optional vec(Z), optional vec(X)]``.
        """
        return self.param_group().pack()

    @parameters.setter
    def parameters(self, value):
        group = self.param_group()
        values = group.unpack(value)
        idx = 0
        self.sigma2 = float(values[idx])
        idx += 1
        for name in self._scalar_kernel_param_names():
            self.kernel.parameters[name] = float(values[idx])
            idx += 1
        if self.optimize_Z:
            self.Z = np.asarray(values[idx], dtype=float).reshape(self.Z.shape).copy()
            idx += 1
        if self.latent_X:
            self.X = np.asarray(values[idx], dtype=float).reshape(self.X.shape).copy()
        self._update_bound()

    def objective(self):
        """Negative variational bound (minimise)."""
        return self._neg_bound

    def log_likelihood(self):
        """Variational bound ``F = -objective()`` (ELBO-style)."""
        return -self._neg_bound

    # ------------------------------------------------------------------
    # Gradients (LQ reverse + whitening; FD kernel / Z / X via ∂K)
    # ------------------------------------------------------------------
    def _A_cotangent(self):
        """
        Cotangent of ``objective`` w.r.t. whitened features ``A``.

        Seeds the LQ adjoint from ``½ log|I+AAᵀ|`` and ``½ r²``, then adds
        the VFE ``-½||A||_F²`` term outside the LQ.
        """
        m = self.A.shape[0]
        sigma = np.sqrt(self.sigma2)
        ys = self.y / sigma
        L, c = self.L, self.c

        # ½ logdet = Σ log L_ii  →  L_bar_ii += 1/L_ii
        L_bar = np.zeros_like(L)
        np.fill_diagonal(L_bar, 1.0 / np.diag(L))

        # ½ r² = ½||ys||² - ½||c||², c = L^{-1} v, v = A ys
        # bar_c = -c; u = L^{-T} bar_c; bar_L += -u cᵀ; bar_v = u
        u = np.linalg.solve(L.T, -c)
        L_bar = L_bar - np.outer(u, c)
        v_bar = u

        W_bar = lq_adjoint(L, self.Q, L_bar, None)
        A_bar = W_bar[:, m:] + np.outer(v_bar, ys)
        if self.approximation == 'vfe':
            A_bar = A_bar - self.A
        return A_bar, v_bar, ys

    def _reverse_to_kernel_blocks(self, A_bar):
        """Map ``A_bar`` through whitening to ``Kmm`` / ``Kmn`` cotangents."""
        Lm, A = self.Lm, self.A
        sigma = np.sqrt(self.sigma2)
        B = A * sigma
        B_bar = A_bar / sigma
        Kmn_bar = np.linalg.solve(Lm.T, B_bar)
        Lm_bar = -Kmn_bar @ B.T
        Kmm_bar = _chol_lower_adjoint(Lm, Lm_bar)
        return Kmm_bar, Kmn_bar

    def _nll_gradient_natural(self):
        """Gradients of objective w.r.t. natural-scale parameter blocks."""
        self._update_bound()
        A_bar, v_bar, ys = self._A_cotangent()
        Kmm_bar, Kmn_bar = self._reverse_to_kernel_blocks(A_bar)

        grads = []
        n = self.y.size

        # σ²: explicit log term, ys-scaling, A∝σ⁻¹, optional VFE trace
        ys_bar = self.A.T @ v_bar + ys
        d_sigma2 = (
            0.5 * n / self.sigma2
            + float(ys_bar @ (-0.5 * ys / self.sigma2))
            + float(np.sum(A_bar * (-0.5 * self.A / self.sigma2)))
        )
        if self.approximation == 'vfe':
            d_sigma2 += -0.5 * self.trKnn / (self.sigma2 ** 2)
        grads.append(d_sigma2)

        h = 1e-5
        for name in self._scalar_kernel_param_names():
            theta = float(self.kernel.parameters[name])
            step = h * max(abs(theta), 1.0)
            self.kernel.parameters[name] = theta + step
            Kmm_p, Kmn_p, diag_p = self._kernel_blocks()
            self.kernel.parameters[name] = theta - step
            Kmm_m, Kmn_m, diag_m = self._kernel_blocks()
            self.kernel.parameters[name] = theta
            dKmm = (Kmm_p - Kmm_m) / (2.0 * step)
            dKmn = (Kmn_p - Kmn_m) / (2.0 * step)
            ddiag = (diag_p - diag_m) / (2.0 * step)
            g = float(np.sum(Kmm_bar * dKmm) + np.sum(Kmn_bar * dKmn))
            if self.approximation == 'vfe':
                g += 0.5 * float(np.sum(ddiag)) / self.sigma2
            grads.append(g)

        if self.optimize_Z:
            Z_grad = np.zeros_like(self.Z)
            for i in range(self.Z.shape[0]):
                for j in range(self.Z.shape[1]):
                    step = h * max(abs(self.Z[i, j]), 1.0)
                    self.Z[i, j] += step
                    Kmm_p, Kmn_p, _ = self._kernel_blocks()
                    self.Z[i, j] -= 2.0 * step
                    Kmm_m, Kmn_m, _ = self._kernel_blocks()
                    self.Z[i, j] += step
                    dKmm = (Kmm_p - Kmm_m) / (2.0 * step)
                    dKmn = (Kmn_p - Kmn_m) / (2.0 * step)
                    Z_grad[i, j] = float(
                        np.sum(Kmm_bar * dKmm) + np.sum(Kmn_bar * dKmn)
                    )
            grads.extend(Z_grad.ravel().tolist())

        if self.latent_X:
            X_grad = np.zeros_like(self.X)
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[1]):
                    step = h * max(abs(self.X[i, j]), 1.0)
                    self.X[i, j] += step
                    _, Kmn_p, diag_p = self._kernel_blocks()
                    self.X[i, j] -= 2.0 * step
                    _, Kmn_m, diag_m = self._kernel_blocks()
                    self.X[i, j] += step
                    dKmn = (Kmn_p - Kmn_m) / (2.0 * step)
                    ddiag = (diag_p - diag_m) / (2.0 * step)
                    g = float(np.sum(Kmn_bar * dKmn))
                    if self.approximation == 'vfe':
                        g += 0.5 * float(np.sum(ddiag)) / self.sigma2
                    X_grad[i, j] = g
            grads.extend(X_grad.ravel().tolist())

        return np.asarray(grads, dtype=float)

    @property
    def gradients(self):
        """
        Gradients of :meth:`objective` w.r.t. :attr:`parameters`.

        Paramz-lite ``+ve`` chain rule for ``σ²`` and kernel hyperparameters;
        identity map for ``Z`` (and ``X`` when ``latent_X``).
        """
        return self.param_group().scale_natural_gradients(self._nll_gradient_natural())

    def __repr__(self):
        return repr(self.param_group())

    def _repr_html_(self):
        return self.param_group()._repr_html_()

    # ------------------------------------------------------------------
    # Prediction (DTC-style posterior at inducing)
    # ------------------------------------------------------------------
    def predict(self, X_test):
        """
        Predictive mean and marginal variance (DTC/VFE inducing posterior).

        Latent-function posterior (no observation noise on the returned
        variance), matching GPy ``predict(..., include_likelihood=False)``.

        With whitening ``A = σ⁻¹ L_m⁻¹ K_{mn}`` and ``L Lᵀ = I+AAᵀ``:

        .. math::

            \\mu_* = W_{*}^{\\top}(I+AA^{\\top})^{-1} A y/\\sigma,
            \\quad
            W_* = L_m^{-1} K_{m*}

        :param X_test: Test inputs
        :returns: ``(mean, var)`` each shape ``(N*,)``
        """
        self._update_bound()
        X_test = np.asarray(X_test, dtype=float)
        Kms = self.kernel.K(self.Z, X_test)
        Kss_diag = self.kernel.diag(X_test).ravel()
        L = self.L
        ys = self.y / np.sqrt(self.sigma2)
        # α = (I+AAᵀ)^{-1} A ys; μ* = Wsmᵀ α  (σ already absorbed in A, ys)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.A @ ys))
        Wsm = np.linalg.solve(self.Lm, Kms)
        mu = Wsm.T @ alpha
        tmp = np.linalg.solve(L, Wsm)
        var = Kss_diag - np.sum(Wsm * Wsm, axis=0) + np.sum(tmp * tmp, axis=0)
        var = np.maximum(var, 0.0)
        return mu, var

    def fit(self):
        """No-op placeholder (use shared optimisers on ``parameters``)."""
        pass

    # ------------------------------------------------------------------
    # Gram+chol reference (tests only)
    # ------------------------------------------------------------------
    def gram_chol_neg_bound(self):
        """Negative bound using Cholesky of ``I+AAᵀ`` (parity reference)."""
        Kmm, Kmn, diag_Knn = self._kernel_blocks()
        _, A = self._whiten(Kmm, Kmn)
        m, n = A.shape
        C = np.eye(m) + A @ A.T
        Lc = np.linalg.cholesky(C)
        logdet = 2.0 * np.sum(np.log(np.abs(np.diag(Lc))))
        ys = self.y / np.sqrt(self.sigma2)
        c = np.linalg.solve(Lc, A @ ys)
        r2 = float(ys @ ys - c @ c)
        froA = float(np.linalg.norm(A, 'fro') ** 2)
        nll_gauss = (
            0.5 * n * np.log(2.0 * np.pi)
            + 0.5 * n * np.log(self.sigma2)
            + 0.5 * logdet
            + 0.5 * r2
        )
        if self.approximation == 'vfe':
            return float(nll_gauss + 0.5 * np.sum(diag_Knn) / self.sigma2 - 0.5 * froA)
        return float(nll_gauss)
