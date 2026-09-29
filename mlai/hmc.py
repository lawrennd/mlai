"""
Hamiltonian Monte Carlo (hybrid Monte Carlo)

A small, readable HMC sampler for teaching, following Radford Neal's
hybrid Monte Carlo formulation for neural networks.

References
----------
Neal, R. M. (1992). *Bayesian training of backpropagation networks by the
hybrid Monte Carlo method*. CRG-TR-92-1.

Neal, R. M. (1994). *Bayesian Learning for Neural Networks*. PhD thesis.

The lecture path loss → potential → kinetic → Hamiltonian maps onto this
module as:

* ``potential(q)`` — potential energy :math:`V(q)` (often a training loss
  or :math:`-\\log p(q\\mid\\mathcal{D})`)
* ``kinetic(p)`` — kinetic energy :math:`K(p)=\\frac12 p^\\top M^{-1} p`
* ``hamiltonian(q, p) = potential(q) + kinetic(p)``

Dynamics use leapfrog integration; proposals are accepted or rejected with
a Metropolis step on :math:`\\Delta H`. This is deliberately not NUTS /
adaptive HMC — clarity over machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np

__all__ = [
    'HMCResult',
    'HamiltonianMonteCarlo',
    'leapfrog',
    'kinetic_energy',
]


ArrayLike = Union[np.ndarray, Sequence[float]]
PotentialFn = Callable[[np.ndarray], float]
GradPotentialFn = Callable[[np.ndarray], np.ndarray]


def kinetic_energy(p: np.ndarray, mass: np.ndarray) -> float:
    """
    Kinetic energy :math:`K(p)=\\frac12 p^\\top M^{-1} p` for diagonal mass.

    :param p: Momentum vector
    :type p: numpy.ndarray
    :param mass: Diagonal mass entries (same shape as ``p``)
    :type mass: numpy.ndarray
    :returns: Scalar kinetic energy
    :rtype: float
    """
    return 0.5 * float(np.sum((p ** 2) / mass))


def leapfrog(
    q: np.ndarray,
    p: np.ndarray,
    grad_potential: GradPotentialFn,
    step_size: float,
    n_steps: int,
    mass: np.ndarray,
    record_path: bool = False,
) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Leapfrog (velocity Verlet) integrator for Hamiltonian dynamics.

    Half-step momentum, full-step position, half-step momentum, repeated
    ``n_steps`` times. Mass is diagonal: position updates use ``p / mass``.

    :param q: Position
    :type q: numpy.ndarray
    :param p: Momentum
    :type p: numpy.ndarray
    :param grad_potential: Callable returning :math:`\\nabla V(q)`
    :type grad_potential: callable
    :param step_size: Leapfrog step size :math:`\\varepsilon`
    :type step_size: float
    :param n_steps: Number of leapfrog steps ``L``
    :type n_steps: int
    :param mass: Diagonal mass entries
    :type mass: numpy.ndarray
    :param record_path: If True, also return positions after each full step
        including the start (shape ``(n_steps+1, dim)``)
    :type record_path: bool
    :returns: Proposed ``(q, p)``, or ``(q, p, path)`` when ``record_path``
    """
    q = np.array(q, dtype=float, copy=True)
    p = np.array(p, dtype=float, copy=True)
    path = [q.copy()] if record_path else None

    # Half step for momentum
    p = p - 0.5 * step_size * np.asarray(grad_potential(q), dtype=float)

    for _ in range(n_steps - 1):
        q = q + step_size * (p / mass)
        if path is not None:
            path.append(q.copy())
        p = p - step_size * np.asarray(grad_potential(q), dtype=float)

    # Final full step for position and closing half step for momentum
    q = q + step_size * (p / mass)
    if path is not None:
        path.append(q.copy())
    p = p - 0.5 * step_size * np.asarray(grad_potential(q), dtype=float)

    if path is not None:
        return q, p, np.asarray(path)
    return q, p


@dataclass
class HMCResult:
    """
    Output of an HMC run.

    :param samples: Chain of accepted positions, shape ``(n_samples, dim)``
    :param accept_rate: Fraction of proposals accepted
    :param hamiltonian_trace: Optional :math:`H` after each proposal (accepted
        or rejected current state), length ``n_samples``
    :param n_accepted: Number of accepted proposals
    """

    samples: np.ndarray
    accept_rate: float
    hamiltonian_trace: Optional[np.ndarray] = None
    n_accepted: int = 0


class HamiltonianMonteCarlo:
    """
    Teachable Hamiltonian Monte Carlo with diagonal mass.

    Each iteration draws momentum from :math:`\\mathcal{N}(0, M)`, integrates
    leapfrog dynamics for ``n_steps`` of size ``step_size``, then accepts or
    rejects with Metropolis probability :math:`\\min(1, e^{-\\Delta H})`.

    Parameters
    ----------
    potential : callable
        ``potential(q) -> float``, the potential energy :math:`V(q)`.
    grad_potential : callable
        ``grad_potential(q) -> array``, gradient :math:`\\nabla V(q)`.
    step_size : float, optional
        Leapfrog step size :math:`\\varepsilon` (default ``0.1``).
    n_steps : int, optional
        Leapfrog steps per proposal ``L`` (default ``10``).
    mass : float or array, optional
        Diagonal mass. Scalar broadcasts to all dimensions (default ``1.0``).

    Examples
    --------
    Standard normal target :math:`V(q)=\\frac12\\|q\\|^2`:

    >>> hmc = HamiltonianMonteCarlo(
    ...     potential=lambda q: 0.5 * np.dot(q, q),
    ...     grad_potential=lambda q: q,
    ...     step_size=0.1,
    ...     n_steps=10,
    ... )
    >>> result = hmc.sample(q0=np.zeros(2), n_samples=1000, random_state=0)
    >>> result.accept_rate  # doctest: +SKIP
    0.9...
    """

    def __init__(
        self,
        potential: PotentialFn,
        grad_potential: GradPotentialFn,
        step_size: float = 0.1,
        n_steps: int = 10,
        mass: Union[float, ArrayLike] = 1.0,
    ):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")

        self.potential = potential
        self.grad_potential = grad_potential
        self.step_size = float(step_size)
        self.n_steps = int(n_steps)
        self._mass_spec = mass

    def _mass_vector(self, dim: int) -> np.ndarray:
        mass = np.asarray(self._mass_spec, dtype=float)
        if mass.ndim == 0:
            mass = np.full(dim, float(mass))
        if mass.shape != (dim,):
            raise ValueError(
                f"mass must be scalar or length-{dim} vector, got shape {mass.shape}"
            )
        if np.any(mass <= 0):
            raise ValueError("mass entries must be positive")
        return mass

    def kinetic(self, p: np.ndarray, mass: Optional[np.ndarray] = None) -> float:
        """Kinetic energy for momentum ``p``."""
        if mass is None:
            mass = self._mass_vector(p.shape[0])
        return kinetic_energy(p, mass)

    def hamiltonian(self, q: np.ndarray, p: np.ndarray, mass: Optional[np.ndarray] = None) -> float:
        """Total energy :math:`H = V(q) + K(p)`."""
        if mass is None:
            mass = self._mass_vector(np.asarray(q).shape[0])
        return float(self.potential(q)) + self.kinetic(p, mass)

    def sample(
        self,
        q0: ArrayLike,
        n_samples: int,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        store_hamiltonian: bool = True,
        return_trajectory: bool = False,
    ) -> Union[HMCResult, tuple[HMCResult, list[np.ndarray]]]:
        """
        Draw ``n_samples`` HMC samples starting from ``q0``.

        :param q0: Initial position
        :type q0: array-like
        :param n_samples: Number of samples to store (after each proposal)
        :type n_samples: int
        :param random_state: Seed or NumPy ``Generator``
        :type random_state: int or numpy.random.Generator, optional
        :param store_hamiltonian: If True, record :math:`H` of the current
            state after each iteration
        :type store_hamiltonian: bool
        :param return_trajectory: If True, also return per-iteration leapfrog
            position paths (list of arrays shape ``(n_steps+1, dim)``)
        :type return_trajectory: bool
        :returns: ``HMCResult``, or ``(HMCResult, trajectories)`` when
            ``return_trajectory`` is True
        """
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")

        q = np.asarray(q0, dtype=float).reshape(-1).copy()
        dim = q.shape[0]
        mass = self._mass_vector(dim)

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        samples = np.empty((n_samples, dim), dtype=float)
        h_trace = np.empty(n_samples, dtype=float) if store_hamiltonian else None
        trajectories: list[np.ndarray] = []
        n_accepted = 0

        for i in range(n_samples):
            p = rng.normal(0.0, np.sqrt(mass))
            h_current = self.hamiltonian(q, p, mass)

            if return_trajectory:
                q_prop, p_prop, path = leapfrog(
                    q,
                    p,
                    self.grad_potential,
                    self.step_size,
                    self.n_steps,
                    mass,
                    record_path=True,
                )
                trajectories.append(path)
            else:
                q_prop, p_prop = leapfrog(
                    q, p, self.grad_potential, self.step_size, self.n_steps, mass
                )

            h_prop = self.hamiltonian(q_prop, p_prop, mass)
            delta_h = h_prop - h_current
            # Metropolis: accept with probability min(1, exp(-ΔH))
            if delta_h < 0.0 or rng.random() < np.exp(-delta_h):
                q = q_prop
                n_accepted += 1
                h_current = h_prop

            samples[i] = q
            if h_trace is not None:
                h_trace[i] = h_current

        result = HMCResult(
            samples=samples,
            accept_rate=n_accepted / n_samples,
            hamiltonian_trace=h_trace,
            n_accepted=n_accepted,
        )
        if return_trajectory:
            return result, trajectories
        return result
